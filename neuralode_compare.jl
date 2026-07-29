using OrdinaryDiffEq
using OrdinaryDiffEqTsit5: Tsit5
using OrdinaryDiffEqRosenbrock: Rodas5
using OrdinaryDiffEqSDIRK: Kvaerno5
using BoundaryValueDiffEqMIRK
using Optimization, OptimizationOptimisers, OptimizationAuglag
using NonlinearSolve
using Lux, Random, ComponentArrays
using SciMLBase: successful_retcode
using ADTypes, SparseConnectivityTracer, SparseMatrixColorings
using LinearAlgebra, Statistics, DelimitedFiles
import SciMLLogging as SL
using Plots

include("src/layers/StiffLayer.jl")
include("problems.jl")

const PROBLEM     = "vanderpol"
const PROFILE     = :fast
const HIDDEN      = 8
const DEPTH       = 4
const SHOOT_ITERS = 500
const SEED        = 123
const SEEDS       = 1:1
const SWEEP_AL    = false # skip augmented lagrangian or not for speed
const SWEEP_SHOOT = false # skip shooting for speed rober

spec   = make_problem(PROBLEM; T = Float64, profile = PROFILE)
tsteps = spec.tsteps
tspan  = spec.tspan
u0     = spec.u0
d      = length(u0)
Ydata  = generate_training_data(spec)
tlen   = length(tsteps)
println("problem $(spec.name): d=$d  tlen=$tlen  tspan=$tspan  solver=$(typeof(spec.solver).name.name)")

rng = Xoshiro(SEED)
_layers = Any[Dense(d, HIDDEN, gelu)]
for _ in 2:DEPTH; push!(_layers, Dense(HIDDEN, HIDDEN, gelu)); end
push!(_layers, Dense(HIDDEN, d))
stdModel = Chain(_layers...)

#########

V = Any[Dense(d, HIDDEN, gelu)]
for _ in 2:Int(DEPTH/2); push!(V, Dense(HIDDEN, HIDDEN, gelu)); end

sigma_affine_sigmoid(ps, j) = 1 / (1 + exp(-(ps[1] * j + ps[2])))
S = StiffLayer2(sigma_affine_sigmoid; P=HIDDEN, init=ones(HIDDEN))

U = Any[Dense(HIDDEN, HIDDEN, gelu)]
for _ in 2:Int(DEPTH/2); push!(U, Dense(HIDDEN, HIDDEN, gelu)); end
push!(U, Dense(HIDDEN, d))

stiffModel = Chain(V..., S, U...)
#######

model = stiffModel
ps, st   = Lux.setup(rng, model); ps = Lux.f64(ps)
ps_axes  = getaxes(ComponentVector(ps))
N_params = length(ComponentVector(ps))

# equation scaling (Stiff Neural Ordinary Differential Equations 2021)
const YMIN   = vec(minimum(Ydata, dims = 2))
const YMAX   = vec(maximum(Ydata, dims = 2))
const YSCALE = max.(YMAX .- YMIN, eps())
const YMID   = (YMAX .+ YMIN) ./ 2
const DSCALE = let dY = diff(Ydata, dims = 2) ./ diff(tsteps)'
    raw = vec(sqrt.(sum(abs2, dY, dims = 2) ./ size(dY, 2)))
    max.(raw, 1e-6 * maximum(raw))
end

# i added input scaling as well (suggestion from Stiff NODEs 2021)
f_theta(u, θ) = model((u .- YMID) ./ YSCALE, ComponentVector(θ, ps_axes), st)[1] .* DSCALE
f_theta!(du, u, θ, t) = (du .= f_theta(u, θ); nothing)
f_theta_bvp!(du, u, p, t) = (du[1:d] .= f_theta(@view(u[1:d]), p); nothing)

function true_field(X)
    D = similar(X); tmp = zeros(d)
    for j in axes(X, 2); spec.true_ode!(tmp, X[:, j], nothing, 0.0); D[:, j] = tmp; end
    D
end

# representation upper bound I just added for referencem, uses true rhs
sup_derivs = true_field(Ydata)
function supR!(r, θ, _)
    k = 0
    @inbounds for j in 1:tlen
        fp = f_theta(Ydata[:, j], θ)
        for c in 1:d; r[k+c] = (fp[c] - sup_derivs[c, j]) / DSCALE[c]; end
        k += d
    end
    return nothing
end
function train_supervised(θ0)
    try
        nf = NonlinearFunction(supR!; resid_prototype = zeros(d * tlen))
        sol = solve(NonlinearLeastSquaresProblem(nf, copy(θ0)), LevenbergMarquardt(); maxiters = 1000)
        return sol.u
    catch e
        @warn "Supervised fit failed" exception=(e,); return nothing
    end
end

const RADAU_A = [5/12 -1/12; 3/4 1/4]
const RADAU_B = [3/4, 1/4]
const SSTAGE  = 2

const REF_INIT   = 16   
const NSUB_CAP   = 48  
const AMR_ROUNDS = 5   
const AMR_FRAC   = 0.4 

function build_mesh(nsub)
    mesh = Float64[]; nob = Int[]
    for j in 1:tlen-1
        t0, t1 = tsteps[j], tsteps[j+1]
        push!(nob, length(mesh) + 1)
        for kk in 0:nsub[j]-1; push!(mesh, t0 + (t1 - t0) * kk / nsub[j]); end
    end
    push!(mesh, tsteps[end]); push!(nob, length(mesh))
    return mesh, nob
end

function interp_data_onto(mesh)
    out = similar(Ydata, d, length(mesh))
    for (q, t) in enumerate(mesh)
        if t <= tsteps[1];        out[:, q] = Ydata[:, 1]
        elseif t >= tsteps[end];  out[:, q] = Ydata[:, end]
        else
            kk = searchsortedlast(tsteps, t)
            α = (t - tsteps[kk]) / (tsteps[kk+1] - tsteps[kk])
            out[:, q] = (1 - α) .* Ydata[:, kk] .+ α .* Ydata[:, kk+1]
        end
    end
    return out
end

function R!(res, z, p)
    Y = z.Y; K = z.K; θ = z.theta
    msh = p.mesh; A = p.A; b = p.b; s = p.s; dd = p.d; Yob = p.Y_obs; nob = p.nob; M = p.M
    k = 0
    @inbounds for i in 1:M-1
        h = msh[i+1] - msh[i]
        for j in 1:s
            stage = copy(view(Y, :, i))
            for l in 1:s; stage = stage .+ (h * A[j, l]) .* view(K, :, i, l); end
            fj = f_theta(stage, θ)
            for c in 1:dd; res[k+c] = (K[c, i, j] - fj[c]) / DSCALE[c]; end
            k += dd
        end
        for c in 1:dd
            acc = Y[c, i+1] - Y[c, i]
            for j in 1:s; acc -= h * b[j] * K[c, i, j]; end
            res[k+c] = acc / YSCALE[c]
        end
        k += dd
    end
    @inbounds for j in 1:tlen
        n = nob[j]
        for c in 1:dd; res[k+c] = (Y[c, n] - Yob[c, j]) / YSCALE[c]; end
        k += dd
    end
    return nothing
end

function solve_collocation(nsub, θ0)
    mesh, nob = build_mesh(nsub); M = length(mesh)
    nres = (M - 1) * SSTAGE * d + (M - 1) * d + tlen * d
    Y0 = interp_data_onto(mesh)
    K0 = zeros(d, M - 1, SSTAGE)
    for i in 1:M-1
        slope = (Y0[:, i+1] .- Y0[:, i]) ./ (mesh[i+1] - mesh[i])
        for j in 1:SSTAGE; K0[:, i, j] = slope; end
    end
    z0 = ComponentVector(Y = Y0, K = K0, theta = copy(θ0))
    dp = (mesh = mesh, A = RADAU_A, b = RADAU_B, s = SSTAGE, d = d, Y_obs = Ydata, nob = nob, M = M)
    nf = NonlinearFunction(R!; resid_prototype = zeros(nres), sparsity = TracerSparsityDetector())
    sol = solve(NonlinearLeastSquaresProblem(nf, z0, dp), LevenbergMarquardt(); maxiters = 500)
    return sol.u, dp
end

function interval_errors(z, dp)
    M = dp.M; nob = dp.nob
    res = zeros((M - 1) * SSTAGE * d + (M - 1) * d + tlen * d); R!(res, z, dp)
    rowsper = SSTAGE * d + d
    err = zeros(tlen - 1)
    for j in 1:tlen-1, i in nob[j]:nob[j+1]-1, r in 1:rowsper
        err[j] += res[(i - 1) * rowsper + r]^2
    end
    return err
end

function train_collocation(θ0)
    speed = vec(sqrt.(sum(abs2, diff(Ydata, dims = 2), dims = 1))) ./ diff(tsteps)
    nsub  = clamp.(round.(Int, 2 .+ (REF_INIT - 2) .* speed ./ maximum(speed)), 2, REF_INIT)  # speed-based start
    θ = copy(θ0)
    try
        for rnd in 1:AMR_ROUNDS
            z, dp = solve_collocation(nsub, θ)
            θ = collect(z.theta)
            err = interval_errors(z, dp); me = maximum(err)
            refined = 0
            if rnd < AMR_ROUNDS
                thr = AMR_FRAC * me
                for j in 1:tlen-1
                    if err[j] > thr && nsub[j] < NSUB_CAP
                        nsub[j] = min(nsub[j] * 2, NSUB_CAP); refined += 1
                    end
                end
            end
            println("  AMR round $rnd: M=$(dp.M)  max-interval-err=$(round(me; sigdigits = 3))  refined=$refined"); flush(stdout)
            refined == 0 && break
        end
        return θ
    catch e
        @warn "AMR collocation failed" exception=(e,); return nothing
    end
end

# shooting
function shooting_loss(θ, _)
    prob = ODEProblem(f_theta!, u0, tspan, θ)
    sol = solve(prob, spec.solver; saveat = tsteps, verbose = SL.None(), spec.solve_kwargs...)
    successful_retcode(sol) || return convert(eltype(θ), 1e6)
    return sum(abs2, (Array(sol) .- Ydata) ./ YSCALE)
end
function train_shooting(θ0)
    try
        optf = OptimizationFunction(shooting_loss, Optimization.AutoForwardDiff())
        res = solve(OptimizationProblem(optf, copy(θ0)), OptimizationOptimisers.Adam(0.01); maxiters = SHOOT_ITERS)
        return res.u
    catch e
        @warn "Shooting failed" exception=(e,); return nothing
    end
end

# augmented-Lagrangian BVP baseline with bounded iterations
al_nodes = PROBLEM == "spiral" ? clamp(2 * tlen, 40, 120) : 60
al_dt    = (tspan[2] - tspan[1]) / al_nodes
al_inner = PROBLEM == "spiral" ? 100 : 25
al_outer = PROBLEM == "spiral" ? 50 : 12
function train_al(θ0)
    try
        cost = (sol, p) -> begin
            acc = zero(eltype(sol(tsteps[2])[1]))
            for j in 1:tlen
                j == 1 && continue
                acc += sum(abs2, (sol(tsteps[j])[1:d] .- Ydata[:, j]) ./ YSCALE)
            end
            acc
        end
        bc!(res, sol, p, t) = (res .= sol(tspan[1])[1:d] .- u0; nothing)
        bvp_fun = BVPFunction(f_theta_bvp!, bc!; bcresid_prototype = zeros(d), cost = cost)
        bvp = BVProblem(bvp_fun, u0, tspan, copy(θ0); tune_parameters = true)
        al = OptimizationAuglag.AugLag(; inner = OptimizationOptimisers.Adam(1e-2),
            inner_kwargs = (maxiters = al_inner,), γ = 2.0, λmin = -1e6, λmax = 1e6,
            μmin = 0.0, μmax = 1e6, ρ_init = 10.0, ϵ_primal = 1e-2, ϵ_dual = 1e6)
        al_sol = solve(bvp, MIRK4(; optimize = al); dt = al_dt, adaptive = false,
            optimize_kwargs = (; maxiters = al_outer))
        return al_sol.prob.p
    catch e
        @warn "AL failed" exception=(e,); return nothing
    end
end

# metrics
# trajectory error: integrate f_θ from u0, L2 distance to the true trajectory
function traj_l2(θ)
    θ === nothing && return nothing
    sol = solve(ODEProblem(f_theta!, u0, tspan, θ), spec.solver; saveat = tsteps, verbose = SL.None(), spec.solve_kwargs...)
    successful_retcode(sol) || return Inf
    return sqrt(sum(abs2, Array(sol) .- Ydata))
end

# velocity field error: per-species relative RMS of f_θ vs the true derivatives, averaged
function vf_error(θ)
    θ === nothing && return nothing
    pred = reduce(hcat, (f_theta(Ydata[:, j], θ) for j in 1:tlen))
    rel  = sqrt.(sum(abs2, pred .- sup_derivs, dims = 2) ./ max.(sum(abs2, sup_derivs, dims = 2), eps()))
    return mean(rel)
end

trained = Dict{String,Any}()   
t = @elapsed (trained["collocation"] = train_collocation(0.1 .* randn(Xoshiro(SEED), N_params)))
println("collocation trained [$(round(Int, t))s]"); flush(stdout)

# plot
let
    logt  = all(>(0), tsteps) && (tspan[2] / max(tspan[1], eps()) > 100)
    tgrid = logt ? exp.(range(log(tsteps[1]), log(tsteps[end]); length = 400)) :
                   collect(range(tspan[1], tspan[2]; length = 400))
    θ = get(trained, "collocation", nothing)
    s = θ === nothing ? nothing :
        solve(ODEProblem(f_theta!, u0, tspan, θ), spec.solver; saveat = tgrid, verbose = SL.None(), spec.solve_kwargs...)
    ok   = s !== nothing && successful_retcode(s)
    pt   = ok ? s.t : Float64[]
    pred = ok ? Array(s) : nothing
    panels = Any[]
    for c in 1:d
        p = plot(; title = "y$c", xlabel = "t", xscale = logt ? :log10 : :identity, legend = (c == 1))
        scatter!(p, tsteps, Ydata[c, :]; label = c == 1 ? "ground truth" : "", mc = :white, msc = :black, ms = 3)
        pred === nothing || plot!(p, pt, pred[c, :]; label = c == 1 ? "neural ODE" : "", color = :orangered, lw = 2)
        push!(panels, p)
    end
    fitplot = plot(panels...; layout = (d, 1), size = (760, 240d))
    savefig(fitplot, "$(spec.name)_fit.png")
    println("\nsaved fit plot → $(spec.name)_fit.png")
end
