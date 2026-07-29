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
using LinearAlgebra, Statistics
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
    max.(vec(sqrt.(sum(abs2, dY, dims = 2) ./ size(dY, 2))), eps())
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

# new Collocation + nonlinear least squares solve. We make our unkown residual vector z = (Y, K, θ)
const RADAU_A = [5/12 -1/12; 3/4 1/4] 
const RADAU_B = [3/4, 1/4]
const SSTAGE  = 2

const REFINE = PROBLEM == "spiral" ? 4 : 2
mesh = Float64[]; node_of_obs = Int[]
for j in 1:tlen-1
    t0, t1 = tsteps[j], tsteps[j+1]
    push!(node_of_obs, length(mesh) + 1)
    for kk in 0:REFINE-1; push!(mesh, t0 + (t1 - t0) * kk / REFINE); end
end
push!(mesh, tsteps[end]); push!(node_of_obs, length(mesh))
M = length(mesh)
@assert all(j -> mesh[node_of_obs[j]] ≈ tsteps[j], 1:tlen)

n_res = (M - 1) * SSTAGE * d + (M - 1) * d + tlen * d
function R!(res, z, p)
    Y = z.Y; K = z.K; θ = z.theta
    msh = p.mesh; A = p.A; b = p.b; s = p.s; dd = p.d; Yob = p.Y_obs; nob = p.nob
    k = 0
    @inbounds for i in 1:M-1
        h = msh[i+1] - msh[i]
        for j in 1:s                            
            stage = copy(view(Y, :, i))
            for l in 1:s
                stage = stage .+ (h * A[j, l]) .* view(K, :, i, l)
            end
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

function interp_data_onto(mesh)
    tt = tsteps; YY = Ydata
    out = similar(YY, d, length(mesh))
    for (q, t) in enumerate(mesh)
        if t <= tt[1]
            out[:, q] = YY[:, 1]
        elseif t >= tt[end]
            out[:, q] = YY[:, end]
        else
            kk = searchsortedlast(tt, t)
            α = (t - tt[kk]) / (tt[kk+1] - tt[kk])
            out[:, q] = (1 - α) .* YY[:, kk] .+ α .* YY[:, kk+1]
        end
    end
    return out
end
Y0 = interp_data_onto(mesh)
K0 = zeros(d, M - 1, SSTAGE)
for i in 1:M-1
    slope = (Y0[:, i+1] .- Y0[:, i]) ./ (mesh[i+1] - mesh[i])
    for j in 1:SSTAGE; K0[:, i, j] = slope; end
end
data_p = (mesh = mesh, A = RADAU_A, b = RADAU_B, s = SSTAGE, d = d,
          Y_obs = Ydata, nob = node_of_obs)
nlfun = NonlinearFunction(R!; resid_prototype = zeros(n_res), sparsity = TracerSparsityDetector())
println("NLS sizes: M=$M  d=$d  N_params=$N_params  n_unknowns=$(M*d + (M-1)*SSTAGE*d + N_params)  n_res=$n_res")

function train_collocation(θ0)
    z0 = ComponentVector(Y = Y0, K = K0, theta = copy(θ0))
    try
        sol = solve(NonlinearLeastSquaresProblem(nlfun, z0, data_p), LevenbergMarquardt(); maxiters = 500)
        return collect(sol.u.theta)
    catch e
        @warn "NLS failed" exception=(e,); return nothing
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

# velcoity field metric: f_θ vs true f squared relative error
let
    logt = all(>(0), tsteps) && (tspan[2] / max(tspan[1], eps()) > 100)
    dense_t = logt ? exp.(range(log(tsteps[1]), log(tsteps[end]); length = 3000)) :
                     collect(range(tspan[1], tspan[2]; length = 3000))
    global VF_STATES = Array(solve(ODEProblem(spec.true_ode!, u0, tspan), spec.solver;
                                   saveat = dense_t, spec.solve_kwargs...))
    global VF_DERIVS = true_field(VF_STATES)
end
function vf_error(θ)
    θ === nothing && return nothing
    sse = zeros(d); ref = zeros(d)
    for j in axes(VF_STATES, 2)
        fp = f_theta(VF_STATES[:, j], θ)
        sse .+= (fp .- VF_DERIVS[:, j]) .^ 2
        ref .+= VF_DERIVS[:, j] .^ 2
    end
    rel = sqrt.(sse ./ max.(ref, eps()))
    return sqrt(sum(abs2, rel) / d)
end

rd(x) = x === nothing ? "  --  " : (isfinite(x) ? rpad(round(x; sigdigits = 4), 7) : "Inf    ")

trained = Dict{String,Any}()   # trained params per method, for the fit plot
for seed in SEEDS
    θ0s = 0.1 .* randn(Xoshiro(seed), N_params)
    do_method = function (name, trainer)
        t = @elapsed (θ = trainer(θ0s))
        trained[name] = θ
        println("seed $seed  $(rpad(name, 11)) vf=$(rd(vf_error(θ)))  traj=$(rd(traj_l2(θ)))   [$(round(Int, t))s]"); flush(stdout)
    end
    do_method("supervised", train_supervised)
    SWEEP_SHOOT && do_method("shooting", train_shooting)
    SWEEP_AL && do_method("AL-BVP", train_al)
    do_method("collocation", train_collocation)
end

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
