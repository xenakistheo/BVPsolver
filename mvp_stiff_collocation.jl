# julia --project=. mvp_stiff_collocation.jl
#
# Self-contained MVP: reproduce `julia --project=. src/run_main.jl` with
# its defaults (--problem vanderpol --model stiff --profile fast
# --pretraining derivmatch --training collocation --seed 123) as a single
# file with no dependency on the StiffNN package. Problem setup, the
# "stiff" field architecture, derivative-matching pretraining, and
# collocation training are all ported 1:1 from src/run_main.jl,
# src/problems.jl, src/models/stiff_field.jl,
# src/training/derivative_matching.jl, and src/training/collocation.jl.
ENV["GKSwstype"] = "100"

using OrdinaryDiffEqRosenbrock: Rodas5P
using OrdinaryDiffEqSDIRK: Kvaerno5
using SciMLBase: ODEProblem, ODEFunction, NonlinearFunction, NonlinearLeastSquaresProblem,
                 solve, successful_retcode
import SciMLLogging as SL
using Lux, Random, ComponentArrays
using NonlinearSolve, LinearSolve
using SparseConnectivityTracer: TracerSparsityDetector
using Optimization, OptimizationOptimisers, Zygote
using Statistics
using Plots

# ------------------------------------------------------------------
# 1. Stiff Van der Pol data -- src/problems.jl "vanderpol", profile=:fast
#    (μ=100 is make_problem's default; :fast selects tspan=(0,200) and
#    the looser Kvaerno5 tolerances). tsteps is the adaptive solver's own
#    accepted grid, exactly as make_problem builds it.
# ------------------------------------------------------------------
const seed  = 123
const μ     = 100.0
const tspan = (0.0, 200.0)
const u0    = [2.0, 0.0]
const solver = Kvaerno5()
const solve_kwargs = (; abstol=1e-7, reltol=1e-7)   # profile = :fast

function vanderpol!(du, u, p, t)
    du[1] = u[2]
    du[2] = μ * (1 - u[1]^2) * u[2] - u[1]
    return nothing
end

tsteps = solve(ODEProblem(vanderpol!, u0, tspan), solver; solve_kwargs...).t
Ydata  = Array(solve(ODEProblem(vanderpol!, u0, tspan), solver;
                     saveat=tsteps, solve_kwargs...))   # d x N
d = size(Ydata, 1)

# ------------------------------------------------------------------
# 2. The "stiff" field architecture
#
# A trunk network maps (state, asinh-compressed state) to a hidden
# representation, which splits into a bounded "production" head (the
# state-independent part of the RHS) and a bounded "loss-rate" head
# (a signed log-linear damping term). This keeps the RHS finite even
# for very fast/slow dynamics, which is what makes it suitable for
# stiff systems -- see src/models/stiff_field.jl for the original.
# ------------------------------------------------------------------
const KAPPA_FRAC    = 1e-3
const PROD_HEADROOM = 3e5
const CHI_FRAC      = 1e-3
const ZNORM         = asinh(1 / CHI_FRAC)

struct StiffField{TR,P,L} <: Lux.AbstractLuxContainerLayer{(:trunk, :prod, :logloss)}
    trunk::TR
    prod::P
    logloss::L
    ymid::Vector{Float64}
    yscale::Vector{Float64}
    chi::Vector{Float64}
    kappa::Vector{Float64}
    lam::Vector{Float64}
    gbound::Vector{Float64}
    abound::Float64
end

function (m::StiffField)(u::AbstractVecOrMat, t, ps, st)   # t unused (time-independent RHS)
    x̂ = (u .- m.ymid) ./ m.yscale
    z = asinh.(u ./ m.chi) ./ ZNORM
    h, _  = m.trunk(vcat(x̂, z), ps.trunk, st.trunk)
    pa, _ = m.prod(h, ps.prod, st.prod)
    lb, _ = m.logloss(h, ps.logloss, st.logloss)
    f = m.kappa .* sinh.(m.abound .* tanh.(pa ./ m.abound))
    c = m.gbound .* tanh.(lb ./ m.gbound)
    return f .+ m.lam .* sinh.(c) .* u, st   # signed_loss = true
end

function build_stiff_field(d, Ydata, tsteps, ymid, yscale; width, depth)
    slope = max.(vec(maximum(abs.(diff(Ydata, dims=2) ./ diff(tsteps)'), dims=2)), 1e-30)
    umax  = max.(vec(maximum(abs.(Ydata); dims=2)), 1e-30)
    lam   = slope ./ max.(collect(Float64, yscale), 1e-12 .* umax, floatmin())
    rmax  = 10 / minimum(diff(tsteps))
    trunk = Chain(Dense(2d, width, tanh), (Dense(width, width, tanh) for _ in 2:depth)...)
    return StiffField(
        trunk, Dense(width, d), Dense(width, d),
        collect(Float64, ymid), collect(Float64, yscale),
        CHI_FRAC .* umax, KAPPA_FRAC .* slope, lam,
        clamp.(asinh.(rmax ./ lam), 1.0, asinh(1e6)),
        asinh(PROD_HEADROOM))
end

# CONFIGS.vanderpol from src/run_main.jl
const CFG = (hidden=12, depth=2, signed_loss=true, dm_iters=10_000, amr_rounds=1, lm_iters=150)

# ymid/yscale computed exactly as in src/run_main.jl (passed into the
# model rather than derived inside it).
ymin, ymax = vec(minimum(Ydata, dims=2)), vec(maximum(Ydata, dims=2))
yscale = max.(ymax .- ymin, 1e-12 .* max.(abs.(ymax), abs.(ymin)), floatmin())
ymid   = (ymax .+ ymin) ./ 2

model = build_stiff_field(d, Ydata, tsteps, ymid, yscale; width=CFG.hidden, depth=CFG.depth)
rng   = Xoshiro(seed)
ps, st = Lux.setup(rng, model)
ps = Lux.f64(ps)
ps.logloss.bias .= (CFG.signed_loss ? -1.0 : -6.0)   # init_stiff!
ps_axes = getaxes(ComponentVector(ps))
θ0      = collect(ComponentVector(ps))

f_theta(ctx, u, t, θ) = ctx.model(u, t, ComponentVector(θ, ctx.ps_axes), ctx.st)[1]

ctx = (; d, tlen = length(tsteps), Ydata, yscale = model.yscale, tsteps, tspan, u0,
         model, st, ps_axes, solver, solve_kwargs)

# ------------------------------------------------------------------
# 3. Collocation training -- ported from src/training/collocation.jl
#
# Discretizes each observation-to-observation interval with a 2-stage
# Radau IIA scheme, treats the stage states/derivatives and the network
# parameters as one joint unknown vector, and solves the (generally
# nonsquare, least-squares) residual system with Levenberg-Marquardt.
# Adaptive mesh refinement (AMR) subdivides intervals with the largest
# collocation residual between rounds.
# ------------------------------------------------------------------
const RADAU_A  = [5/12 -1/12; 3/4 1/4]
const RADAU_B  = [3/4, 1/4]
const RADAU_C  = vec(sum(RADAU_A, dims=2))
const SSTAGE   = 2
const NSUB_CAP = 48
const AMR_FRAC = 0.4
const INIT_MAX_NORM_JUMP = 0.10
const INIT_NSUB_CAP      = 8
const INIT_NSUB_FLOOR    = 1

function initial_nsub(ctx)
    Δ = abs.(diff(ctx.Ydata, dims=2)) ./ reshape(ctx.yscale, :, 1)
    jump = vec(maximum(Δ, dims=1))
    return clamp.(ceil.(Int, jump ./ INIT_MAX_NORM_JUMP), INIT_NSUB_FLOOR, INIT_NSUB_CAP)
end

n_residuals(d, tlen, M) = (M - 1) * (SSTAGE + 1) * d + tlen * d

function build_mesh(ctx, nsub)
    mesh, nob = Float64[], Int[]
    for j in 1:ctx.tlen-1
        t0, t1 = ctx.tsteps[j], ctx.tsteps[j+1]
        push!(nob, length(mesh) + 1)
        for k in 0:nsub[j]-1; push!(mesh, t0 + (t1 - t0) * k / nsub[j]); end
    end
    push!(mesh, ctx.tsteps[end]); push!(nob, length(mesh))
    return mesh, nob
end

function interp_onto(Y, t, mesh)
    out = similar(Y, size(Y, 1), length(mesh))
    for (q, s) in enumerate(mesh)
        if s <= t[1];        out[:, q] = Y[:, 1]
        elseif s >= t[end];  out[:, q] = Y[:, end]
        else
            k = searchsortedlast(t, s)
            α = (s - t[k]) / (t[k+1] - t[k])
            out[:, q] = (1 - α) .* Y[:, k] .+ α .* Y[:, k+1]
        end
    end
    return out
end

function R!(res, z, p)
    Ŷ, K̂, θ = z.Y, z.K, z.theta
    ctx, msh, nob, k̂ = p.ctx, p.mesh, p.nob, p.khat
    d, ysc, s, M = ctx.d, ctx.yscale, SSTAGE, length(msh)
    S = similar(Ŷ, d, (M - 1) * s)
    Tstage = Vector{eltype(msh)}(undef, (M - 1) * s)
    @inbounds for i in 1:M-1
        hi = msh[i+1] - msh[i]
        for j in 1:s
            q = (i - 1) * s + j
            Tstage[q] = msh[i] + RADAU_C[j] * hi
            for c in 1:d
                acc = Ŷ[c, i]
                for l in 1:s; acc += RADAU_A[j, l] * K̂[c, i, l]; end
                S[c, q] = ysc[c] * acc
            end
        end
    end
    F = f_theta(ctx, S, Tstage, θ)
    k = 0
    @inbounds for i in 1:M-1
        h = msh[i+1] - msh[i]
        for j in 1:s
            for c in 1:d
                g = (h / ysc[c]) * F[c, (i-1)*s + j]
                res[k+c] = (K̂[c, i, j] - g) / k̂[c]
            end
            k += d
        end
        for c in 1:d
            acc = Ŷ[c, i+1] - Ŷ[c, i]
            for j in 1:s; acc -= RADAU_B[j] * K̂[c, i, j]; end
            res[k+c] = acc
        end
        k += d
    end
    @inbounds for j in 1:ctx.tlen
        n = nob[j]
        for c in 1:d; res[k+c] = Ŷ[c, n] - p.yob[c, j]; end
        k += d
    end
    return nothing
end

function solve_collocation(ctx, nsub, θ, lm_iters, linsolve)
    mesh, nob = build_mesh(ctx, nsub); M = length(mesh)
    Y0 = interp_onto(ctx.Ydata, ctx.tsteps, mesh)
    K0 = zeros(ctx.d, M - 1, SSTAGE)
    for i in 1:M-1, j in 1:SSTAGE
        K0[:, i, j] = (Y0[:, i+1] .- Y0[:, i]) ./ ctx.yscale
    end
    khat = [max(1e-2 * maximum(abs, @view K0[c, :, :]), 1e-12) for c in 1:ctx.d]
    z0 = ComponentVector(Y=Y0 ./ ctx.yscale, K=K0, theta=copy(θ))
    dp = (; ctx, mesh, nob, yob = ctx.Ydata ./ ctx.yscale, khat)
    nf = NonlinearFunction(R!; resid_prototype=zeros(n_residuals(ctx.d, ctx.tlen, M)),
                           sparsity=TracerSparsityDetector())
    ls = linsolve === :qr ? QRFactorization() : KLUFactorization()
    lm = LevenbergMarquardt(; b_uphill=0.0, linsolve=ls)
    freq = max(1, lm_iters ÷ 20)
    println("  collocation: M=$M residuals=$(n_residuals(ctx.d, ctx.tlen, M))  detecting sparsity + solving...")
    flush(stdout)
    sol = solve(NonlinearLeastSquaresProblem(nf, z0, dp), lm; maxiters=lm_iters,
                show_trace=Val(true), trace_level=TraceMinimal(; print_frequency=freq))
    return sol.u, dp
end

function interval_errors(z, dp)
    ctx  = dp.ctx
    res  = zeros(n_residuals(ctx.d, ctx.tlen, length(dp.mesh))); R!(res, z, dp)
    rows = (SSTAGE + 1) * ctx.d
    err  = zeros(ctx.tlen - 1)
    for j in 1:ctx.tlen-1
        acc = 0.0
        for i in dp.nob[j]:dp.nob[j+1]-1, r in 1:rows
            acc += res[(i - 1) * rows + r]^2
        end
        err[j] = acc
    end
    return err
end

function obs_residual(z, dp)
    ctx = dp.ctx
    res = zeros(n_residuals(ctx.d, ctx.tlen, length(dp.mesh))); R!(res, z, dp)
    k   = (length(dp.mesh) - 1) * (SSTAGE + 1) * ctx.d
    O   = @view res[k+1:k+ctx.tlen*ctx.d]
    return sqrt(sum(abs2, O) / length(O))
end

function train_collocation(ctx, θ0, cfg; score)
    nsub  = initial_nsub(ctx)
    best0 = score(θ0)
    θ, best_θ, best = copy(θ0), copy(θ0), best0
    ok_rounds = 0
    for rnd in 1:cfg.amr_rounds
        try
            z, dp = solve_collocation(ctx, nsub, θ, cfg.lm_iters, get(cfg, :linsolve, :klu))
            θ_candidate = collect(z.theta)
            sc = score(θ_candidate)
            if sc < best
                best = sc
                best_θ = copy(θ_candidate)
            end
            err, refined = interval_errors(z, dp), 0
            me = maximum(err)
            if rnd < cfg.amr_rounds
                for j in 1:ctx.tlen-1
                    if err[j] > AMR_FRAC * me && nsub[j] < NSUB_CAP
                        nsub[j] = min(2nsub[j], NSUB_CAP); refined += 1
                    end
                end
            end
            println("  AMR $rnd/$(cfg.amr_rounds): M=$(length(dp.mesh))  " *
                    "err=$(round(me; sigdigits=3))  " *
                    "obs=$(round(obs_residual(z, dp); sigdigits=3))  " *
                    "nrmse=$(round(sc; sigdigits=3))  " *
                    "best=$(round(best; sigdigits=3))  refined=$refined"); flush(stdout)
            ok_rounds += 1
            θ = θ_candidate
            refined == 0 && break
        catch e
            @warn "AMR round $rnd failed — keeping the best result so far" exception=(e,)
            break
        end
    end
    ok_rounds == 0 && println(" !! NO COLLOCATION ROUND COMPLETED — reporting the initial guess.")
    return best_θ
end

# ------------------------------------------------------------------
# 4. Derivative-matching pretraining -- ported from
#    src/training/derivative_matching.jl
#
# Fits the network's RHS directly against finite-difference-estimated
# derivatives of the data (in an asinh-compressed, kappa-scaled space)
# with plain Adam, in a few decaying-learning-rate stages. This gives
# collocation a much better starting point than a random init, since
# collocation alone is a local (Levenberg-Marquardt) solve.
# ------------------------------------------------------------------
function fd_derivatives(Y, t)
    D = similar(Y)
    D[:, 1]   = (Y[:, 2]   .- Y[:, 1])     ./ (t[2]   - t[1])
    D[:, end] = (Y[:, end] .- Y[:, end-1]) ./ (t[end] - t[end-1])
    for j in 2:length(t)-1
        h1, h2 = t[j] - t[j-1], t[j+1] - t[j]
        D[:, j] = (-h2/(h1*(h1+h2))) .* Y[:, j-1] .+
                  ((h2-h1)/(h1*h2))  .* Y[:, j]   .+
                  ( h1/(h2*(h1+h2))) .* Y[:, j+1]
    end
    return D
end

function derivative_matching(ctx, θ0, target, iters)
    κ   = ctx.model.kappa
    tgt = asinh.(target ./ κ)
    d, n = size(ctx.Ydata)
    w    = ones(d, n)
    fwd  = diff(ctx.Ydata, dims=2) ./ diff(ctx.tsteps)'
    for c in 1:d, j in 2:n-1
        a, b = abs(fwd[c, j-1]), abs(fwd[c, j])
        max(a, b) / max(min(a, b), 1e-300) > 100 && (w[c, j] = 0)
    end
    loss(θ, _) = sum(abs2, (asinh.(f_theta(ctx, ctx.Ydata, ctx.tsteps, θ) ./ κ) .- tgt) .* w) / sum(w)
    optf = OptimizationFunction(loss, Optimization.AutoZygote())
    θ = copy(θ0)
    for (lr, frac) in ((1e-2, 0.34), (3e-3, 0.25), (1e-3, 0.25), (3e-4, 0.16))
        cur_iters = max(1, round(Int, frac * iters))
        every = max(1, cur_iters ÷ 20)
        cb(state, l) = begin
            if state.iter % every == 0
                println("  derivmatch lr=$lr iter=$(state.iter)/$cur_iters loss=$(round(l; sigdigits=4))")
                flush(stdout)
            end
            false
        end
        θ = solve(OptimizationProblem(optf, θ), OptimizationOptimisers.Adam(lr);
                  maxiters=cur_iters, callback=cb).u
    end
    return θ, loss(θ, nothing)
end

# ------------------------------------------------------------------
# 5. Scoring / reporting -- ported from src/utils/eval.jl
# ------------------------------------------------------------------
function rollout(ctx, θ)
    rhs!(du, u, p, t) = (du .= f_theta(ctx, u, t, p); nothing)
    prob = ODEProblem(rhs!, ctx.u0, ctx.tspan, θ)
    return solve(prob, ctx.solver; saveat=ctx.tsteps, verbose=SL.None(), ctx.solve_kwargs...)
end

function metrics(ctx, θ)
    sol = rollout(ctx, θ)
    if !successful_retcode(sol)
        println("  rollout FAILED: $(sol.retcode) at t=$(sol.t[end]) of $(ctx.tspan[2])")
        return (nrmse=Inf, l2=Inf)
    end
    P = Array(sol)
    return (nrmse=sqrt(mean(abs2, (P .- ctx.Ydata) ./ ctx.yscale)),
            l2=sqrt(sum(abs2, P .- ctx.Ydata)))
end

function report(ctx, label, θ)
    m = metrics(ctx, θ)
    println("  [$label]  nrmse=$(round(m.nrmse; sigdigits=4))  l2=$(round(m.l2; sigdigits=4))")
    flush(stdout)
    return m
end

# ------------------------------------------------------------------
# 6. Run it: derivative matching, then collocation (src/run_main.jl's
#    --pretraining derivmatch --training collocation path)
# ------------------------------------------------------------------
report(ctx, "initial", θ0)

θ_dm, dm_loss = derivative_matching(ctx, θ0, fd_derivatives(Ydata, tsteps), CFG.dm_iters)
println("derivative matching  loss=$(round(dm_loss; sigdigits=3))")
report(ctx, "derivative matching", θ_dm)

θ = train_collocation(ctx, θ_dm, CFG; score = p -> metrics(ctx, p).nrmse)
report(ctx, "final", θ)

# ------------------------------------------------------------------
# 7. Plot data vs. the learned trajectory, both on the training window
#    [0, T] and the extrapolation window [T, 2T].
# ------------------------------------------------------------------
T = tspan[2]
tfull = collect(range(0.0, 2T; length=2length(tsteps) - 1))

true_full = Array(solve(ODEProblem(vanderpol!, u0, (0.0, 2T)), Rodas5P();
                        saveat=tfull, reltol=1e-10, abstol=1e-12))

learned_rhs!(du, u, p, t) = (du .= f_theta(ctx, u, t, p); nothing)
learned_sol = solve(ODEProblem(learned_rhs!, u0, (0.0, 2T), θ), ctx.solver;
                    saveat=tfull, verbose=SL.None(), ctx.solve_kwargs...)
learned_full = successful_retcode(learned_sol) ? Array(learned_sol) : fill(NaN, d, length(tfull))
successful_retcode(learned_sol) || println("  extrapolation rollout FAILED: $(learned_sol.retcode) at t=$(learned_sol.t[end])")

state_labels = ["y1", "y2"]
panels = map(1:d) do k
    p = plot(tfull, true_full[k, :]; label="data", lw=2, color=:black)
    plot!(p, tfull, learned_full[k, :]; label="learned", lw=2, ls=:dash, color=:crimson)
    vline!(p, [T]; label = k == 1 ? "T (train | extrapolate)" : "", ls=:dot, color=:gray)
    ylabel!(p, state_labels[k])
    p
end
plt = plot(panels...; layout=(d, 1), size=(900, 320 * d), xlabel="t",
           title="Stiff Van der Pol: train [0,T] | extrapolate [T,2T]")
savefig(plt, "vanderpol_stiff_collocation_fit.png")
println("saved plot -> vanderpol_stiff_collocation_fit.png")
