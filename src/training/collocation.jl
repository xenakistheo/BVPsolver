const RADAU_A  = [5/12 -1/12; 3/4 1/4]
const RADAU_B  = [3/4, 1/4]
const RADAU_C  = vec(sum(RADAU_A, dims = 2))
const SSTAGE   = 2
const NSUB_CAP = 48
const AMR_FRAC = 0.4
const INIT_MAX_NORM_JUMP = 0.10
const INIT_NSUB_CAP      = 8
const INIT_NSUB_FLOOR    = 1

function initial_nsub(ctx)
    Δ = abs.(diff(ctx.Ydata, dims = 2)) ./ reshape(ctx.yscale, :, 1)
    jump = vec(maximum(Δ, dims = 1))
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
    z0 = ComponentVector(Y = Y0 ./ ctx.yscale, K = K0, theta = copy(θ))
    dp = (; ctx, mesh, nob, yob = ctx.Ydata ./ ctx.yscale, khat)
    nf = NonlinearFunction(R!; resid_prototype = zeros(n_residuals(ctx.d, ctx.tlen, M)),
                           sparsity = TracerSparsityDetector())
    ls = linsolve === :qr ? QRFactorization() : KLUFactorization()
    lm = LevenbergMarquardt(; b_uphill = 0.0, linsolve = ls)
    sol = solve(NonlinearLeastSquaresProblem(nf, z0, dp), lm; maxiters = lm_iters)
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
    if ok_rounds == 0
        println(" !! NO COLLOCATION ROUND COMPLETED — every round threw. " *
                "The reported result is DERIVATIVE MATCHING ONLY, not collocation.")
    elseif best == best0
        println(" derivative matching was best ($ok_rounds/$(cfg.amr_rounds) rounds ran)")
    end
    return best_θ
end
