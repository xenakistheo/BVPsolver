function shooting(ctx, θ0, iters)
    rhs!(du, u, p, t) = (du .= f_theta(ctx, u, t, p); nothing)
    kw  = (; verbose = SL.None(), ctx.spec.solve_kwargs...)
    w   = collect(Float64, ctx.yscale)
    pen = sum(abs, ctx.Ydata ./ w) / length(ctx.Ydata)
    function loss(θ, _)
        sol = solve(ODEProblem(rhs!, ctx.u0, ctx.tspan, θ), ctx.spec.solver;
                    saveat = ctx.tsteps, kw...)
        P = Array(sol)
        n = size(P, 2)
        n == 0 && return pen
        R = (P .- view(ctx.Ydata, :, 1:n)) ./ w
        return (sum(abs, R) + pen * ctx.d * (ctx.tlen - n)) / (ctx.d * ctx.tlen)
    end
    optf  = OptimizationFunction(loss, Optimization.AutoForwardDiff())
    every = max(1, iters ÷ 20)
    cb = function (state, l)
        if state.iter % every == 0
            println("  shooting iter=$(state.iter)/$iters loss=$(round(l; sigdigits=4))")
            flush(stdout)
        end
        return false
    end
    θ = solve(OptimizationProblem(optf, copy(θ0)), OptimizationOptimisers.Adam(5e-3);
              maxiters = iters, callback = cb).u
    return θ, loss(θ, nothing)
end
