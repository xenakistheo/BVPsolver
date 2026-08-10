function shooting(ctx, θ0, iters; paper = false)
    rhs!(du, u, p, t) = (du .= f_theta(ctx, u, t, p); nothing)
    kw = (; verbose = SL.None(), ctx.spec.solve_kwargs...)
    w  = paper ? collect(Float64, ctx.yscale) : ones(ctx.d)
    pen = paper ? sum(abs, ctx.Ydata ./ w) / length(ctx.Ydata) :
                  sum(abs2, ctx.Ydata) / length(ctx.Ydata)
    function loss(θ, _)
        sol = solve(ODEProblem(rhs!, ctx.u0, ctx.tspan, θ), ctx.spec.solver;
                    saveat = ctx.tsteps, kw...)
        P = Array(sol)
        n = size(P, 2)
        n == 0 && return pen
        R = (P .- view(ctx.Ydata, :, 1:n)) ./ w
        e = paper ? sum(abs, R) : sum(abs2, R)
        return (e + pen * ctx.d * (ctx.tlen - n)) / (ctx.d * ctx.tlen)
    end
    optf  = OptimizationFunction(loss, Optimization.AutoForwardDiff())
    sched = paper ? ((5e-3, 1.0),) :
                    ((1e-2, 0.34), (3e-3, 0.25), (1e-3, 0.25), (3e-4, 0.16))
    θ = copy(θ0)
    for (lr, frac) in sched
        θ = solve(OptimizationProblem(optf, θ), OptimizationOptimisers.Adam(lr);
                  maxiters = max(1, round(Int, frac * iters))).u
    end
    return θ, loss(θ, nothing)
end
