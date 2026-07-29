using OrdinaryDiffEq, Statistics, Plots
using SciMLBase: successful_retcode
import SciMLLogging as SL

function rollout(ctx, θ; dense = false)
    rhs!(du, u, p, t) = (du .= f_theta(ctx, u, p); nothing)
    prob = ODEProblem(rhs!, ctx.u0, ctx.tspan, θ)
    kw   = (; verbose = SL.None(), ctx.spec.solve_kwargs...)
    return dense ? solve(prob, ctx.spec.solver; kw...) :
                   solve(prob, ctx.spec.solver; saveat = ctx.tsteps, kw...)
end

function metrics(ctx, θ)
    sol = rollout(ctx, θ)
    if !successful_retcode(sol)
        println("  rollout FAILED: $(sol.retcode) at t=$(sol.t[end]) of $(ctx.tspan[2])")
        return (nrmse = Inf, l2 = Inf)
    end
    P = Array(sol)
    return (nrmse = sqrt(mean(abs2, (P .- ctx.Ydata) ./ ctx.yscale)),
            l2    = sqrt(sum(abs2, P .- ctx.Ydata)))
end

function report(ctx, label, θ)
    m = metrics(ctx, θ)
    println("  [$label]  nrmse=$(round(m.nrmse; sigdigits=4))  l2=$(round(m.l2; sigdigits=4))")
    flush(stdout)
    return m
end

function plot_fit(ctx, θ)
    logt  = all(>(0), ctx.tsteps) && ctx.tspan[2] / max(ctx.tspan[1], eps()) > 100
    tgrid = logt ? exp.(range(log(ctx.tsteps[1]), log(ctx.tsteps[end]); length = 400)) :
                   collect(range(ctx.tspan[1], ctx.tspan[2]; length = 400))
    sol  = rollout(ctx, θ; dense = true) 
    ok   = successful_retcode(sol)
    ts   = ok ? sort(unique(vcat(tgrid, sol.t))) : Float64[]
    pred = ok ? reduce(hcat, (sol(t) for t in ts)) : zeros(ctx.d, 0)
    panels = map(1:ctx.d) do c
        p = plot(; title = "y$c", xlabel = "t", xscale = logt ? :log10 : :identity, legend = c == 1)
        scatter!(p, ctx.tsteps, ctx.Ydata[c, :]; label = c == 1 ? "data" : "",
                 mc = :white, msc = :black, ms = 3)
        ok && plot!(p, ts, pred[c, :]; label = c == 1 ? "neural ODE" : "",
                    color = :orangered, lw = 2)
        p
    end
    savefig(plot(panels...; layout = (ctx.d, 1), size = (760, 240 * ctx.d)),
            "$(ctx.spec.name)_fit.png")
    println("saved $(ctx.spec.name)_fit.png")
end
