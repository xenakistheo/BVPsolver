
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

function plot_fit(ctx, θ; dir = ".")
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
    path = joinpath(dir, "$(ctx.spec.name)_fit.png")
    savefig(plot(panels...; layout = (ctx.d, 1), size = (760, 240 * ctx.d)), path)
    println("saved $path")
end


function eigenvalues(ctx, θ; absolute=false)
    F_true(x) = (du = similar(x); ctx.spec.true_ode!(du, x, nothing, zero(eltype(x))); du)
    F_learned(x) = f_theta(ctx, x, θ)


    eigenvalues_true = zeros(ComplexF64, size(ctx.Ydata))
    eigenvalues_learned = zeros(ComplexF64, size(ctx.Ydata))

    for i in 1:size(ctx.Ydata)[2]
        y = ctx.Ydata[:, i]
        J_true = ForwardDiff.jacobian(F_true, y)
        eigenvalues_true[:, i] = sort(eigvals(J_true), by = abs, rev = true)
        J_learned = ForwardDiff.jacobian(F_learned, y)
        eigenvalues_learned[:, i] = sort(eigvals(J_learned), by = abs, rev = true)
    end

    if absolute
        eigenvalues_true = abs.(eigenvalues_true)
        eigenvalues_learned = abs.(eigenvalues_learned)
    end

    return eigenvalues_true, eigenvalues_learned
end


function plot_spectral_fit(ctx, θ; dir = ".")
    logt = all(>(0), ctx.tsteps) && ctx.tspan[2] / max(ctx.tspan[1], eps()) > 100
    eigenvalues_true, eigenvalues_learned = eigenvalues(ctx, θ; absolute = true)
    panels = map(1:ctx.d) do c
        p = plot(; title = "λ$c", xlabel = "t", xscale = logt ? :log10 : :identity, legend = c == 1)
        scatter!(p, ctx.tsteps, eigenvalues_true[c, :]; label = c == 1 ? "true" : "",
                 mc = :white, msc = :black, ms = 3)
        plot!(p, ctx.tsteps, eigenvalues_learned[c, :]; label = c == 1 ? "learned" : "",
              color = :orangered, lw = 2)
        p
    end
    path = joinpath(dir, "$(ctx.spec.name)_spectral_fit.png")
    savefig(plot(panels...; layout = (ctx.d, 1), size = (760, 240 * ctx.d)), path)
    println("saved $path")
end
