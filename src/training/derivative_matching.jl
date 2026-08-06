
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
    fwd  = diff(ctx.Ydata, dims = 2) ./ diff(ctx.tsteps)'
    for c in 1:d, j in 2:n-1
        a, b = abs(fwd[c, j-1]), abs(fwd[c, j])
        max(a, b) / max(min(a, b), 1e-300) > 100 && (w[c, j] = 0)
    end
    loss(θ, _) = sum(abs2, (asinh.(f_theta(ctx, ctx.Ydata, θ) ./ κ) .- tgt) .* w) / sum(w)
    optf = OptimizationFunction(loss, Optimization.AutoZygote())
    θ = copy(θ0)
    for (lr, frac) in ((1e-2, 0.34), (3e-3, 0.25), (1e-3, 0.25), (3e-4, 0.16))
        θ = solve(OptimizationProblem(optf, θ), OptimizationOptimisers.Adam(lr);
                  maxiters = max(1, round(Int, frac * iters))).u
    end
    return θ, loss(θ, nothing)
end
