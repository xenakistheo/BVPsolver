function cheb_diffmatrix(tspan, N)
    t0, tf = tspan
    xi = sort([cos(i * pi / (N - 1)) for i in 0:N-1])
    w  = [(prod(xi[j] - xi[k] for k in 1:N if k != j))^-1 for j in 1:N]
    D  = zeros(N, N)
    for i in 1:N, j in 1:N
        i != j && (D[i, j] = (w[j] / w[i]) / (xi[i] - xi[j]))
    end
    for i in 1:N
        D[i, i] = -sum(D[i, j] for j in 1:N if j != i)
    end
    tcol = @. (t0 + tf) / 2 + (tf - t0) / 2 * xi
    return tcol, (2 / (tf - t0)) .* D
end

function interp_rows(t, Y, tq)
    Yq = zeros(length(tq), size(Y, 2))
    for k in axes(Y, 2), i in eachindex(tq)
        q = tq[i]
        j = clamp(searchsortedlast(t, q), 1, length(t) - 1)
        α = (q - t[j]) / (t[j+1] - t[j])
        Yq[i, k] = (1 - α) * Y[j, k] + α * Y[j+1, k]
    end
    return Yq
end

function shapovalova(ctx, θ0, ncol; lambda = 1e-6)
    d, nps   = ctx.d, length(θ0)
    tcol, Dt = cheb_diffmatrix(ctx.tspan, ncol)
    Yobs     = interp_rows(ctx.tsteps, permutedims(ctx.Ydata), tcol)

    m = JuMP.Model(Ipopt.Optimizer)
    JuMP.@variable(m, ps[1:nps])
    JuMP.@variable(m, Y[1:ncol, 1:d])
    JuMP.set_start_value.(ps, θ0)
    JuMP.set_start_value.(Y, Yobs)

    for k in 1:d
        f_k = let ctx = ctx, d = d, k = k
            (args...) -> f_theta(ctx, collect(args[2:d+1]), args[1],
                                 collect(args[d+2:end]))[k]
        end
        nm = Symbol("nn_F$k")
        m[nm] = JuMP.add_nonlinear_operator(m, d + nps + 1, f_k; name = nm)
    end

    JuMP.@constraint(m, [k = 1:d], Y[1, k] == Yobs[1, k])
    for i in 1:ncol, k in 1:d
        op = m[Symbol("nn_F$k")]
        JuMP.@constraint(m, sum(Dt[i, j] * Y[j, k] for j in 1:ncol) ==
                            op(tcol[i], Y[i, 1:d]..., ps...))
    end

    JuMP.@objective(m, Min,
        sum((Y[i, j] - Yobs[i, j])^2 for i in 1:ncol, j in 1:d) +
        lambda * sum(ps[k]^2 for k in 1:nps))

    JuMP.optimize!(m)
    return collect(JuMP.value.(ps)), JuMP.objective_value(m)
end
