
using JuMP, Ipopt
using Lux, Random, ComponentArrays
using Plots
using OrdinaryDiffEqTsit5
# using OrdinaryDiffEqRosenbrock

RNG = Xoshiro()

#### Create Data
tspan = (0.0, 1.5)
N = 30
tsteps = collect(range(tspan[1], tspan[2]; length=N))

function spiralODE(du, u, p, t)
    a = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)' * a)'
end



# Create training data
data_prob = ODEProblem(spiralODE, [2.0, 0.0], tspan)
data = solve(data_prob, Tsit5(), saveat=tsteps)
# plot(data)
X, Y_obs = data.t, reduce(hcat, data.u)'


basicNN = Chain(
    Dense(2, 4, tanh),
    Dense(4, 2)
)

model = basicNN

ps, st = Lux.setup(RNG, model)
ps = Lux.f64(ps)

ps_ca = ComponentVector(ps)
length(ps_ca)
ps_vec = collect(ps_ca)
ps_axes = getaxes(ps_ca)


function F(u, p)
    return model(@view(u[1:2]), ComponentVector(p, ps_axes), st)[1]
end

function F!(du, u, p, t)
    du .= F(u, p)
end

F([1,1], ps_vec)
stack(F(Y_obs[i, :], ps_vec) for i in 1:N)'

#### Define Collocation Points and Variables 
function differentiation_variables(tspan, N)
    t0, tf = tspan
    xi = sort([cos(i*pi/(N-1)) for i in 0:N-1])

    w = [(prod(xi[j] - xi[k] for k in 1:N if k != j))^-1 for j in 1:N]

    D = zeros(N, N)
    for i in 1:N
        for j in 1:N
            if i != j
                D[i, j] = (w[j]/w[i]) / (xi[i] - xi[j])
            end 
        end 
    end 

    for i in 1:N
        D[i, i] = -sum(D[i, j] for j in 1:N if j != i)
    end

    tcol = @. (t0 + tf) / 2 + (tf - t0) / 2 * xi
    Dt = (2 / (tf - t0)) .* D

    return tcol, Dt
end

function interp_rows(t, Y, tq)
    Yq = zeros(length(tq), size(Y, 2))

    for k in axes(Y, 2)
        for i in eachindex(tq)
            q = tq[i]
            j = clamp(searchsortedlast(t, q), 1, length(t)-1)
            α = (q - t[j]) / (t[j+1] - t[j])
            Yq[i, k] = (1 - α) * Y[j, k] + α * Y[j+1, k]
        end
    end

    return Yq
end


function collocation_train(model, Y_obs_col, p0, tspan; λ=1e-6)
    N, d = size(Y_obs_col)
    tcol, Dt = differentiation_variables(tspan, N)
    n_ps = length(p0)

    m = JuMP.Model(Ipopt.Optimizer)

    @variable(m, ps[1:n_ps])
    @variable(m, Y[1:N, 1:d])

    set_start_value.(ps, p0)
    set_start_value.(Y, Y_obs_col)

    for k in 1:d
        f_k = let model = model, d = d, k = k
            (args...) -> model(collect(args[1:d]), collect(args[d+1:end]))[k]
        end
        op_name = Symbol("nn_F$k")
        m[op_name] = add_nonlinear_operator(m, d + n_ps, f_k; name = op_name)
    end

    # Anchor the IVP initial condition
    @constraint(m, [k=1:d], Y[1, k] == Y_obs_col[1, k])

    for i in 1:N
        for k in 1:d
            op = m[Symbol("nn_F$k")]
            @constraint(m,
                sum(Dt[i, j] * Y[j, k] for j in 1:N)
                ==
                op(Y[i, 1:d]..., ps...)
            )
        end
    end

    @objective(m, Min,
        sum((Y[i, j] - Y_obs_col[i, j])^2 for i in 1:N for j in 1:d)
        + λ * sum(ps[k]^2 for k in 1:n_ps)
    )

    JuMP.optimize!(m)

    return value.(ps), value.(Y), tcol
end

tcol, _ = differentiation_variables(tspan, N)
Y_obs_col = interp_rows(X, Y_obs, tcol)
ps_opt, Y_opt, tcol = collocation_train(F, Y_obs_col, ps_vec, tspan)


ps_opt


pred_prob = ODEProblem(F!, [2.0, 0.0], tspan, ps_opt)
pred = solve(pred_prob, Tsit5(), saveat=tsteps)

#
plot(ps_opt, label="Optimized Parameters")
plot!(ps_vec, label="Initial Parameters")

# plot the trajectories
plot(X, Y_obs[:, 1], label="Observed x(t)", lw=2)
plot!(X, Y_obs[:, 2], label="Observed y(t)", lw=2)
plot!(pred.t, reduce(hcat, pred.u)', label=["Predicted x(t)", "Predicted y(t)"], lw=2, ls=:dot)
xlabel!("Time")
ylabel!("State")
title!("Collocation Training of Neural ODE")


