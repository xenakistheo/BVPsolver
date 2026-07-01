using JuMP, Ipopt
using Lux, Random, ComponentArrays
using SciMLBase
using OrdinaryDiffEqRosenbrock
using Plots


include("src/layers/StiffLayer.jl")
include("src/derivativeMatching.jl")


# ============================================================
# Robertson + simultaneous collocation training in log-time
# ============================================================
# The neural network Fθ(y) represents the physical-time RHS dy/dt.
# Since we collocate in τ = log10(t), the constraint is
#
#     dY/dτ = log(10) * t(τ) * Fθ(Y).
#
# This is the important scaling for log-spaced Robertson data.

rng = Xoshiro(1)

# -------------------------------
# Robertson data
# -------------------------------
function rober!(du, u, p, t)
    k1, k2, k3 = 0.04, 3e7, 1e4
    y1, y2, y3 = u
    du[1] = -k1 * y1 + k3 * y2 * y3
    du[2] =  k1 * y1 - k2 * y2^2 - k3 * y2 * y3
    du[3] =  k2 * y2^2
    return nothing
end

const tspan_data = (0.0, 1e5)
const tmin_train = 1e-6
const tmax_train = 1e5
const tlen_obs   = 100

# Observation times: nonuniform in physical time, uniform in log-time.
t_obs = collect(10.0 .^ range(log10(tmin_train), log10(tmax_train); length=tlen_obs))
τ_obs = log10.(t_obs)

data_prob = ODEProblem(rober!, [1.0, 0.0, 0.0], tspan_data)
data = solve(data_prob, Rodas5P(); saveat=t_obs, reltol=1e-10, abstol=1e-12)

X_obs = collect(data.t)
Y_obs_raw = Matrix(reduce(hcat, data.u)')



# -------------------------------
# Collocation utilities
# -------------------------------
function chebyshev_nodes_interval(a, b, N)
    # Ascending Chebyshev-Lobatto nodes mapped from [-1, 1] to [a, b].
    ζ = sort([cos(i * pi / (N - 1)) for i in 0:N-1])
    return @. (a + b) / 2 + (b - a) / 2 * ζ
end

function differentiation_matrix_from_nodes(nodes)
    nodes = collect(nodes)
    N = length(nodes)
    @assert all(diff(nodes) .> 0) "nodes must be sorted and distinct"

    w = [
        inv(prod(nodes[j] - nodes[k] for k in 1:N if k != j))
        for j in 1:N
    ]

    D = zeros(eltype(nodes), N, N)
    for i in 1:N
        for j in 1:N
            if i != j
                D[i, j] = (w[j] / w[i]) / (nodes[i] - nodes[j])
            end
        end
    end

    for i in 1:N
        D[i, i] = -sum(D[i, j] for j in 1:N if j != i)
    end

    return D
end

function interpolate_rows(x, Y, xq)
    # Simple piecewise-linear interpolation of matrix rows.
    # Assumes x is sorted ascending and xq lies in [x[1], x[end]].
    x = collect(x)
    xq = collect(xq)
    Yq = zeros(length(xq), size(Y, 2))

    for i in eachindex(xq)
        q = xq[i]
        j = clamp(searchsortedlast(x, q), 1, length(x) - 1)
        α = (q - x[j]) / (x[j+1] - x[j])
        @views Yq[i, :] .= (1 - α) .* Y[j, :] .+ α .* Y[j+1, :]
    end

    return Yq
end

# Number of collocation points. Start with 40--60; increase to 100 after it works.
const Ncol = 100

τ_col = chebyshev_nodes_interval(log10(tmin_train), log10(tmax_train), Ncol)
t_col = 10.0 .^ τ_col
Dτ = differentiation_matrix_from_nodes(τ_col)
scale_logtime = log(10.0) .* t_col

# Interpolate observed data onto the collocation grid, in log-time.
Y_obs = interpolate_rows(τ_obs, Y_obs_raw, τ_col)

# -------------------------------
# Neural RHS
# -------------------------------
# We use a 2-output network and define the third derivative as -(dy1 + dy2)
# so that the learned vector field conserves y1 + y2 + y3.
function scale_state(u)
    return [
        u[1],
        1e2 * u[2],
        u[3],
    ]
end

##############################

##### Example usage
input_dim = 3
m = 8
output_dim = 2

V = Lux.Chain(
    Lux.Dense(input_dim => 8, tanh),
    Lux.Dense(8 => 8, tanh),
    Lux.Dense(8 => 8, tanh),
    Lux.Dense(8 => m),
)

U = Lux.Chain(
    Lux.Dense(m => 8, tanh),
    Lux.Dense(8 => 8, tanh),
    Lux.Dense(8 => 8, tanh),
    Lux.Dense(8 => output_dim),
)

gate = Lux.Chain(
    Lux.Dense(input_dim => 8, tanh),
    Lux.Dense(8 => 2),
)


# For idea 1
S1 = StateDepStiffLayer1(
    gate,
    m;
    bmin=0.0f0,
    bmax=5.0f0,
)

stateDepStiffNet1 = StiffenedNet(V, S1, U) #244 params

## For idea 2
S2 = StateDepStiffLayer2(
    gate,
    m;
    amin=0.0f0,
    amax=3.0f0,
    bmin=0.0f0,
    bmax=5.0f0,
)

stateDepStiffNet2 = StiffenedNet(V, S2, U)



###############################

sigma_affine_sigmoid(ps, j) = 1 / (1 + exp(-(ps[1] * j + ps[2])))
sigma_exp(ps, j) = exp(-(ps[1] * j))

V = Dense(3, 16, tanh)
S = StiffLayer2(sigma_affine_sigmoid; P=16, init=ones(16))
U = Chain(Dense(16, 2, tanh), Dense(2,2))

S_exp = StiffLayer2(sigma_exp; P=16, init=ones(16))


StiffNN = Chain(V, S, U) #120 params
StiffNN_exp = Chain(V, S_exp, U)

#142 params
baseNN = Chain(
    Dense(3, 16, tanh),
    # Dense(16, 16),
    Dense(16, 4, tanh),
    Dense(4, 2)
)



V2 = Chain(Dense(3, 8, tanh), Dense(8, 8, tanh), Dense(8, 8, tanh))
S2 = StiffLayer2(sigma_affine_sigmoid; P=8, init=ones(8))
S2_exp = StiffLayer2(sigma_exp; P=8, init=ones(8))
U2 = Chain(Dense(8, 8, tanh), Dense(8, 8, tanh), Dense(8, 2))

StiffNN_Deep = Chain(V2, S2, U2) # 346 params 
StiffNN_Deep_exp = Chain(V2, S2_exp, U2) # 346 params

V3 = Chain(Dense(3, 8, tanh), Dense(8, 8, tanh), Dense(8, 8, tanh), Dense(8, 8, tanh), Dense(8, 8, tanh), Dense(8, 8, tanh))
S3 = StiffLayer2(sigma_affine_sigmoid; P=8, init=ones(8))
U3 = Chain(Dense(8, 8, tanh), Dense(8, 8, tanh), Dense(8, 8, tanh), Dense(8, 8, tanh), Dense(8, 8, tanh),Dense(8, 2))

StiffNN_SuperDeep = Chain(V3, S3, U3) # 778 params 

baseNN_Deep = Chain(
    Dense(3, 8, tanh),
    Dense(8, 8, tanh),
    Dense(8, 8, tanh),
    Dense(8, 8, tanh),
    Dense(8, 8, tanh),
    Dense(8, 2)
) # 338 params


modelNN = StiffNN_Deep

ps, st = Lux.setup(rng, modelNN)
ps = Lux.f64(ps)
ps_ca = ComponentVector(ps)
ps_vec = collect(ps_ca)
ps_axes = getaxes(ps_ca)

function F(u, p)
    θ = ComponentVector(p, ps_axes)
    z = modelNN(scale_state(u), θ, st)[1]
    return [z[1], z[2], -z[1] - z[2]]
end

function F!(du, u, p, t)
    du .= F(u, p)
    return nothing
end

# -------------------------------
# Simultaneous collocation training
# -------------------------------
function collocation_train_logtime(rhsfun, τnodes, tnodes, Dτ, Y_obs, p0;
        λ = 1e-8,
        parameter_bound = 50.0,
        use_state_bounds = true,
        use_mass_constraint = true,
        use_weighted_state_loss = true,
        ipopt_print_level = 5,
        max_iter = 3000,
    )

    N, d = size(Y_obs)
    n_ps = length(p0)
    scale = log(10.0) .* tnodes

    # Robertson's y2 is tiny, so an unweighted MSE mostly ignores it.
    if use_weighted_state_loss
        state_scale = vec(maximum(abs.(Y_obs); dims=1))
        state_weights = 1.0 ./ max.(state_scale, 1e-6).^2
    else
        state_weights = ones(d)
    end

    m = JuMP.Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level", ipopt_print_level)
    set_optimizer_attribute(m, "max_iter", max_iter)
    set_optimizer_attribute(m, "tol", 1e-7)
    set_optimizer_attribute(m, "constr_viol_tol", 1e-7)

    @variable(m, -parameter_bound <= psvar[1:n_ps] <= parameter_bound)

    if use_state_bounds #Perhaps change this. 
        @variable(m, 0 <= Y[1:N, 1:d] <= 1)
    else
        @variable(m, Y[1:N, 1:d])
    end

    set_start_value.(psvar, p0)
    set_start_value.(Y, Y_obs)

    # Register one scalar nonlinear operator per output component.
    # JuMP will pass scalar numeric values to rhsfun through these operators.
    for k in 1:d
        f_k = let rhsfun = rhsfun, d = d, k = k
            (args...) -> rhsfun(collect(args[1:d]), collect(args[d+1:end]))[k]
        end
        op_name = Symbol("nn_F$k")
        m[op_name] = add_nonlinear_operator(m, d + n_ps, f_k; name=op_name)
    end

    # Anchor the first collocation state. Since t_col[1] = tmin_train,
    # this is the IVP initial condition for the training interval.
    @constraint(m, [k = 1:d], Y[1, k] == Y_obs[1, k])

    # Robertson mass conservation.
    if use_mass_constraint
        @constraint(m, [i = 1:N], sum(Y[i, k] for k in 1:d) == 1.0)
    end

    # Collocation constraints in τ = log10(t):
    #     Dτ Y = log(10) * t * Fθ(Y).
    for i in 1:N
        for k in 1:d
            op = m[Symbol("nn_F$k")]
            @constraint(m,
                sum(Dτ[i, j] * Y[j, k] for j in 1:N)
                ==
                scale[i] * op(Y[i, 1:d]..., psvar...)
            )
        end
    end

    @objective(m, Min,
        sum(state_weights[k] * (Y[i, k] - Y_obs[i, k])^2 for i in 1:N for k in 1:d)
        + λ * sum(psvar[q]^2 for q in 1:n_ps)
    )

    optimize!(m)

    return value.(psvar), value.(Y), objective_value(m), termination_status(m)
end

# ---- ForwardDiff diagnostic ----
let n_ps = length(ps_vec), d = 3
    println("n_ps = $n_ps, operator arity = $(d + n_ps)")
    try
        r = F(ones(d), ones(n_ps))
        println("F(ones, ones) = $r  ✓")
    catch e
        println("F with Float64 FAILED:"); showerror(stdout, e); println()
    end
    try
        import ForwardDiff
        g = ForwardDiff.gradient(x -> F(collect(x[1:d]), collect(x[d+1:end]))[1], ones(d + n_ps))
        println("ForwardDiff gradient ✓, first 3: $(g[1:3])")
    catch e
        println("ForwardDiff FAILED:"); showerror(stdout, e); println()
        for fr in first(stacktrace(catch_backtrace()), 20); println(fr); end
    end
end
# --------------------------------

@time ps_opt, Y_opt, obj, status = collocation_train_logtime(
    F, τ_col, t_col, Dτ, Y_obs, ps_vec;
    λ = 1e-8,
    parameter_bound = 50.0,
    use_state_bounds = true,
    use_mass_constraint = true,
    use_weighted_state_loss = true,
    ipopt_print_level = 5,
    max_iter = 1000,
)

println("Ipopt status: ", status)
println("Objective: ", obj)

# -------------------------------
# Sequential prediction using the trained physical-time RHS
# -------------------------------
pred_prob = ODEProblem(F!, vec(Y_obs[1, :]), (t_col[1], t_col[end]), ps_opt)
pred = solve(pred_prob, Rodas5P(); saveat=t_col, reltol=1e-4, abstol=1e-4)
Y_pred = Matrix(reduce(hcat, pred.u)')

# For reference, the true data at the collocation points.
true_prob = ODEProblem(rober!, vec(Y_obs[1, :]), (t_col[1], t_col[end]))
true_col = solve(true_prob, Rodas5P(); saveat=t_col, reltol=1e-10, abstol=1e-12)
Y_true_col = Matrix(reduce(hcat, true_col.u)')

# -------------------------------
# Plots
# -------------------------------

# p1 = plot(X_obs, Y_obs_raw[:, 1]; xscale=:log10, label="true y1", lw=2)
# plot!(p1, t_col, Y_obs[:, 1]; label="obs/interp y1", lw=2, ls=:dash)
# plot!(p1, t_col, Y_opt[:, 1]; label="Y_opt y1", lw=2, ls=:dot)
# plot!(p1, t_col, Y_pred[:, 1]; label="pred y1", lw=2, ls=:dashdot)
# ylabel!(p1, "y1")

# p2 = plot(X_obs, Y_obs_raw[:, 2]; xscale=:log10, label="true y2", lw=2)
# plot!(p2, t_col, Y_obs[:, 2]; label="obs/interp y2", lw=2, ls=:dash)
# plot!(p2, t_col, Y_opt[:, 2]; label="Y_opt y2", lw=2, ls=:dot)
# plot!(p2, t_col, Y_pred[:, 2]; label="pred y2", lw=2, ls=:dashdot)
# ylabel!(p2, "y2")

# p3 = plot(X_obs, Y_obs_raw[:, 3]; xscale=:log10, label="true y3", lw=2)
# plot!(p3, t_col, Y_obs[:, 3]; label="obs/interp y3", lw=2, ls=:dash)
# plot!(p3, t_col, Y_opt[:, 3]; label="Y_opt y3", lw=2, ls=:dot)
# plot!(p3, t_col, Y_pred[:, 3]; label="pred y3", lw=2, ls=:dashdot)
# xlabel!(p3, "t")
# ylabel!(p3, "y3")

# plt = plot(p1, p2, p3; layout=(3, 1), size=(900, 850), title="Robertson collocation training in log-time")
# display(plt)

# Basic diagnostics
println("Max abs error, Y_opt vs obs: ", maximum(abs.(Y_opt .- Y_obs)))
println("Max abs error, pred  vs true: ", maximum(abs.(Y_pred .- Y_true_col)))
println("Mass drift in prediction: ", maximum(abs.(sum(Y_pred; dims=2) .- 1.0)))



# -------------------------------
# Plots: only observed vs predicted
# -------------------------------
p1 = plot(t_col, Y_obs[:, 1]; xscale=:log10, label="true y1", lw=2)
plot!(p1, t_col, Y_pred[:, 1]; label="predicted y1", lw=2, ls=:dash)
ylabel!(p1, "y1")

p2 = plot(t_col, Y_obs[:, 2]; xscale=:log10, label="true y2", lw=2)
plot!(p2, t_col, Y_pred[:, 2]; label="predicted y2", lw=2, ls=:dash)
ylabel!(p2, "y2");

p3 = plot(t_col, Y_obs[:, 3]; xscale=:log10, label="true y3", lw=2)
plot!(p3, t_col, Y_pred[:, 3]; label="predicted y3", lw=2, ls=:dash)
xlabel!(p3, "t");
ylabel!(p3, "y3");

plt = plot(
    p1, p2, p3;
    layout=(3, 1),
    size=(900, 850),
    title="Robertson Neural ODE: Stiffnet Deep"
)

display(plt)
savefig(plt, "robertson_collocation_logtime_stiffNetDeep.png")



#651s (10 min) for 1000 epochs of baseNN_deep. 
#696 s (11 min) for 1000 epochs of stiffNN_deep. 
# 99s (1.5min) for 1000 epochs of stiffNN_exp.
# 418 (7 min) for 1000 epochs of stiffNN_deep_exp.
#2091.546689 (34min) for 1000 epochs of stiffNN_superdeep.

#StiffNet Deep and StiffNet are the best. 


#Play with line 106 const Ncol = 100. Original was 50. 