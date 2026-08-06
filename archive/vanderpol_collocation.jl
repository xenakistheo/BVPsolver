using JuMP, Ipopt
using Lux, Random, ComponentArrays
using SciMLBase
using OrdinaryDiffEqRosenbrock
using Plots
using Statistics

include("src/layers/StiffLayer.jl")
include("src/derivativeMatching.jl")
# ============================================================
# Stiff Van der Pol + simultaneous training on a uniform grid
# ============================================================
# This is the same spirit as the Robertson file, but adapted to
# a uniform physical-time grid t ∈ [0, 200].
#
# IMPORTANT:
#   For N = 500 uniform points, do NOT use one global Lagrange
#   differentiation matrix on the full interval. That matrix is
#   extremely ill-conditioned on equispaced nodes.
#
#   Instead, this file uses a local collocation / fully-discretized
#   simultaneous formulation. By default it uses trapezoidal dynamics:
#
#       (Y[i+1] - Y[i]) / h = 0.5 * (Fθ(Y[i]) + Fθ(Y[i+1])).
#
#   This keeps the uniform timestep you asked for while avoiding the
#   global-polynomial pathology.

rng = Xoshiro(1)

# -------------------------------
# Stiff Van der Pol data
# -------------------------------
const μ = 10.0
const tspan = (0.0, 50.0)
const N = 100
const tsteps = collect(range(tspan[1], tspan[2]; length=N))
const u0 = [2.0, 0.0]

function vanderpol!(du, u, p, t)
    y1, y2 = u
    du[1] = y2
    du[2] = μ * (1 - y1^2) * y2 - y1
    return nothing
end

# Generate data with a stiff solver.
data_prob = ODEProblem(vanderpol!, u0, tspan)
data = solve(data_prob, Rodas5P(); saveat=tsteps, reltol=1e-10, abstol=1e-12)
plot(data)
X_obs = collect(data.t)
Y_obs = Matrix(reduce(hcat, data.u)')

# -------------------------------
# Scaling utilities
# -------------------------------
function true_rhs_at(u)
    du = zeros(length(u))
    vanderpol!(du, u, nothing, 0.0)
    return du
end

# Input scaling makes y1 and y2 comparable for the network.
const input_scale = vec(maximum(abs.(Y_obs); dims=1)) .+ 1e-12

# Output scaling lets a tanh network represent the derivative magnitudes.
Dtrue = Matrix(reduce(hcat, [true_rhs_at(vec(Y_obs[i, :])) for i in 1:size(Y_obs, 1)])')
const output_scale = max.(vec(maximum(abs.(Dtrue); dims=1)), 1.0)

function scale_state(u)
    return [u[1] / input_scale[1], u[2] / input_scale[2]]
end

# -------------------------------
# Neural RHS
# -------------------------------
# The NN represents the physical-time RHS dy/dt.
# We scale input and output for numerical conditioning.

sigma_affine_sigmoid(ps, j) = 1 / (1 + exp(-(ps[1] * j + ps[2])))
sigma_exp(ps, j) = exp(-(ps[1] * j))

V = Dense(2, 16, tanh)
S = StiffLayer2(sigma_affine_sigmoid; P=16, init=ones(16))
U = Chain(Dense(16, 16, tanh), Dense(16,2))

#370 params
StiffNN = Chain(V, S, U)

#354 params
baseNN = Chain(
    Dense(2, 16, tanh),
    Dense(16, 16, tanh),
    Dense(16, 2)
)

modelNN = StiffNN

ps, st = Lux.setup(rng, modelNN)
ps = Lux.f64(ps)
ps_ca = ComponentVector(ps)
ps_vec = collect(ps_ca)
ps_axes = getaxes(ps_ca)

function F(u, p)
    θ = ComponentVector(p, ps_axes)
    z = modelNN(scale_state(u), θ, st)[1]
    return output_scale .* z
end

function F!(du, u, p, t)
    du .= F(u, p)
    return nothing
end

# -------------------------------
# Simultaneous uniform-grid training
# -------------------------------
function collocation_train_uniform(rhsfun, tnodes, Y_obs, p0;
        λ = 1e-8,
        parameter_bound = 100.0,
        state_box_factor = 2.0,
        discretization = :trapezoid,   # :trapezoid or :backward_euler
        use_weighted_state_loss = false,
        ipopt_print_level = 5,
        max_iter = 1000,
    )

    N, d = size(Y_obs)
    n_ps = length(p0)

    @assert length(tnodes) == N
    @assert all(diff(tnodes) .> 0) "tnodes must be sorted and distinct"
    @assert discretization in (:trapezoid, :backward_euler)

    # State bounds: generous box around observed values.
    y_min = vec(minimum(Y_obs; dims=1))
    y_max = vec(maximum(Y_obs; dims=1))
    y_rng = max.(y_max .- y_min, 1.0)
    y_lower = y_min .- state_box_factor .* y_rng
    y_upper = y_max .+ state_box_factor .* y_rng

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
    @variable(m, Y[1:N, 1:d])

    for i in 1:N, k in 1:d
        set_lower_bound(Y[i, k], y_lower[k])
        set_upper_bound(Y[i, k], y_upper[k])
    end

    set_start_value.(psvar, p0)
    set_start_value.(Y, Y_obs)

    # Register one scalar nonlinear operator per output component.
    # JuMP passes scalar numeric values to rhsfun through these operators.
    for k in 1:d
        f_k = let rhsfun = rhsfun, d = d, k = k
            (args...) -> rhsfun(collect(args[1:d]), collect(args[d+1:end]))[k]
        end
        op_name = Symbol("nn_F$k")
        m[op_name] = add_nonlinear_operator(m, d + n_ps, f_k; name=op_name)
    end

    # Anchor the IVP initial condition.
    @constraint(m, [k = 1:d], Y[1, k] == Y_obs[1, k])

    # Local fully-discretized dynamics on the uniform physical-time grid.
    for i in 1:N-1
        h = tnodes[i+1] - tnodes[i]
        for k in 1:d
            op = m[Symbol("nn_F$k")]

            if discretization == :trapezoid
                @constraint(m,
                    (Y[i+1, k] - Y[i, k]) / h
                    ==
                    0.5 * (
                        op(Y[i, 1:d]..., psvar...)
                        + op(Y[i+1, 1:d]..., psvar...)
                    )
                )
            elseif discretization == :backward_euler
                @constraint(m,
                    (Y[i+1, k] - Y[i, k]) / h
                    ==
                    op(Y[i+1, 1:d]..., psvar...)
                )
            end
        end
    end

    data_loss = sum(
        state_weights[k] * (Y[i, k] - Y_obs[i, k])^2
        for i in 1:N for k in 1:d
    )

    param_reg = sum(psvar[q]^2 for q in 1:n_ps)

    @objective(m, Min, data_loss + λ * param_reg)

    optimize!(m)

    return value.(psvar), value.(Y), objective_value(m), termination_status(m)
end

@time ps_opt, Y_opt, obj, status = collocation_train_uniform(
    F, X_obs, Y_obs, ps_vec;
    λ = 1e-8,
    parameter_bound = 100.0,
    state_box_factor = 2.0,
    discretization = :trapezoid,
    use_weighted_state_loss = false,
    ipopt_print_level = 5,
    max_iter = 1000,
)

println("Ipopt status: ", status)
println("Objective: ", obj)
println("Mean abs error per component, Y_opt vs obs: ", vec(mean(abs.(Y_opt .- Y_obs); dims=1)))
println("Max  abs error per component, Y_opt vs obs: ", vec(maximum(abs.(Y_opt .- Y_obs); dims=1)))

# -------------------------------
# Sequential prediction using the trained physical-time RHS
# -------------------------------
pred_prob = ODEProblem(F!, vec(Y_obs[1, :]), tspan, ps_opt)
pred = solve(pred_prob, Rodas5P(); saveat=X_obs, reltol=1e-8, abstol=1e-10)
Y_pred = Matrix(reduce(hcat, pred.u)')

# Reference true data at the same grid. Since data is generated at X_obs,
# this is mostly a plotting/diagnostic convenience.
Y_true = Y_obs

println("Mean abs error per component, pred vs obs: ", vec(mean(abs.(Y_pred .- Y_obs); dims=1)))
println("Max  abs error per component, pred vs obs: ", vec(maximum(abs.(Y_pred .- Y_obs); dims=1)))

# -------------------------------
# Plots: true/obs, optimized state, and predicted rollout
# -------------------------------
p1 = plot(X_obs, Y_true[:, 1]; label="true/obs y1", lw=2)
plot!(p1, X_obs, Y_opt[:, 1]; label="Y_opt y1", lw=2, ls=:dash)
plot!(p1, X_obs, Y_pred[:, 1]; label="pred y1", lw=2, ls=:dot)
ylabel!(p1, "y1")

p2 = plot(X_obs, Y_true[:, 2]; label="true/obs y2", lw=2)
plot!(p2, X_obs, Y_opt[:, 2]; label="Y_opt y2", lw=2, ls=:dash)
plot!(p2, X_obs, Y_pred[:, 2]; label="pred y2", lw=2, ls=:dot)
xlabel!(p2, "t")
ylabel!(p2, "y2")

plt = plot(p1, p2; layout=(2, 1), size=(950, 700), title="Stiff Van der Pol: simultaneous uniform-grid training")
display(plt)

# Clean observed-vs-predicted plot only.
p1 = plot(X_obs, Y_obs[:, 1]; label="observed y1", lw=2)
plot!(p1, X_obs, Y_pred[:, 1]; label="predicted y1", lw=2, ls=:dash)
p2 = plot(X_obs, Y_obs[:, 2]; label="observed y2", lw=2)
plot!(p2, X_obs, Y_pred[:, 2]; label="predicted y2", lw=2, ls=:dash)

plt = plot(
    p1, p2;
    layout=(3, 1),
    size=(900, 850),
    title="Van der Pol Neural ODE: StiffNN"
)

display(plt)
savefig(plt, "vanderpol_collocation_StiffNN_1000epochs.png")
