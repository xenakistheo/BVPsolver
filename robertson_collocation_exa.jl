using ExaModels, MadNLP
using Lux, Random
using SciMLBase
using OrdinaryDiffEqRosenbrock
using Plots

include("src/layers/StiffLayer.jl")
include("src/derivativeMatching.jl")

# ============================================================
# Robertson + simultaneous collocation training in log-time
# -- ExaModels / MadNLP backend --
# ============================================================
# This is the ExaModels/MadNLP counterpart of `robertson_collocation.jl`.
# The data, collocation grid, neural architecture (StiffNN_Deep), training
# objective, and constraints are identical to the JuMP/Ipopt version. Only
# the optimization backend differs.
#
# JuMP could embed the Lux forward pass directly as a single opaque
# registered nonlinear operator (JuMP differentiates through it via
# ForwardDiff). ExaModels has no such black-box registration: every
# constraint/objective must be a plain algebraic `Generator` expression, and
# subexpressions (`@add_expr`) are *inlined* rather than memoized, so naively
# chaining several 8-wide Dense layers as nested subexpressions blows up the
# expression tree combinatorially (verified experimentally -- it stack
# overflows). Instead, each hidden layer's activations become their own
# lifted decision variables, tied to the previous layer by explicit equality
# constraints. This is mathematically the same forward pass (verified below
# against the Lux network directly), just written as a "lifted" NLP
# encoding instead of a black-box function call -- the standard way to
# embed a small MLP in an algebraic modeling system.

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
# Neural RHS (StiffNN_Deep, 346 params)
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

sigma_affine_sigmoid(ps, j) = 1 / (1 + exp(-(ps[1] * j + ps[2])))

V2 = Chain(Dense(3, 8, tanh), Dense(8, 8, tanh), Dense(8, 8, tanh))
S2 = StiffLayer2(sigma_affine_sigmoid; P=8, init=ones(8))
U2 = Chain(Dense(8, 8, tanh), Dense(8, 8, tanh), Dense(8, 2))

StiffNN_Deep = Chain(V2, S2, U2) # 346 params

modelNN = StiffNN_Deep

# Same rng seed and same architecture as robertson_collocation.jl, so this
# produces bit-identical initial parameter values to the JuMP/Ipopt version.
ps0, st = Lux.setup(rng, modelNN)
ps0 = Lux.f64(ps0)

function F(u, p)
    z = modelNN(scale_state(u), p, st)[1]
    return [z[1], z[2], -z[1] - z[2]]
end

function F!(du, u, p, t)
    du .= F(u, p)
    return nothing
end

# ---- sanity check: hand-written arithmetic used inside ExaModels must
#      match the Lux forward pass exactly ----
function nn_forward_raw(u, ps)
    x1, x2, x3 = scale_state(u)
    W1, b1 = ps.layer_1.layer_1.weight, ps.layer_1.layer_1.bias
    W2, b2 = ps.layer_1.layer_2.weight, ps.layer_1.layer_2.bias
    W3, b3 = ps.layer_1.layer_3.weight, ps.layer_1.layer_3.bias
    Sps    = ps.layer_2.ps
    W4, b4 = ps.layer_3.layer_1.weight, ps.layer_3.layer_1.bias
    W5, b5 = ps.layer_3.layer_2.weight, ps.layer_3.layer_2.bias
    W6, b6 = ps.layer_3.layer_3.weight, ps.layer_3.layer_3.bias

    h1 = [tanh(W1[a,1]*x1 + W1[a,2]*x2 + W1[a,3]*x3 + b1[a]) for a in 1:8]
    h2 = [tanh(sum(W2[a,b]*h1[b] for b in 1:8) + b2[a]) for a in 1:8]
    h3 = [tanh(sum(W3[a,b]*h2[b] for b in 1:8) + b3[a]) for a in 1:8]
    g  = [h3[a] / (1 + exp(-(Sps[1]*a + Sps[2]))) for a in 1:8]
    h4 = [tanh(sum(W4[a,b]*g[b] for b in 1:8) + b4[a]) for a in 1:8]
    h5 = [tanh(sum(W5[a,b]*h4[b] for b in 1:8) + b5[a]) for a in 1:8]
    z  = [sum(W6[c,b]*h5[b] for b in 1:8) + b6[c] for c in 1:2]
    return [z[1], z[2], -z[1] - z[2]]
end

let
    u_test = [0.6, 1e-4, 0.4]
    lux_out = F(u_test, ps0)
    raw_out = nn_forward_raw(u_test, ps0)
    println("F(u_test) via Lux:              ", lux_out)
    println("F(u_test) via raw arithmetic:   ", raw_out)
    println("match: ", isapprox(lux_out, raw_out; atol=1e-12), " ✓")
end

# -------------------------------
# ExaModels model construction
# -------------------------------
# Lifted encoding: each hidden layer's post-activation values are their own
# variables (H1, H2, H3, G, H4, H5, Z), tied to the previous layer by
# equality constraints. Weight/bias blocks are ExaModels variables (so they
# are optimized, exactly like `psvar` in the JuMP version), started at the
# same Lux-initialized values `ps0`.
function build_collocation_model(τnodes, tnodes, Dτ, Y_obs, ps0;
        λ = 1e-8,
        parameter_bound = 50.0,
        use_state_bounds = true,
        use_mass_constraint = true,
        use_weighted_state_loss = true,
    )

    N, d = size(Y_obs)
    scale = log(10.0) .* tnodes
    pb = parameter_bound

    if use_weighted_state_loss
        state_scale = vec(maximum(abs.(Y_obs); dims=1))
        state_weights = 1.0 ./ max.(state_scale, 1e-6).^2
    else
        state_weights = ones(d)
    end

    W1_0, b1_0 = ps0.layer_1.layer_1.weight, ps0.layer_1.layer_1.bias
    W2_0, b2_0 = ps0.layer_1.layer_2.weight, ps0.layer_1.layer_2.bias
    W3_0, b3_0 = ps0.layer_1.layer_3.weight, ps0.layer_1.layer_3.bias
    Sps_0      = ps0.layer_2.ps
    W4_0, b4_0 = ps0.layer_3.layer_1.weight, ps0.layer_3.layer_1.bias
    W5_0, b5_0 = ps0.layer_3.layer_2.weight, ps0.layer_3.layer_2.bias
    W6_0, b6_0 = ps0.layer_3.layer_3.weight, ps0.layer_3.layer_3.bias

    c = ExaCore(concrete = Val(true))

    @add_par(c, Y_obsp, Y_obs)
    @add_par(c, Dτp, Dτ)
    @add_par(c, scalep, scale)
    @add_par(c, swp, state_weights)

    @add_var(c, W1, 8, 3; start = W1_0, lvar = -pb, uvar = pb)
    @add_var(c, b1, 8;    start = b1_0, lvar = -pb, uvar = pb)
    @add_var(c, W2, 8, 8; start = W2_0, lvar = -pb, uvar = pb)
    @add_var(c, b2, 8;    start = b2_0, lvar = -pb, uvar = pb)
    @add_var(c, W3, 8, 8; start = W3_0, lvar = -pb, uvar = pb)
    @add_var(c, b3, 8;    start = b3_0, lvar = -pb, uvar = pb)
    @add_var(c, Sps, 8;   start = Sps_0, lvar = -pb, uvar = pb)
    @add_var(c, W4, 8, 8; start = W4_0, lvar = -pb, uvar = pb)
    @add_var(c, b4, 8;    start = b4_0, lvar = -pb, uvar = pb)
    @add_var(c, W5, 8, 8; start = W5_0, lvar = -pb, uvar = pb)
    @add_var(c, b5, 8;    start = b5_0, lvar = -pb, uvar = pb)
    @add_var(c, W6, 2, 8; start = W6_0, lvar = -pb, uvar = pb)
    @add_var(c, b6, 2;    start = b6_0, lvar = -pb, uvar = pb)

    if use_state_bounds
        @add_var(c, Y, N, d; start = Y_obs, lvar = 0.0, uvar = 1.0)
    else
        @add_var(c, Y, N, d; start = Y_obs)
    end

    # Lifted hidden-layer variables (tanh/gate outputs are bounded in
    # (-1, 1) by construction; Z, the pre-conservation-split raw network
    # output, is left free).
    @add_var(c, H1, N, 8; start = 0.0, lvar = -1.0, uvar = 1.0)
    @add_var(c, H2, N, 8; start = 0.0, lvar = -1.0, uvar = 1.0)
    @add_var(c, H3, N, 8; start = 0.0, lvar = -1.0, uvar = 1.0)
    @add_var(c, G,  N, 8; start = 0.0, lvar = -1.0, uvar = 1.0)
    @add_var(c, H4, N, 8; start = 0.0, lvar = -1.0, uvar = 1.0)
    @add_var(c, H5, N, 8; start = 0.0, lvar = -1.0, uvar = 1.0)
    @add_var(c, Z,  N, 2; start = 0.0)

    # V2: Dense(3,8,tanh) -> Dense(8,8,tanh) -> Dense(8,8,tanh)
    @add_con(c, H1[i,a] - tanh(W1[a,1]*Y[i,1] + W1[a,2]*100.0*Y[i,2] + W1[a,3]*Y[i,3] + b1[a]) for i in 1:N, a in 1:8)
    @add_con(c, H2[i,a] - tanh(sum(W2[a,b]*H1[i,b] for b in 1:8) + b2[a]) for i in 1:N, a in 1:8)
    @add_con(c, H3[i,a] - tanh(sum(W3[a,b]*H2[i,b] for b in 1:8) + b3[a]) for i in 1:N, a in 1:8)
    # S2: StiffLayer2 elementwise gate, sigma_affine_sigmoid(Sps, a)
    @add_con(c, G[i,a] - H3[i,a] / (1 + exp(-(Sps[1]*a + Sps[2]))) for i in 1:N, a in 1:8)
    # U2: Dense(8,8,tanh) -> Dense(8,8,tanh) -> Dense(8,2) (linear output)
    @add_con(c, H4[i,a] - tanh(sum(W4[a,b]*G[i,b] for b in 1:8) + b4[a]) for i in 1:N, a in 1:8)
    @add_con(c, H5[i,a] - tanh(sum(W5[a,b]*H4[i,b] for b in 1:8) + b5[a]) for i in 1:N, a in 1:8)
    @add_con(c, Z[i,cc] - (sum(W6[cc,b]*H5[i,b] for b in 1:8) + b6[cc]) for i in 1:N, cc in 1:2)

    # Anchor the first collocation state. Since t_col[1] = tmin_train,
    # this is the IVP initial condition for the training interval.
    @add_con(c, Y[1,k] - Y_obsp[1,k] for k in 1:d)

    # Robertson mass conservation.
    if use_mass_constraint
        @add_con(c, sum(Y[i,k] for k in 1:d) - 1.0 for i in 1:N)
    end

    # Collocation constraints in τ = log10(t):
    #     Dτ Y = log(10) * t * Fθ(Y),  Fθ = (Z[:,1], Z[:,2], -(Z[:,1]+Z[:,2])).
    @add_con(c, sum(Dτp[i,j]*Y[j,1] for j in 1:N) - scalep[i]*Z[i,1] for i in 1:N)
    @add_con(c, sum(Dτp[i,j]*Y[j,2] for j in 1:N) - scalep[i]*Z[i,2] for i in 1:N)
    @add_con(c, sum(Dτp[i,j]*Y[j,3] for j in 1:N) - scalep[i]*(-(Z[i,1] + Z[i,2])) for i in 1:N)

    @add_obj(c, swp[k]*(Y[i,k] - Y_obsp[i,k])^2 for i in 1:N, k in 1:d)

    c, _ = add_obj(c, λ*W1[a,b]^2 for a in 1:8, b in 1:3)
    c, _ = add_obj(c, λ*b1[a]^2 for a in 1:8)
    c, _ = add_obj(c, λ*W2[a,b]^2 for a in 1:8, b in 1:8)
    c, _ = add_obj(c, λ*b2[a]^2 for a in 1:8)
    c, _ = add_obj(c, λ*W3[a,b]^2 for a in 1:8, b in 1:8)
    c, _ = add_obj(c, λ*b3[a]^2 for a in 1:8)
    c, _ = add_obj(c, λ*Sps[a]^2 for a in 1:8)
    c, _ = add_obj(c, λ*W4[a,b]^2 for a in 1:8, b in 1:8)
    c, _ = add_obj(c, λ*b4[a]^2 for a in 1:8)
    c, _ = add_obj(c, λ*W5[a,b]^2 for a in 1:8, b in 1:8)
    c, _ = add_obj(c, λ*b5[a]^2 for a in 1:8)
    c, _ = add_obj(c, λ*W6[cc,b]^2 for cc in 1:2, b in 1:8)
    c, _ = add_obj(c, λ*b6[cc]^2 for cc in 1:2)

    vars = (; W1, b1, W2, b2, W3, b3, Sps, W4, b4, W5, b5, W6, b6, Y)
    return c, vars
end

# -------------------------------
# Simultaneous collocation training (ExaModels / MadNLP)
# -------------------------------
function collocation_train_logtime_exa(τnodes, tnodes, Dτ, Y_obs, ps0;
        λ = 1e-8,
        parameter_bound = 50.0,
        use_state_bounds = true,
        use_mass_constraint = true,
        use_weighted_state_loss = true,
        madnlp_print_level = MadNLP.INFO,
        max_iter = 3000,
        tol = 1e-7,
    )

    c, vars = build_collocation_model(
        τnodes, tnodes, Dτ, Y_obs, ps0;
        λ, parameter_bound, use_state_bounds, use_mass_constraint, use_weighted_state_loss,
    )
    m = ExaModel(c)

    result = madnlp(m;
        print_level = madnlp_print_level,
        max_iter = max_iter,
        tol = tol,
        hessian_approximation = MadNLP.CompactLBFGS, # matches Ipopt "limited-memory"
    )

    ps_opt = (
        layer_1 = (
            layer_1 = (weight = solution(result, vars.W1), bias = solution(result, vars.b1)),
            layer_2 = (weight = solution(result, vars.W2), bias = solution(result, vars.b2)),
            layer_3 = (weight = solution(result, vars.W3), bias = solution(result, vars.b3)),
        ),
        layer_2 = (ps = solution(result, vars.Sps),),
        layer_3 = (
            layer_1 = (weight = solution(result, vars.W4), bias = solution(result, vars.b4)),
            layer_2 = (weight = solution(result, vars.W5), bias = solution(result, vars.b5)),
            layer_3 = (weight = solution(result, vars.W6), bias = solution(result, vars.b6)),
        ),
    )
    Y_opt = solution(result, vars.Y)

    return ps_opt, Y_opt, result.objective, result.status
end

@time ps_opt, Y_opt, obj, status = collocation_train_logtime_exa(
    τ_col, t_col, Dτ, Y_obs, ps0;
    λ = 1e-8,
    parameter_bound = 50.0,
    use_state_bounds = true,
    use_mass_constraint = true,
    use_weighted_state_loss = true,
    madnlp_print_level = MadNLP.INFO,
    max_iter = 4000,
    tol = 1e-7,
)

println("MadNLP status: ", status)
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
    title="Robertson Neural ODE: StiffNet Deep (ExaModels/MadNLP)"
)

display(plt)
savefig(plt, "robertson_collocation_logtime_stiffNetDeep_exa.png")
