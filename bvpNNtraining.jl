using BoundaryValueDiffEqFIRK, BoundaryValueDiffEqMIRK
using OrdinaryDiffEqTsit5
using Plots
using Lux
using Random, ComponentArrays
using BoundaryValueDiffEqCore: BVPVerbosity
# using SciMLLogging: Detailed
using OptimizationAuglag, OptimizationOptimisers
using NonlinearSolve: NewtonRaphson
using LinearSolve: KrylovJL_GMRES
include("src/layers/StiffLayer.jl")

rng = Xoshiro()

# True ODE
function spiralODE(du, u, p, t)
    a = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)' * a)'
end

tspan = (0.0, 1.5)
tlen = 30
tsteps = collect(range(tspan[1], tspan[2]; length=tlen))

# Create training data
data_prob = ODEProblem(spiralODE, [2.0, 0.0], tspan)
data = solve(data_prob, Tsit5(), saveat=tsteps)
# plot(data)
X, Y = data.t, reduce(hcat, data.u)


# Define Neural Network
function NN(u, p)
    W1, W2, W3 = reshape(p[1:4],2,2), reshape(p[5:8],2,2), reshape(p[9:12],2,2)
    b1, b2, b3 = p[13:14], p[15:16], p[17:18]

    h1 = W1 * u + b1
    z1 = tanh.(h1)
    h2 = W2 * z1 + b2
    z2 = tanh.(h2)
    return W3 * z2 + b3
end

model = Chain(
    Dense(2, 4, tanh),
    Dense(4, 2)
)

sigma_affine_sigmoid(ps, j) = 1 / (1 + exp(-(ps[1] * j + ps[2])))
sigma_exp(ps, j) = exp(-(ps[1] * j))

V = Dense(2, 4, tanh)
S = StiffLayer2(sigma_affine_sigmoid; P=4, init=ones(4))
U = Chain(Dense(4, 2, tanh), Dense(2,2))

model2 = Chain(V, S, U)

ps, st = Lux.setup(rng, model2)#change to model
ps = Lux.f64(ps)

ps_ca = ComponentVector(ps)
length(ps_ca)
ps_vec = collect(ps_ca)
ps_axes = getaxes(ps_ca)

# Define "Differential Equation".
# With `tune_parameters=true`, the BVP solver augments the state to
# `[state(2); params(18)]` and wraps `F!` (see mirk.jl:181-195 /
# firk.jl:215-220) so that:
#   - `p` arrives already sliced from `u[3:20]` (the current trained params)
#   - `du[3:20]` is auto-zeroed after this returns
# After solve, trained params are exposed via `sol.prob.p` (remade at mirk.jl:293).
function F!(du, u, p, t)
    du[1:2] .= NN(@view(u[1:2]), p)
end

# Change so that it does not allocate every time when building the ComponentVector
function F2!(du, u, p, t)
    du[1:2] .= model2(@view(u[1:2]), ComponentVector(p, ps_axes), st)[1]
end

N_params = length(ps_ca)
p0 = randn(N_params) #Initialise parameters
const p0_copy = deepcopy(p0)
u0_guess = Y[:, 1]   # IC comes from the data

# ============================================================
# Training via MIRK4's `optimize` slot with pure-penalty AugLag (λ ≡ 0, fixed ρ).
# BC is the known IC; interior data points go into the cost.
# MIRK4's optimize path builds an equality-constrained OptimizationProblem
# (cons = collocation + BC residual, lcons = ucons = 0); AugLag with clamped
# multipliers reduces it to `min data_loss + (ρ/2)‖c‖²`.
# ============================================================


# Data-fitting loss. IC is handled by the BC, so we start at i=2.
# Side-channel: log each scalar eval for a fine-grained loss curve.
losses = Float64[]
cost_fn = (sol, p) -> begin
    s = zero(eltype(sol(X[2])))
    for i in 2:tlen                      # skip i=1 — it's the BC
        r1 = sol(X[i])[1] - Y[1, i]
        r2 = sol(X[i])[2] - Y[2, i]
        s += r1*r1 + r2*r2
    end
    s isa Float64 && push!(losses, s)    # skip AD duals
    s
end

# Hard BC on the known initial condition.
ic_bc!(res, sol, _p, _t) = begin
    res[1] = sol(X[1])[1] - Y[1, 1]
    res[2] = sol(X[1])[2] - Y[2, 1]
end

bvp_fun_train = BVPFunction(F2!, ic_bc!;
    bcresid_prototype = zeros(2),
    cost = cost_fn,
)
bvp_train = BVProblem(bvp_fun_train, u0_guess, tspan, p0; tune_parameters = true)

# Callback, one fire per outer iteration. In pure-penalty mode ρ is fixed and
# λ ≡ 0, so the only meaningful signals are raw_loss and r_primal.
# r_dual scales with ρ × inner Adam stationarity error; informational only.
al_callback = function (state, obj)
    o = state.original
    raw_loss = isempty(losses) ? NaN : losses[end]
    @info "AL" iter=state.iter obj=obj raw_loss=raw_loss r_primal=o.r_primal r_dual=o.r_dual ρ=o.ρ
    return false
end

empty!(losses)
# γ=1, λmin=λmax=0, μmin=μmax=0 → pure penalty (no multiplier, no ρ growth).
# ρ_init pins the penalty weight directly — cleaner than capping the auto-scaled
# init with ρmax. ϵ_dual huge since r_dual isn't meaningful here.
# Outer iters are mathematically redundant (same subproblem, warm-started Adam),
# so use few outer × many inner.
@time NN_sol_al_mirk = solve(bvp_train,
    MIRK4(; optimize = OptimizationAuglag.AugLag(;
        inner = Adam(1e-2),
        inner_kwargs = (maxiters = 100,), 
        γ = 2.0,
        λmin = -1e6, λmax = 1e6,
        μmin = 0.0, μmax = 1e6,
        ρ_init = 10.0,
        ϵ_primal = 1e-2,
        ϵ_dual = 1e6,
    ));
    dt = 0.02,
    adaptive = false,
    verbose = BVPVerbosity(Detailed()),
    optimize_kwargs = (; maxiters = 50, callback = al_callback),
) #17.9 sec on second run. 

# # Loss curve — every cost_fn evaluation (many per outer AL iter).
# plot(losses; xlabel="cost evaluation", ylabel="data loss", yscale=:log10,
#      title="AL + Adam on BVP-NN", legend=false)

# Doesn't work well 
empty!(losses)

@time NN_sol_al_radau = solve(bvp_train,
    RadauIIa5(; optimize = OptimizationAuglag.AugLag(;
        inner = Adam(1e-2),
        inner_kwargs = (maxiters = 100,), 
        γ = 2.0,
        λmin = -1e6, λmax = 1e6,
        μmin = 0.0, μmax = 1e6,
        ρ_init = 10.0,
        ϵ_primal = 1e-2,
        ϵ_dual = 1e6,
    ));
    dt = 0.02,
    adaptive = false,
    verbose = BVPVerbosity(Detailed()),
    optimize_kwargs = (; maxiters = 10, callback = al_callback),
);


# # Loss curve — every cost_fn evaluation (many per outer AL iter).
# plot(losses; xlabel="cost evaluation", ylabel="data loss", yscale=:log10,
#      title="AL + Adam on BVP-NN", legend=false)


#####
# empty!(losses)

# @time NN_sol_krylov_mirk = solve(bvp_train,
#     MIRK4(; nlsolve = NewtonRaphson());
#     dt = 0.05,
#     adaptive = false,
#     verbose = BVPVerbosity(Detailed()),
#     optimize_kwargs = (; maxiters = 100, callback = al_callback),
# );

# ### Replace optimization problem with nonlinear problem (Newton-Krylov)
# empty!(losses)

# @time NN_sol_krylov_radau = solve(bvp_train,
#     RadauIIa5(; nlsolve = NewtonRaphson(linsolve = KrylovJL_GMRES()));
#     dt = 0.05,
#     adaptive = false,
#     verbose = BVPVerbosity(Detailed()),
#     optimize_kwargs = (; maxiters = 100, callback = al_callback),
# );


# # Loss curve — every cost_fn evaluation (many per outer AL iter).
# plot(losses; xlabel="cost evaluation", ylabel="data loss", yscale=:log10,
#      title="AL + Adam on BVP-NN", legend=false)


# Solve using trained parameters. 
pred_prob_al_mirk = ODEProblem(F2!, [2.0, 0.0], tspan, NN_sol_al_mirk.prob.p)
pred_al_mirk = solve(pred_prob_al_mirk, Tsit5(), saveat=tsteps)

pred_prob_al_radau = ODEProblem(F2!, [2.0, 0.0], tspan, NN_sol_al_radau.prob.p)
pred_al_radau = solve(pred_prob_al_radau, Tsit5(), saveat=tsteps)

# pred_prob_krylov_mirk = ODEProblem(F2!, [2.0, 0.0], tspan, NN_sol_krylov_mirk.prob.p)
# pred_krylov_mirk = solve(pred_prob_krylov_mirk, Tsit5(), saveat=tsteps) 

# pred_prob_krylov_radau = ODEProblem(F2!, [2.0, 0.0], tspan, NN_sol_krylov_radau.prob.p)
# pred_krylov_radau = solve(pred_prob_krylov_radau, Tsit5(), saveat=tsteps)


# Compare with intial parameters
pretrained_pred_prob = ODEProblem(F2!, [2.0, 0.0], tspan, p0)
pretrained_pred = solve(pretrained_pred_prob, Tsit5(), saveat=tsteps)



p1 = plot(data, labels=["u₁ (data)" "u₂ (data)"], title="Data + Pretrained")
plot!(p1, pretrained_pred, labels=["u₁ (Pretrained)" "u₂ (Pretrained)"], linestyle=:dash)
# plot!(p1, pred_al_mirk, labels=["u₁" "u₂"], linestyle=:dash)


p2 = plot(data, labels=["u₁ (data)" "u₂ (data)"], title="AL + MIRK4")
plot!(p2, pred_al_mirk, labels=["u₁" "u₂"], linestyle=:dash)

p3 = plot(data, labels=["u₁ (data)" "u₂ (data)"], title="AL + Radau")
plot!(p3, pred_al_radau, labels=["u₁" "u₂"], linestyle=:dash)

# p4 = plot(data, labels=["u₁ (data)" "u₂ (data)"], title="Krylov + MIRK4")
# plot!(p4, pred_krylov_mirk, labels=["u₁" "u₂"], linestyle=:dash)

# p5 = plot(data, labels=["u₁ (data)" "u₂ (data)"], title="Krylov + Radau")
# plot!(p5, pred_krylov_radau, labels=["u₁" "u₂"], linestyle=:dash)



plot(p1, p2, p3; layout=(1, 3), size=(1200, 400))


param_plot = plot(p0, label="Initial", title="Trained Parameters")
plot!(param_plot, NN_sol_al_mirk.prob.p, label="MIRK4 + AL")
# plot!(param_plot, NN_sol_krylov_mirk.prob.p, label="MIRK4 + Krylov")
plot!(param_plot, NN_sol_al_radau.prob.p, label="Radau + AL")
# plot!(param_plot, NN_sol_krylov_radau.prob.p, label="Radau + Krylov", linestyle=:dash)
param_plot


# using SciMLBase
# using BoundaryValueDiffEq
# using Optimization

# @show pathof(BoundaryValueDiffEq)
# @show pathof(BoundaryValueDiffEqFIRK)
# @show pathof(BoundaryValueDiffEqMIRK)
# @show pathof(Optimization)
# @show pathof(OptimizationAuglag)
# @show pathof(SciMLBase)
# @show pathof(BoundaryValueDiffEqCore)

# @show pathof(OptimizationBase)

# using SciMLLogging
# @show pathof(SciMLLogging)

