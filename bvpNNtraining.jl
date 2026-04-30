using BoundaryValueDiffEqFIRK, BoundaryValueDiffEqMIRK
using OrdinaryDiffEqTsit5
using Plots


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

N_params = 18
p0 = randn(N_params) #Initialise parameters
u0_guess = Y[:, 1]   # IC comes from the data

# ============================================================
# Training via MIRK4's `optimize` slot with pure-penalty AugLag (λ ≡ 0, fixed ρ).
# BC is the known IC; interior data points go into the cost.
# MIRK4's optimize path builds an equality-constrained OptimizationProblem
# (cons = collocation + BC residual, lcons = ucons = 0); AugLag with clamped
# multipliers reduces it to `min data_loss + (ρ/2)‖c‖²`.
# ============================================================
using BoundaryValueDiffEqCore: BVPVerbosity
# using SciMLLogging: Detailed
using OptimizationAuglag, OptimizationOptimisers

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

bvp_fun_train = BVPFunction(F!, ic_bc!;
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
NN_sol_al = solve(bvp_train,
    MIRK4(; optimize = OptimizationAuglag.AugLag(;
        inner = Adam(0.01),
        inner_maxiters = 5000,
        γ = 1.0,
        λmin = 0.0, λmax = 0.0,
        μmin = 0.0, μmax = 0.0,
        ρ_init = 10.0,
        ϵ_primal = 1e-3,
        ϵ_dual = 1e6,
    ));
    dt = 0.05,
    adaptive = false,
    verbose = BVPVerbosity(Detailed()),
    optimize_kwargs = (; maxiters = 5, callback = al_callback),
)

# Loss curve — every cost_fn evaluation (many per outer AL iter).
plot(losses; xlabel="cost evaluation", ylabel="data loss", yscale=:log10,
     title="AL + Adam on BVP-NN", legend=false)

# TODO: try with RadauIIa5/7 (L-stable FIRK) for stiff variants of this problem.
# The pure-penalty AugLag config carries over unchanged; change MIRK4(...) to
# RadauIIa5(...).

using SciMLBase
using BoundaryValueDiffEq
using Optimization

@show pathof(BoundaryValueDiffEq)
@show pathof(BoundaryValueDiffEqFIRK)
@show pathof(BoundaryValueDiffEqMIRK)
@show pathof(Optimization)
@show pathof(OptimizationAuglag)
@show pathof(SciMLBase)
@show pathof(BoundaryValueDiffEqCore)
