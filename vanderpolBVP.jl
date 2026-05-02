using BoundaryValueDiffEqFIRK, BoundaryValueDiffEqMIRK
using OrdinaryDiffEqTsit5
using OrdinaryDiffEqRosenbrock
using Plots
using Lux
using Random, ComponentArrays
using BenchmarkTools
using Ipopt
using OptimizationMOI

include("src/layers/StiffLayer.jl")

rng = Xoshiro()

# True ODE
function spiralODE(du, u, p, t)
    a = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)' * a)'
end


function vanderpol(du, u, p, t)
    k1, k2, k3 = 0.04, 3e7, 1e4
    y1, y2, y3 = u
    du[1] = -k1 * y1 + k3 * y2 * y3
    du[2] = k1 * y1 - k2 * y2^2 - k3 * y2 * y3
    du[3] = k2 * y2^2
end 

const tspan = (1e-6, 1e5)
const tlen = 100

const tsteps = collect(10.0 .^ range(log10(tspan[1]), log10(tspan[2]); length=tlen))

# Create training data
data_prob = ODEProblem(vanderpol, [1.0, 0.0, 0.0], tspan)
data = solve(data_prob, Rodas5P(), saveat=tsteps)
# plot(data)
const X, Y = data.t, reduce(hcat, data.u)


# Define Neural Network
const model = Chain(
    Dense(3, 4, tanh),
    Dense(4, 4, tanh),
    Dense(4, 3)
)

sigma_affine_sigmoid(ps, j) = 1 / (1 + exp(-(ps[1] * j + ps[2])))
sigma_exp(ps, j) = exp(-(ps[1] * j))

V = Dense(2, 4, tanh)
S = StiffLayer2(sigma_affine_sigmoid; P=4, init=ones(4))
U = Chain(Dense(4, 2, tanh), Dense(2,2))

model2 = Chain(V, S, U)

ps, _st = Lux.setup(rng, model)
ps = Lux.f64(ps)
const st = _st

ps_ca = ComponentVector(ps)
length(ps_ca)
ps_vec = collect(ps_ca)
const ps_axes = getaxes(ps_ca)

# Define "Differential Equation".
# With `tune_parameters=true`, the BVP solver augments the state to
# `[state(2); params(18)]` and wraps `F!` (see mirk.jl:181-195 /
# firk.jl:215-220) so that:
#   - `p` arrives already sliced from `u[3:20]` (the current trained params)
#   - `du[3:20]` is auto-zeroed after this returns
# After solve, trained params are exposed via `sol.prob.p` (remade at mirk.jl:293).
N_params = 51 #18
p0 = randn(N_params) #Initialise parameters
u0_guess = Y[:, 1]   # IC comes from the data



function F!(du, u, p, t)
    du[1:3] .= model(@view(u[1:3]), ComponentVector(p, ps_axes), st)[1]
    return nothing
end

F!(similar(u0_guess), u0_guess, p0, 0.0)                                                                                                                                                                                
@benchmark F!(du, $u0_guess, $p0, 0.0) setup=(du=similar(u0_guess))
@allocated F!(similar(u0_guess), u0_guess, p0, 0.0) # MAAANY allocations 
@allocated ComponentVector(p0, ps_axes)
@allocated model(@view(u0_guess[1:3]), ComponentVector(p0, ps_axes), st) 


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
    s = zero(eltype(sol.u[2][1]))
    for i in 2:tlen                      # skip i=1 — it's the BC
        ui = sol.u[i]
        r1 = ui[1] - Y[1, i]
        r2 = ui[2] - Y[2, i]
        r3 = ui[3] - Y[3, i]
        s += r1*r1 + r2*r2 + r3*r3
    end
    s isa Float64 && push!(losses, s)    # skip AD duals
    s
end

# Hard BC on the known initial condition.
ic_bc!(res, sol, _p, _t) = begin
    u1 = sol.u[1]
    res[1] = u1[1] - Y[1, 1]
    res[2] = u1[2] - Y[2, 1]
    res[3] = u1[3] - Y[3, 1]
end

bvp_fun_train = BVPFunction(F!, ic_bc!;
    bcresid_prototype = zeros(3),
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
@info "Starting training"
#@profview_allocs

@time NN_sol_al = solve(bvp_train,
    RadauIIa5(; optimize = OptimizationAuglag.AugLag(;
        inner = Adam(0.01),
        inner_maxiters = 5, #5000
        γ = 1.0,
        λmin = 0.0, λmax = 0.0,
        μmin = 0.0, μmax = 0.0,
        ρ_init = 10.0,
        ϵ_primal = 1e-3,
        ϵ_dual = 1e6,
    ));
    dt = (tspan[2] - tspan[1])/tlen,
    saveat = tsteps,
    adaptive = true, # before it was false
    verbose = BVPVerbosity(Detailed()),
    optimize_kwargs = (; maxiters = 1, callback = al_callback), #maxiters = 5
)


@time NN_sol_al = solve(bvp_train,
    RadauIIa5(; optimize = Ipopt.Optimizer());
    dt = (tspan[2] - tspan[1])/tlen,
    saveat = tsteps,
    adaptive = true, # before it was false
    verbose = BVPVerbosity(Detailed()),
    optimize_kwargs = (; maxiters = 1, callback = al_callback), #maxiters = 5
)

# Loss curve — every cost_fn evaluation (many per outer AL iter).
plot(losses; xlabel="cost evaluation", ylabel="data loss", yscale=:log10,
     title="AL + Adam on BVP-NN", legend=false)



# Solve using trained parameters. 
prediction_prob = ODEProblem(F!, [2.0, 0.0, 0.0], tspan, NN_sol_al.prob.p)
prediction = solve(prediction_prob, Tsit5(), saveat=tsteps)

# Compare with intial parameters
pretrained_pred_prob = ODEProblem(F!, [2.0, 0.0, 0.0], tspan, p0)
pretrained_pred = solve(pretrained_pred_prob, Tsit5(), saveat=tsteps)



plot(data, labels=["u₁ (data)" "u₂ (data)"])
plot!(prediction, labels=["u₁ (StiffNet)" "u₂ (StiffNet)"])
plot!(pretrained_pred, labels=["u₁ (Pretraining - StiffNet)" "u₂ (Pretraining - StiffNet)"])

########################################################################
########################################################################
######################################################################## 
###### Benchmark 

    


                                                                                                                                                           

@allocated model(u0_guess[1:3], ps, st)  
@allocated u0_guess[1:3]
@allocated @view(u0_guess[1:3])
@allocated view(u0_guess, 1:3)
u0_guess


cost_fn(NN_sol_al, p0)
@benchmark cost_fn($NN_sol_al, $p0)
@allocated cost_fn(NN_sol_al, p0)
#New, 38.84KiB, 1487 Allocs, Allocated, 25376

ic_bc!(res, sol, _p, _t) = begin
    u1 = sol.u[1]
    res[1] = u1[1] - Y[1, 1]
    res[2] = u1[2] - Y[2, 1]
    res[3] = u1[3] - Y[3, 1]
end

res = zeros(3)       
ic_bc!(res, NN_sol_al, p0, nothing)                                                                              
@benchmark ic_bc!($res, $NN_sol_al, $p0, nothing)                                                   
@allocated ic_bc!(res, NN_sol_al, p0, nothing)  


@code_warntype F!(similar(u0_guess, 54), rand(54), p0, 0.0) 

