using BoundaryValueDiffEqFIRK, BoundaryValueDiffEqMIRK
using OrdinaryDiffEqTsit5
using OrdinaryDiffEqRosenbrock
using Plots
using Lux
using Random, ComponentArrays
using BenchmarkTools
using BoundaryValueDiffEqCore: BVPVerbosity
using OptimizationAuglag, OptimizationOptimisers

include("src/layers/StiffLayer.jl")
include("src/derivativeMatching.jl")


rng = Xoshiro()


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
const NN = Chain(
    Dense(3, 4, tanh),
    Dense(4, 4, tanh),
    Dense(4, 3)
)

sigma_affine_sigmoid(ps, j) = 1 / (1 + exp(-(ps[1] * j + ps[2])))
sigma_exp(ps, j) = exp(-(ps[1] * j))

V = Dense(3, 4, tanh)
S = StiffLayer2(sigma_affine_sigmoid; P=4, init=ones(4))
U = Chain(Dense(4, 3, tanh), Dense(3,3))

const model2 = Chain(V, S, U)


model = model2
ps, _st = Lux.setup(rng, model)
ps = Lux.f64(ps)
const st = _st

ps_ca = ComponentVector(ps)
length(ps_ca)
ps_vec = collect(ps_ca)
const ps_axes = getaxes(ps_ca)


N_params = length(ps_ca)
p0 = randn(N_params) #Initialise parameters
u0_guess = Y[:, 1]   # IC comes from the data



#### Collocate
ode_data = Array(data)
node = NeuralODE(model, tspan, Rodas5P(); saveat=tsteps)

# Collocation loss
param_collocate, st_collocate, losses, lrs = train!(
    node, model, ps_ca, st, u0_guess, Array(data), tsteps;
    nepochs = 200,
    learning_rate = 0.04f0,
    min_learning_rate = 1f-4,
    lr_schedule = :cosine,
    loss_mode = :collocation
)

ps_collocate_vec = collect(param_collocate)
##################################################################


function F!(du, u, p, t)
    du[1:3] .= model(@view(u[1:3]), ComponentVector(p, ps_axes), st)[1]
    return nothing
end

# F!(similar(u0_guess), u0_guess, p0, 0.0)                                                                                                                                                                                
# @benchmark F!(du, $u0_guess, $p0, 0.0) setup=(du=similar(u0_guess))
# @allocated F!(similar(u0_guess), u0_guess, p0, 0.0) # MAAANY allocations 
# @allocated ComponentVector(p0, ps_axes)
# @allocated model(@view(u0_guess[1:3]), ComponentVector(p0, ps_axes), st) 


# ============================================================
# Training via MIRK4's `optimize` slot with pure-penalty AugLag (λ ≡ 0, fixed ρ).
# BC is the known IC; interior data points go into the cost.
# MIRK4's optimize path builds an equality-constrained OptimizationProblem
# (cons = collocation + BC residual, lcons = ucons = 0); AugLag with clamped
# multipliers reduces it to `min data_loss + (ρ/2)‖c‖²`.
# ============================================================

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
bvp_train = BVProblem(bvp_fun_train, u0_guess, tspan, ps_collocate_vec; tune_parameters = true)

# Callback, one fire per outer iteration. In pure-penalty mode ρ is fixed and
# λ ≡ 0, so the only meaningful signals are raw_loss and r_primal.
# r_dual scales with ρ × inner Adam stationarity error; informational only.
al_callback = function (state, obj)
    o = state.original
    raw_loss = isempty(losses) ? NaN : losses[end]
    @info "AL" iter=state.iter obj=obj raw_loss=raw_loss r_primal=o.r_primal r_dual=o.r_dual ρ=o.ρ
    return false
end


# γ=1, λmin=λmax=0, μmin=μmax=0 → pure penalty (no multiplier, no ρ growth).
# ρ_init pins the penalty weight directly — cleaner than capping the auto-scaled
# init with ρmax. ϵ_dual huge since r_dual isn't meaningful here.
# Outer iters are mathematically redundant (same subproblem, warm-started Adam),
# so use few outer × many inner.
@info "Starting training"
#@profview_allocs

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
    dt = (tspan[2] - tspan[1])/tlen,
    saveat = tsteps,
    adaptive = true, # before it was false
    verbose = BVPVerbosity(Detailed()),
    optimize_kwargs = (; maxiters = 5, callback = al_callback), #maxiters = 5
)


@time NN_sol_al_radau = solve(bvp_train,
    RadauIIa5(; optimize = OptimizationAuglag.AugLag(;
        inner = Adam(1e-2),
        inner_kwargs = (maxiters = 5,), 
        γ = 2.0,
        λmin = -1e6, λmax = 1e6,
        μmin = 0.0, μmax = 1e6,
        ρ_init = 10.0,
        ϵ_primal = 1e-2,
        ϵ_dual = 1e6,
    ));
    dt = (tspan[2] - tspan[1])/tlen,
    saveat = tsteps,
    adaptive = true, # before it was false
    verbose = BVPVerbosity(Detailed()),
    optimize_kwargs = (; maxiters = 2, callback = al_callback), #maxiters = 5
)

# Loss curve — every cost_fn evaluation (many per outer AL iter).
plot(losses; xlabel="cost evaluation", ylabel="data loss", yscale=:log10,
     title="AL + Adam on BVP-NN", legend=false)



# Solve using trained parameters. 
prediction_prob = ODEProblem(F!, [1.0, 0.0, 0.0], tspan, NN_sol_al.prob.p)
prediction = solve(prediction_prob, Rodas5P()(), saveat=tsteps)

pred_prob_al_mirk = ODEProblem(F2!, [1.0, 0.0, 0.0], tspan, NN_sol_al_mirk.prob.p)
pred_al_mirk = solve(pred_prob_al_mirk, Rodas5P(), saveat=tsteps)

pred_prob_al_radau = ODEProblem(F2!, [1.0, 0.0, 0.0], tspan, NN_sol_al_radau.prob.p)
pred_al_radau = solve(pred_prob_al_radau, Rodas5P(), saveat=tsteps)

# Compare with intial parameters
pretrained_pred_prob = ODEProblem(F!, [1.0, 0.0, 0.0], tspan, p0)
pretrained_pred = solve(pretrained_pred_prob, Rodas5P(), saveat=tsteps)

# Compare with collocation parameters
collocate_pred_prob = ODEProblem(F!, [1.0, 0.0, 0.0], tspan, ps_collocate_vec)
collocate_pred = solve(collocate_pred_prob, Rodas5P(), saveat=tsteps)




plot(data, labels=["u₁ (data)" "u₂ (data)" "u₃ (data)"], title="Data + AL-trained NN")
plot!(collocate_pred, labels=["u₁ (Collocation)" "u₂ (Collocation)" "u₃ (Collocation)"])
plot!(pretrained_pred, labels=["u₁ (Pretraining - StiffNet)" "u₂ (Pretraining - StiffNet)" "u₃ (Pretraining - StiffNet)"])
plot!(prediction, labels=["u₁ (StiffNet)" "u₂ (StiffNet)" "u₃ (StiffNet)"])



p1 = plot(data, labels=["u₁ (data)" "u₂ (data)" "u₃ (data)"], title="Data + Pretrained")
plot!(p1, pretrained_pred, labels=["u₁ (Pretrained)" "u₂ (Pretrained)" "u₃ (Pretrained)"], linestyle=:dash)


p2 = plot(data, labels=["u₁ (data)" "u₂ (data)" "u₃ (data)"], title="AL + MIRK4")
plot!(p2, pred_al_mirk, labels=["u₁" "u₂" "u₃"], linestyle=:dash)

p3 = plot(data, labels=["u₁ (data)" "u₂ (data)" "u₃ (data)"], title="AL + Radau")
plot!(p3, pred_al_radau, labels=["u₁" "u₂" "u₃"], linestyle=:dash)


plot(p1, p2, p3; layout=(1, 3), size=(1200, 400))





param_plot = plot(p0, label="Initial", title="Trained Parameters")
plot!(param_plot, ps_collocate_vec, label="Collocation")
plot!(param_plot, NN_sol_al_mirk.prob.p, label="MIRK4 + AL")
# plot!(param_plot, NN_sol_krylov_mirk.prob.p, label="MIRK4 + Krylov")
plot!(param_plot, NN_sol_al_radau.prob.p, label="Radau + AL")
# plot!(param_plot, NN_sol_krylov_radau.prob.p, label="Radau + Krylov", linestyle=:dash)
param_plot
########################################################################
########################################################################
######################################################################## 
###### Benchmark 

using LinearAlgebra
norm(p0 .- param_collocate)
                                                                                                                                                           

# @allocated model(u0_guess[1:3], ps, st)  
# @allocated u0_guess[1:3]
# @allocated @view(u0_guess[1:3])
# @allocated view(u0_guess, 1:3)
# u0_guess


# cost_fn(NN_sol_al, p0)
# @benchmark cost_fn($NN_sol_al, $p0)
# @allocated cost_fn(NN_sol_al, p0)
# #New, 38.84KiB, 1487 Allocs, Allocated, 25376

# ic_bc!(res, sol, _p, _t) = begin
#     u1 = sol.u[1]
#     res[1] = u1[1] - Y[1, 1]
#     res[2] = u1[2] - Y[2, 1]
#     res[3] = u1[3] - Y[3, 1]
# end

# res = zeros(3)       
# ic_bc!(res, NN_sol_al, p0, nothing)                                                                              
# @benchmark ic_bc!($res, $NN_sol_al, $p0, nothing)                                                   
# @allocated ic_bc!(res, NN_sol_al, p0, nothing)  


# @code_warntype F!(similar(u0_guess, 54), rand(54), p0, 0.0) 

