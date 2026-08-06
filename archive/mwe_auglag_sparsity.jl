using BoundaryValueDiffEqMIRK
using OrdinaryDiffEqTsit5
using OptimizationAuglag, OptimizationOptimisers

pathof(SciMLBase)

# Simple linear ODE: du/dt = p[1]*u, true solution u(t) = exp(-t) for p=-1
function simple_ode!(du, u, p, t)
    du[1] = p[1] * u[1]
end

tspan = (0.0, 1.0)
tsteps = range(tspan[1], tspan[2]; length=10) |> collect
Y = exp.(-tsteps)

cost_fn = (sol, p) -> sum((sol(t)[1] - Y[i])^2 for (i, t) in enumerate(tsteps))

bc!(res, sol, p, t) = (res[1] = sol(tspan[1])[1] - 1.0)

bvp_fun = BVPFunction(simple_ode!, bc!;
    bcresid_prototype = zeros(1),
    cost = cost_fn,
)

bvp = BVProblem(bvp_fun, [1.0], tspan, [0.5]; tune_parameters = true)

sol = solve(bvp,
    MIRK4(; optimize = OptimizationAuglag.AugLag(;
        inner = Adam(0.01),
        inner_kwargs = (maxiters = 3,),
        γ = 1.0,
        λmin = 0.0, λmax = 0.0,
        μmin = 0.0, μmax = 0.0,
        ρ_init = 10.0,
        ϵ_primal = 1e-3,
        ϵ_dual = 1e6,
    ));
    dt = 0.1,
    adaptive = false,
    optimize_kwargs = (; maxiters = 1),
)

#https://github.com/SebastianM-C/SciMLBase.jl#smc/allow-optimize-kwargs-v2.153

"""
ERROR: DimensionMismatch: `A` and `bg.S2` must have the same sparsity pattern.
Stacktrace:
  [1] check_compatible_pattern
    @ ~/.julia/packages/SparseMatrixColorings/3N42A/src/matrices.jl:96 [inlined]
  [2] decompress!(A::SubArray{…}, B::Matrix{…}, result::SparseMatrixColorings.ColumnColoringResult{…})
    @ SparseMatrixColorings ~/.julia/packages/SparseMatrixColorings/3N42A/src/decompression.jl:344
  [3] _sparse_jacobian_aux!(f_or_f!y::Tuple{…}, jac::SubArray{…}, prep::DifferentiationInterfaceSparseMatrixColoringsExt.SMCPushforwardSparseJacobianPrep{…}, backend::AutoSparse{…}, x::Vector{…}, contexts::DifferentiationInterface.Constant{…})
    @ DifferentiationInterfaceSparseMatrixColoringsExt ~/.julia/packages/DifferentiationInterface/IS0Dg/ext/DifferentiationInterfaceSparseMatrixColoringsExt/jacobian.jl:326
  [4] (::OptimizationAuglag.var"#generate_auglag##4#generate_auglag##5"{…})(G::Vector{…}, θ::Vector{…}, p::Vector{…})
    @ OptimizationAuglag ~/.julia/packages/OptimizationAuglag/l2Xp7/src/auglag_function.jl:145
  [5] (::OptimizationBase.var"#fg#378"{Vector{…}, OptimizationFunction{…}})(G::Vector{Float64}, x::Vector{Float64})
    @ OptimizationBase ~/.julia/packages/OptimizationBase/zfZsZ/src/function.jl:151
  [6] __solve(cache::OptimizationCache{…})
    @ OptimizationOptimisers ~/.julia/packages/OptimizationOptimisers/duyYv/src/OptimizationOptimisers.jl:88
  [7] solve!(cache::OptimizationCache{…})
    @ OptimizationBase ~/.julia/packages/OptimizationBase/zfZsZ/src/solve.jl:236
  [8] __solve(cache::OptimizationCache{…})
    @ OptimizationAuglag ~/.julia/packages/OptimizationAuglag/l2Xp7/src/OptimizationAuglag.jl:148
  [9] solve!(cache::OptimizationCache{…})
    @ OptimizationBase ~/.julia/packages/OptimizationBase/zfZsZ/src/solve.jl:236
 [10] solve_call(::OptimizationProblem{…}, ::AugLag{…}; merge_callbacks::Bool, kwargshandle::Nothing, kwargs::@Kwargs{…})
    @ OptimizationBase ~/.julia/packages/OptimizationBase/zfZsZ/src/solve.jl:282
 [11] solve_call
    @ ~/.julia/packages/OptimizationBase/zfZsZ/src/solve.jl:269 [inlined]
 [12] solve_up(prob::OptimizationProblem{…}, sensealg::Nothing, u0::Vector{…}, p::Vector{…}, args::AugLag{…}; originator::SciMLBase.ChainRulesOriginator, kwargs::@Kwargs{…})
    @ OptimizationBase ~/.julia/packages/OptimizationBase/zfZsZ/src/solve.jl:265
 [13] solve_up
    @ ~/.julia/packages/OptimizationBase/zfZsZ/src/solve.jl:256 [inlined]
 [14] solve(prob::OptimizationProblem{…}, args::AugLag{…}; sensealg::Nothing, u0::Nothing, p::Nothing, wrap::Val{…}, kwargs::@Kwargs{…})
    @ OptimizationBase ~/.julia/packages/OptimizationBase/zfZsZ/src/solve.jl:104
 [15] __internal_solve
    @ ~/.julia/packages/BoundaryValueDiffEqCore/rkFhw/src/default_internal_solve.jl:99 [inlined]
 [16] __perform_mirk_iteration(cache::BoundaryValueDiffEqMIRK.MIRKCache{…}, abstol::Float64, adaptive::Bool, controller::DefectControl{…})
    @ BoundaryValueDiffEqMIRK ~/.julia/packages/BoundaryValueDiffEqMIRK/NqKEC/src/mirk.jl:317
 [17] solve!(cache::BoundaryValueDiffEqMIRK.MIRKCache{…})
    @ BoundaryValueDiffEqMIRK ~/.julia/packages/BoundaryValueDiffEqMIRK/NqKEC/src/mirk.jl:275
 [18] #__solve#93
    @ ~/.julia/packages/BoundaryValueDiffEqCore/rkFhw/src/BoundaryValueDiffEqCore.jl:48 [inlined]
 [19] __solve
    @ ~/.julia/packages/BoundaryValueDiffEqCore/rkFhw/src/BoundaryValueDiffEqCore.jl:43 [inlined]
 [20] #solve_call#22
    @ ~/.julia/packages/DiffEqBase/bcYrc/src/solve.jl:172 [inlined]
 [21] solve_call
    @ ~/.julia/packages/DiffEqBase/bcYrc/src/solve.jl:137 [inlined]
 [22] #solve_up#29
    @ ~/.julia/packages/DiffEqBase/bcYrc/src/solve.jl:630 [inlined]
 [23] solve_up
    @ ~/.julia/packages/DiffEqBase/bcYrc/src/solve.jl:603 [inlined]
 [24] solve(prob::BVProblem{…}, args::MIRK4{…}; sensealg::Nothing, u0::Nothing, p::Nothing, wrap::Val{…}, kwargs::@Kwargs{…})
    @ DiffEqBase ~/.julia/packages/DiffEqBase/bcYrc/src/solve.jl:587
 [25] top-level scope
    @ ~/Downloads/GoCart/MIT/JuliaLab/BVPsolver/mwe_auglag_sparsity.jl:25
Some type information was truncated. Use `show(err)` to see complete types.

"""