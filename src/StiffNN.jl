module StiffNN

# Load dependencies
using OrdinaryDiffEq
using OrdinaryDiffEqTsit5: Tsit5
using OrdinaryDiffEqRosenbrock: Rodas5, Rodas5P
using OrdinaryDiffEqSDIRK: Kvaerno5

using Statistics
using SciMLBase: successful_retcode
import SciMLLogging as SL
using ForwardDiff, LinearAlgebra

using Lux

using NonlinearSolve, ComponentArrays, LinearSolve
using SparseConnectivityTracer, SparseMatrixColorings

using Optimization, OptimizationOptimisers
import JuMP, Ipopt
using Zygote  

using Plots

# Include submodules
include("problems.jl")
include("models/init.jl")
include("models/stiff_field.jl")
include("models/mlp_field.jl")
include("training/derivative_matching.jl")
include("training/shooting.jl")
include("training/shapovalova.jl")
include("training/collocation.jl")
include("utils/eval.jl")


# problems.jl
export generate_training_data, make_problem

# init.jl
export glorot_uniform, zeros_init, init_kwargs

# stiff_field.jl
export build_stiff_field, init_stiff!

# mlp_field.jl
export build_mlp_field

# collocation.jl
export initial_nsub, build_mesh, interp_onto, R!, solve_collocation
export interval_errors, obs_residual, train_collocation

# derivative_matching.jl
export fd_derivatives, derivative_matching

# shooting.jl
export shooting

# shapovalova.jl
export shapovalova, cheb_diffmatrix, interp_rows

# eval.jl
export rollout, metrics, report, plot_fit, eigenvalues, plot_spectral_fit

# Data
export generate_diffusion_data
export load_burgers_batch, load_rd_batch, load_ns_batch, load_heat_batch

# Constraints
export heat_constraints!, rd_constraints!, rd_constraints_2!, burgers_constraints_BC_Mass!, burgers_constraints_IC_Mass_Flux!, ns_constraints!, ns_enstrophy_constraints!
export burgers_constraints_IC!, burgers_constraints_IC_Mass!

end # module StiffNN
