# julia --project=. src/extrapolate/extrapolate.jl --run-dir data/runs/run_XXXXXXXX
#                                                    [--t-test VALUE] [--n-extra N] [--out-dir DIR]
#
# Loads a trained model from a run saved by run_main.jl (trained on t ∈ [0, T]) and
# tests it on the extrapolation window t ∈ [T, T_test], where T is the training
# horizon (spec.tspan[2]) baked into the saved run. Reports nrmse/l2 separately for
# the training window and the extrapolation window, and saves a fit plot marking the
# boundary between them.
ENV["GKSwstype"] = "100"
using ArgParse, JLD2, TOML, Dates
using SciMLBase: successful_retcode
using Statistics
using StiffNN

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--run-dir"
            help = "Path to a saved run directory (as produced by run_main.jl), containing state.jld2"
            arg_type = String
            required = true
        "--t-test"
            help = "End time of the extrapolation window (default: 2x the training horizon T " *
                   "on a linear axis; on a log axis, extend by the same number of decades as " *
                   "training, i.e. T^2/t_min)"
            arg_type = Float64
            default = nothing
        "--n-extra"
            help = "Number of evaluation points added in (T, T_test] (default: 2x the number " *
                   "of training points)"
            arg_type = Int
            default = nothing
        "--out-dir"
            help = "Where to save the plot and metrics (default: <run-dir>/extrapolate)"
            arg_type = String
            default = nothing
    end
    return parse_args(s)
end

function region_metrics(P, Ydata, yscale, mask)
    any(mask) || return (nrmse = NaN, l2 = NaN)
    Pm, Ym = @view(P[:, mask]), @view(Ydata[:, mask])
    return (nrmse = sqrt(mean(abs2, (Pm .- Ym) ./ yscale)),
            l2    = sqrt(sum(abs2, Pm .- Ym)))
end

function main(args = parse_cli())
    run_dir = args["run-dir"]
    state   = load(joinpath(run_dir, "state.jld2"))
    ctx0, θ = state["ctx"], state["θ"]
    spec    = ctx0.spec

    T       = spec.tspan[2]
    n_extra = args["n-extra"]

    ctx, train = extrapolation_ctx(ctx0; t_test = args["t-test"], n_extra = n_extra)
    T_test = ctx.tspan[2]

    sol = rollout(ctx, θ)
    if !successful_retcode(sol)
        error("rollout FAILED: $(sol.retcode) at t=$(sol.t[end]) of $(T_test)")
    end
    P = Array(sol)

    m_train = region_metrics(P, ctx.Ydata, ctx0.yscale, train)
    m_test  = region_metrics(P, ctx.Ydata, ctx0.yscale, .!train)
    println("$(spec.name)  T=$T  T_test=$T_test  n_train=$(count(train))  n_test=$(count(.!train))")
    println("  [train, t<=T]        nrmse=$(round(m_train.nrmse; sigdigits=4))  l2=$(round(m_train.l2; sigdigits=4))")
    println("  [test,  T<t<=T_test] nrmse=$(round(m_test.nrmse; sigdigits=4))  l2=$(round(m_test.l2; sigdigits=4))")
    flush(stdout)

    out_dir = args["out-dir"] === nothing ? joinpath(run_dir, "extrapolate") : args["out-dir"]
    mkpath(out_dir)
    plot_fit(ctx, θ; dir = out_dir, train_mask = train)

    info = Dict(
        "run_dir"     => run_dir,
        "T"           => T,
        "t_test"      => T_test,
        "n_extra"     => count(.!train),
        "train_nrmse" => m_train.nrmse,
        "train_l2"    => m_train.l2,
        "test_nrmse"  => m_test.nrmse,
        "test_l2"     => m_test.l2,
        "timestamp"   => string(now()),
    )
    open(joinpath(out_dir, "extrapolate_info.toml"), "w") do io
        TOML.print(io, info; sorted = true)
    end
    println("saved $(joinpath(out_dir, "extrapolate_info.toml"))")

    return (; ctx, m_train, m_test)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


# julia --project=. src/extrapolate/extrapolate.jl --run-dir data/runs/run_AdvfCqR0    vanderpol 
# julia --project=. src/extrapolate/extrapolate.jl --run-dir data/runs/run_0H5EbMNc    rober   
# julia --project=. src/extrapolate/extrapolate.jl --run-dir data/runs/run_AdvfCqR0         
# julia --project=. src/extrapolate/extrapolate.jl --run-dir data/runs/run_AdvfCqR0         