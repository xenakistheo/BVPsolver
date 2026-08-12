# julia --project=. src/run_main.jl [--problem NAME] [--param VALUE] [--profile fast|default]
#                                    [--seed N] [--pretraining none|derivmatch]
#                                    [--training none|shooting|shapovalova|collocation]
ENV["GKSwstype"] = "100"
using Lux, Random, ComponentArrays
using ArgParse, JLD2, TOML, Dates
using StiffNN

const CONFIGS = (
    pollu     = (hidden = 32, depth = 3, signed_loss = false, dm_iters = 30_000,
                 amr_rounds = 1, lm_iters = 150),
    rober     = (hidden = 12, depth = 2, signed_loss = false, dm_iters = 10_000,
                 amr_rounds = 1, lm_iters = 150),
    vanderpol = (hidden = 12, depth = 2, signed_loss = true,  dm_iters = 10_000,
                 amr_rounds = 1, lm_iters = 150),
    hires     = (hidden = 16, depth = 2, signed_loss = false, dm_iters = 10_000,
                 amr_rounds = 1, lm_iters = 150, linsolve = :qr),
    orego     = (hidden = 16, depth = 2, signed_loss = false, dm_iters = 10_000,
                 amr_rounds = 1, lm_iters = 150),
    davisskodje = (hidden = 12, depth = 2, signed_loss = false, dm_iters = 10_000,
                 amr_rounds = 3, lm_iters = 150),
    brusselator = (hidden = 32, depth = 3, signed_loss = false, dm_iters = 30_000,
                 amr_rounds = 1, lm_iters = 150, time_dependent = true),
)

config_key(name) = Symbol(replace(lowercase(name), "-" => ""))

gelu_arch(problem) = config_key(problem) === :pollu ? (hidden = 10, depth = 3) :
                                                      (hidden = 5,  depth = 6)

function problem_kwargs(problem, param)
    param === nothing && return (;)
    key = config_key(problem)
    key === :vanderpol   && return (; mu = param)
    key === :davisskodje && return (; epsilon = param)
    key === :brusselator && return (; n = round(Int, param))
    error("$problem takes no scalar parameter " *
          "(vanderpol=mu, davis-skodje=epsilon, brusselator=n)")
end

# ── CLI arguments ─────────────────────────────────────────────────────────
function parse_cli()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--problem"
            help = "Problem (rober, vanderpol, pollu, hires, orego, davis-skodje, brusselator)"
            arg_type = String
            default = "vanderpol"
        "--param"
            help = "Problem parameter: mu (vanderpol), epsilon (davis-skodje), n grid (brusselator)"
            arg_type = Float64
            default = nothing
        "--model"
            help = "Field architecture (stiff, mlp, GELU-scaled)"
            arg_type = String
            default = "stiff"
            range_tester = x -> x in ("stiff", "mlp", "GELU-scaled")
        "--profile"
            help = "Problem profile"
            arg_type = String
            default = "fast"
            range_tester = x -> x in ("fast", "default")
        "--seed"
            help = "RNG seed for model initialization"
            arg_type = Int
            default = 123
        "--computer"
            help = "Machine this run was launched from (for run_info.toml bookkeeping)"
            arg_type = String
            default = "local"
            range_tester = x -> x in ("local", "orcd", "supercloud")
        "--out-file"
            help = "Path to this job's SLURM stdout log (for run_info.toml bookkeeping)"
            arg_type = String
            default = nothing
        "--err-file"
            help = "Path to this job's SLURM stderr log (for run_info.toml bookkeeping)"
            arg_type = String
            default = nothing
        "--init"
            help = "Weight initialization (default = Lux kaiming+gain, glorot = Glorot uniform + zero bias)"
            arg_type = String
            default = "default"
            range_tester = x -> x in ("default", "glorot")
        "--sigma"
            help = "Gaussian noise std as a fraction of each species' range (0.01 = 1%)"
            arg_type = Float64
            default = 0.0
        "--pretraining"
            help = "Pretraining stage run before --training"
            arg_type = String
            default = "derivmatch"
            range_tester = x -> x in ("none", "derivmatch")
        "--training"
            help = "Main training method"
            arg_type = String
            default = "collocation"
            range_tester = x -> x in ("none", "shooting", "shapovalova", "collocation")
        "--shoot-iters"
            help = "Adam iterations for single shooting (--training shooting)"
            arg_type = Int
            default = 1000
        "--ncol"
            help = "Chebyshev collocation nodes for --training shapovalova (0 = one per observation)"
            arg_type = Int
            default = 0
        "--shap-hessian"
            help = "Ipopt hessian_approximation for --training shapovalova"
            arg_type = String
            default = "exact"
            range_tester = x -> x in ("exact", "limited-memory")
    end
    return parse_args(s)
end

toml_safe(value) = value === nothing ? "none" : value
toml_safe(value::Symbol) = String(value)

function main(args = parse_cli())
    problem         = args["problem"]
    param           = args["param"]
    arch            = args["model"]
    profile         = Symbol(args["profile"])
    seed            = args["seed"]
    sigma           = args["sigma"]
    init            = Symbol(args["init"])
    shoot_iters     = args["shoot-iters"]
    ncol            = args["ncol"]
    shap_hessian    = args["shap-hessian"]
    use_derivmatch  = args["pretraining"] == "derivmatch"
    use_shooting    = args["training"] == "shooting"
    use_shap        = args["training"] == "shapovalova"
    use_collocation = args["training"] == "collocation"

    cfg     = CONFIGS[config_key(problem)]
    spec    = make_problem(problem; T = Float64, profile = profile,
                           problem_kwargs(problem, param)...)
    Ydata   = generate_training_data(spec)
    d, tlen = length(spec.u0), length(spec.tsteps)
    if sigma > 0
        lo, hi = vec(minimum(Ydata, dims = 2)), vec(maximum(Ydata, dims = 2))
        nscale = max.(hi .- lo, 1e-12 .* max.(abs.(hi), abs.(lo)), floatmin())
        Ydata .+= (sigma .* nscale) .* randn(Xoshiro(hash((seed, :noise))), size(Ydata))
    end

    ymin, ymax = vec(minimum(Ydata, dims = 2)), vec(maximum(Ydata, dims = 2))
    yscale = max.(ymax .- ymin, 1e-12 .* max.(abs.(ymax), abs.(ymin)), floatmin())

    ymid = (ymax .+ ymin) ./ 2
    td   = get(cfg, :time_dependent, false)
    model = if arch == "stiff"
        build_stiff_field(d, Ydata, spec.tsteps, ymid, yscale;
                          width = cfg.hidden, depth = cfg.depth,
                          signed_loss = cfg.signed_loss, time_dependent = td, init = init)
    elseif arch == "GELU-scaled"
        a = gelu_arch(problem)
        build_mlp_field(d, Ydata, spec.tsteps, yscale;
                        width = a.hidden, depth = a.depth,
                        scaling = :eq, activation = gelu, time_dependent = td, init = init)
    else
        build_mlp_field(d, Ydata, spec.tsteps, yscale;
                        width = cfg.hidden, depth = cfg.depth, time_dependent = td, init = init)
    end
    ps, st = Lux.setup(Xoshiro(seed), model); ps = Lux.f64(ps)
    init_stiff!(ps, model)
    ctx = (; spec, tsteps = spec.tsteps, tspan = spec.tspan, u0 = spec.u0,
             d, tlen, Ydata, yscale, model, st, ps_axes = getaxes(ComponentVector(ps)))

    n_params = length(ComponentVector(ps))
    println("$(spec.name)  model=$arch d=$d tlen=$tlen profile=$profile seed=$seed" *
            (sigma > 0 ? " sigma=$sigma" : "") *
            (init === :default ? "" : " init=$init") *
            "  solver=$(typeof(spec.solver).name.name)  N=$n_params")
    flush(stdout)

    θ = collect(ComponentVector(ps))
    derivmatch_time  = 0.0
    collocation_time = 0.0
    shooting_time    = 0.0
    shapovalova_time = 0.0
    if use_derivmatch
        derivmatch_time = @elapsed ((θ, fitloss) = derivative_matching(ctx, θ, fd_derivatives(Ydata, spec.tsteps),
                                                         cfg.dm_iters))
        println("derivative matching [$(round(Int, derivmatch_time))s]  fitloss=$(round(fitloss; sigdigits=3))")
        report(ctx, "derivative matching", θ)
    end
    if use_shap
        shapovalova_time = @elapsed ((θ, shaploss) =
            shapovalova(ctx, θ, ncol > 0 ? ncol : tlen; hessian_approximation = shap_hessian))
        println("shapovalova [$(round(Int, shapovalova_time))s]  loss=$(round(shaploss; sigdigits=3))")
        report(ctx, "final", θ)
    end
    if use_shooting
        shooting_time = @elapsed ((θ, shootloss) = shooting(ctx, θ, shoot_iters))
        println("shooting [$(round(Int, shooting_time))s]  loss=$(round(shootloss; sigdigits=3))")
        report(ctx, "final", θ)
    end
    if use_collocation
        collocation_time = @elapsed (θ = train_collocation(ctx, θ, cfg; score = p -> metrics(ctx, p).nrmse))
        println("collocation [$(round(Int, collocation_time))s]")
        report(ctx, "final", θ)
    end

    # ── Extrapolation / error metrics (see metrics.md) ──────────────────────
    train_err = error_metrics(ctx, θ)
    ext_ctx, train_mask = extrapolation_ctx(ctx)
    test_ctx  = merge(ext_ctx, (; tsteps = ext_ctx.tsteps[.!train_mask],
                                   Ydata  = ext_ctx.Ydata[:, .!train_mask],
                                   tlen   = count(.!train_mask)))
    test_err  = error_metrics(test_ctx, θ)
    println("extrapolation window: T=$(spec.tspan[2]) -> T_test=$(round(ext_ctx.tspan[2]; sigdigits=6))" *
            "  n_test=$(count(.!train_mask))")
    println("  [train]  E_traj=$(round(train_err.traj; sigdigits=4))  " *
            "E_VF=$(round(train_err.vf; sigdigits=4))  E_spec=$(round(train_err.spec; sigdigits=4))")
    println("  [test ]  E_traj=$(round(test_err.traj; sigdigits=4))  " *
            "E_VF=$(round(test_err.vf; sigdigits=4))  E_spec=$(round(test_err.spec; sigdigits=4))")
    flush(stdout)

    # ── Save run artifacts ──────────────────────────────────────────────────
    # Every run gets its own folder tagged with a random suffix, holding the
    # fit plots, the trained state, and a record of the parameters used.
    run_dir = joinpath(@__DIR__, "..", "data", "runs", "run_$(randstring(8))")
    mkpath(run_dir)

    plot_fit(ctx, θ; dir = run_dir)
    plot_fit(ext_ctx, θ; dir = run_dir, train_mask)
    try
        plot_spectral_fit(ext_ctx, θ; dir = run_dir, train_mask)
    catch e
        println("  spectral fit plot FAILED: $e")
    end
    try
        plot_vf_fit(ext_ctx, θ; dir = run_dir, train_mask)
    catch e
        println("  velocity field fit plot FAILED: $e")
    end

    jldsave(joinpath(run_dir, "state.jld2"); ctx = ctx, cfg = cfg, θ = θ)

    run_info = Dict(k => toml_safe(v) for (k, v) in args)
    run_info["cfg"] = Dict(String(k) => toml_safe(v) for (k, v) in pairs(cfg))
    run_info["derivmatch_time"]  = derivmatch_time
    run_info["collocation_time"] = collocation_time
    run_info["shooting_time"]    = shooting_time
    run_info["shapovalova_time"] = shapovalova_time
    run_info["training_time"]    = derivmatch_time + collocation_time +
                                   shooting_time + shapovalova_time
    run_info["n_params"]         = n_params
    run_info["timestamp"]        = string(now())

    run_info["E_trajectory_species_train"] = train_err.traj_species
    run_info["E_trajectory_train"]         = train_err.traj
    run_info["E_trajectory_species_test"]  = test_err.traj_species
    run_info["E_trajectory_test"]          = test_err.traj

    run_info["E_VF_species_train"] = train_err.vf_species
    run_info["E_VF_train"]         = train_err.vf
    run_info["E_VF_species_test"]  = test_err.vf_species
    run_info["E_VF_test"]          = test_err.vf

    run_info["E_spec_species_train"] = train_err.spec_species
    run_info["E_spec_train"]         = train_err.spec
    run_info["E_spec_species_test"]  = test_err.spec_species
    run_info["E_spec_test"]          = test_err.spec
    open(joinpath(run_dir, "run_info.toml"), "w") do io
        TOML.print(io, run_info; sorted = true)
    end
    println("saved run → $run_dir")

    return (; ctx, cfg, θ)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
