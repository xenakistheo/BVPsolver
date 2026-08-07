# julia --project=. src/run_main.jl [--problem NAME] [--param VALUE] [--profile fast|default]
#                                    [--seed N] [--no-derivmatch] [--no-collocation]
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
)

config_key(name) = Symbol(replace(lowercase(name), "-" => ""))

function problem_kwargs(problem, param)
    param === nothing && return (;)
    key = config_key(problem)
    key === :vanderpol   && return (; mu = param)
    key === :davisskodje && return (; epsilon = param)
end

# ── CLI arguments ─────────────────────────────────────────────────────────
function parse_cli()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--problem"
            help = "Problem to solve (spiral, rober, vanderpol, hires, orego, davisskodje)"
            arg_type = String
            default = "vanderpol"
        "--param"
            help = "Override the problem's stiffness parameter (mu for vanderpol, epsilon for davisskodje)"
            arg_type = Float64
            default = nothing
        "--profile"
            help = "Problem profile"
            arg_type = String
            default = "fast"
            range_tester = x -> x in ("fast", "default")
        "--seed"
            help = "RNG seed for model initialization"
            arg_type = Int
            default = 123
        "--no-derivmatch"
            help = "Skip the derivative-matching pretraining stage"
            action = :store_false
            dest_name = "use_derivmatch"
        "--no-collocation"
            help = "Skip the collocation training stage"
            action = :store_false
            dest_name = "use_collocation"
    end
    return parse_args(s)
end

toml_safe(value) = value === nothing ? "none" : value
toml_safe(value::Symbol) = String(value)

function main(args = parse_cli())
    problem         = args["problem"]
    param           = args["param"]
    profile         = Symbol(args["profile"])
    seed            = args["seed"]
    use_derivmatch  = args["use_derivmatch"]
    use_collocation = args["use_collocation"]

    cfg     = CONFIGS[config_key(problem)]
    spec    = make_problem(problem; T = Float64, profile = profile,
                           problem_kwargs(problem, param)...)
    Ydata   = generate_training_data(spec)
    d, tlen = length(spec.u0), length(spec.tsteps)

    ymin, ymax = vec(minimum(Ydata, dims = 2)), vec(maximum(Ydata, dims = 2))
    yscale = max.(ymax .- ymin, 1e-12 .* max.(abs.(ymax), abs.(ymin)), floatmin())

    model  = build_stiff_field(d, Ydata, spec.tsteps, (ymax .+ ymin) ./ 2, yscale;
                               width = cfg.hidden, depth = cfg.depth,
                               signed_loss = cfg.signed_loss)
    ps, st = Lux.setup(Xoshiro(seed), model); ps = Lux.f64(ps)
    init_stiff!(ps, model)
    ctx = (; spec, tsteps = spec.tsteps, tspan = spec.tspan, u0 = spec.u0,
             d, tlen, Ydata, yscale, model, st, ps_axes = getaxes(ComponentVector(ps)))

    n_params = length(ComponentVector(ps))
    println("$(spec.name)  d=$d tlen=$tlen profile=$profile seed=$seed  " *
            "solver=$(typeof(spec.solver).name.name)  N=$n_params")
    flush(stdout)

    θ = collect(ComponentVector(ps))
    derivmatch_time  = 0.0
    collocation_time = 0.0
    if use_derivmatch
        derivmatch_time = @elapsed ((θ, fitloss) = derivative_matching(ctx, θ, fd_derivatives(Ydata, spec.tsteps),
                                                         cfg.dm_iters))
        println("derivative matching [$(round(Int, derivmatch_time))s]  fitloss=$(round(fitloss; sigdigits=3))")
        report(ctx, "derivative matching", θ)
    end
    if use_collocation
        collocation_time = @elapsed (θ = train_collocation(ctx, θ, cfg; score = p -> metrics(ctx, p).nrmse))
        println("collocation [$(round(Int, collocation_time))s]")
        report(ctx, "final", θ)
    end

    # ── Save run artifacts ──────────────────────────────────────────────────
    # Every run gets its own folder tagged with a random suffix, holding the
    # fit plots, the trained state, and a record of the parameters used.
    run_dir = joinpath(@__DIR__, "..", "data", "runs", "run_$(randstring(8))")
    mkpath(run_dir)

    plot_fit(ctx, θ; dir = run_dir)
    plot_spectral_fit(ctx, θ; dir = run_dir)

    jldsave(joinpath(run_dir, "state.jld2"); ctx = ctx, cfg = cfg, θ = θ)

    run_info = Dict(k => toml_safe(v) for (k, v) in args)
    run_info["cfg"] = Dict(String(k) => toml_safe(v) for (k, v) in pairs(cfg))
    run_info["derivmatch_time"]  = derivmatch_time
    run_info["collocation_time"] = collocation_time
    run_info["training_time"]    = derivmatch_time + collocation_time
    run_info["n_params"]         = n_params
    run_info["timestamp"]        = string(now())
    open(joinpath(run_dir, "run_info.toml"), "w") do io
        TOML.print(io, run_info; sorted = true)
    end
    println("saved run → $run_dir")

    return (; ctx, cfg, θ)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
