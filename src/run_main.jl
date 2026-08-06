# julia --project=. stiffnet/run.jl [problem] [parameter]
ENV["GKSwstype"] = "100"
using Lux, Random, ComponentArrays



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

const PROBLEM         = "vanderpol"
const PROFILE         = :fast
const SEED            = 123
const USE_DERIVMATCH  = true
const USE_COLLOCATION = true

function problem_kwargs(problem, param)
    param === nothing && return (;)
    key = config_key(problem)
    key === :vanderpol   && return (; mu = param)
    key === :davisskodje && return (; epsilon = param)
end

function main(problem = PROBLEM, param = nothing)
    cfg     = CONFIGS[config_key(problem)]
    spec    = make_problem(problem; T = Float64, profile = PROFILE,
                           problem_kwargs(problem, param)...)
    Ydata   = generate_training_data(spec)
    d, tlen = length(spec.u0), length(spec.tsteps)

    ymin, ymax = vec(minimum(Ydata, dims = 2)), vec(maximum(Ydata, dims = 2))
    yscale = max.(ymax .- ymin, 1e-12 .* max.(abs.(ymax), abs.(ymin)), floatmin())

    model  = build_stiff_field(d, Ydata, spec.tsteps, (ymax .+ ymin) ./ 2, yscale;
                               width = cfg.hidden, depth = cfg.depth,
                               signed_loss = cfg.signed_loss)
    ps, st = Lux.setup(Xoshiro(SEED), model); ps = Lux.f64(ps)
    init_stiff!(ps, model)
    ctx = (; spec, tsteps = spec.tsteps, tspan = spec.tspan, u0 = spec.u0,
             d, tlen, Ydata, yscale, model, st, ps_axes = getaxes(ComponentVector(ps)))

    println("$(spec.name)  d=$d tlen=$tlen profile=$PROFILE seed=$SEED  " *
            "solver=$(typeof(spec.solver).name.name)  N=$(length(ComponentVector(ps)))")
    flush(stdout)

    θ = collect(ComponentVector(ps))
    if USE_DERIVMATCH
        t = @elapsed ((θ, fitloss) = derivative_matching(ctx, θ, fd_derivatives(Ydata, spec.tsteps),
                                                         cfg.dm_iters))
        println("derivative matching [$(round(Int, t))s]  fitloss=$(round(fitloss; sigdigits=3))")
        report(ctx, "derivative matching", θ)
    end
    if USE_COLLOCATION
        t = @elapsed (θ = train_collocation(ctx, θ, cfg; score = p -> metrics(ctx, p).nrmse))
        println("collocation [$(round(Int, t))s]")
        report(ctx, "final", θ)
    end

    plot_fit(ctx, θ)
    plot_spectral_fit(ctx, θ)
    plot_spectral_fit(ctx, θ)
    return (; ctx, cfg, θ)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? PROBLEM : ARGS[1],
         length(ARGS) > 1 ? parse(Float64, ARGS[2]) : nothing)
end
