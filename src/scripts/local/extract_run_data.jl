# Extracts trajectory + spectral fit data (train/test scatter, dense rollout,
# eigenvalue curves) from a saved run's state.jld2 into plain CSV files, so the
# figures for the paper can be built with a Python plotting library instead of
# Plots.jl.
#
# Usage: julia --project=. src/scripts/local/extract_run_data.jl <run_dir> <out_dir>
ENV["GKSwstype"] = "100"
using JLD2, StiffNN, DelimitedFiles

function main()
    length(ARGS) == 2 || error("usage: extract_run_data.jl <run_dir> <out_dir>")
    run_dir, out_dir = ARGS
    mkpath(out_dir)

    state = JLD2.load(joinpath(run_dir, "state.jld2"))
    ctx, θ = state["ctx"], state["θ"]

    ext_ctx, train_mask = extrapolation_ctx(ctx)
    d = ext_ctx.d

    # ── Dense trajectory rollout (smooth neural-ODE curve) ──────────────────
    logt  = is_log_spaced(ext_ctx.tsteps)
    tgrid = logt ? exp.(range(log(ext_ctx.tsteps[1]), log(ext_ctx.tsteps[end]); length = 400)) :
                   collect(range(ext_ctx.tspan[1], ext_ctx.tspan[2]; length = 400))
    sol  = rollout(ext_ctx, θ; dense = true)
    ts   = sort(unique(vcat(tgrid, sol.t)))
    pred = reduce(hcat, (sol(t) for t in ts))

    open(joinpath(out_dir, "traj_dense.csv"), "w") do io
        println(io, join(vcat("t", ["y$c" for c in 1:d]), ","))
        for i in eachindex(ts)
            println(io, join(vcat(ts[i], pred[:, i]), ","))
        end
    end

    # ── Scatter data (train + test), for both trajectory and spectral plots ─
    open(joinpath(out_dir, "traj_data.csv"), "w") do io
        println(io, join(vcat("t", ["y$c" for c in 1:d], "is_train"), ","))
        for i in eachindex(ext_ctx.tsteps)
            println(io, join(vcat(ext_ctx.tsteps[i], ext_ctx.Ydata[:, i], train_mask[i] ? 1 : 0), ","))
        end
    end

    eig_true, eig_learned = eigenvalues(ext_ctx, θ; absolute = true)
    open(joinpath(out_dir, "spectral.csv"), "w") do io
        cols = vcat("t", ["lambda$(c)_true" for c in 1:d], ["lambda$(c)_learned" for c in 1:d], "is_train")
        println(io, join(cols, ","))
        for i in eachindex(ext_ctx.tsteps)
            println(io, join(vcat(ext_ctx.tsteps[i], eig_true[:, i], eig_learned[:, i],
                                   train_mask[i] ? 1 : 0), ","))
        end
    end

    open(joinpath(out_dir, "meta.csv"), "w") do io
        println(io, "key,value")
        println(io, "name,$(ctx.spec.name)")
        println(io, "d,$d")
        println(io, "T_train,$(ctx.spec.tspan[2])")
        println(io, "T_test,$(ext_ctx.tspan[2])")
        println(io, "logt,$(logt ? 1 : 0)")
    end

    println("wrote $out_dir")
end

main()
