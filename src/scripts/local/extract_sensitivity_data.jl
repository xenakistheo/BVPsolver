# Reads sensitivity_by_stage (written by --track-sensitivity in run_main.jl) out of
# every run's state.jld2 under a runs directory and writes one long-format CSV:
# run_dir,stage,point_index,stiffness,sensitivity_norm,trajectory_sensitivity_norm
#
# trajectory_sensitivity_norm is written as "NaN" for runs from before that field
# existed (pointwise sensitivity_norm/stiffness only), so old and new runs can be
# combined in one CSV -- pandas reads "NaN" as a float NaN, not a parse error.
#
# Only the sensitivity_by_stage key is loaded from each state.jld2 (not ctx/θ/cfg),
# so this needs only JLD2 -- no StiffNN/Lux model reconstruction -- and stays fast
# across thousands of runs.
#
# Usage: julia --project=. src/scripts/local/extract_sensitivity_data.jl <runs_dir> <out_csv>
using JLD2

function main()
    length(ARGS) == 2 || error("usage: extract_sensitivity_data.jl <runs_dir> <out_csv>")
    runs_dir, out_csv = ARGS

    n_runs = 0
    open(out_csv, "w") do io
        println(io, "run_dir,stage,point_index,stiffness,sensitivity_norm,trajectory_sensitivity_norm")
        for entry in sort(readdir(runs_dir))
            state_path = joinpath(runs_dir, entry, "state.jld2")
            isfile(state_path) || continue

            # Most run dirs predate --track-sensitivity, so state.jld2 simply lacks
            # the key -- expected and skipped. FileIO logs a "Fatal error" line to
            # stderr on that KeyError regardless of the catch below, so redirect
            # stderr around the load rather than let it spam thousands of lines.
            data = try
                redirect_stderr(devnull) do
                    JLD2.load(state_path, "sensitivity_by_stage")
                end
            catch
                nothing
            end
            (data === nothing || isempty(data)) && continue

            for (stage, sd) in data
                traj = hasproperty(sd, :trajectory_sensitivity_norm) ?
                    sd.trajectory_sensitivity_norm : fill(NaN, length(sd.stiffness))
                for i in eachindex(sd.stiffness)
                    println(io, join((entry, stage, i, sd.stiffness[i], sd.sensitivity_norm[i], traj[i]), ","))
                end
            end
            n_runs += 1
        end
    end
    println("wrote $out_csv from $n_runs runs")
end

main()
