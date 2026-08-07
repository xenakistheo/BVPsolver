
# README

Idea with this refactoring is that each time you launch a run, the results/data it generates will be saved for future analysis. Easier to log. Moreover, this structure will make it easier to scale up and run multiple jobs in the coming weeks on a compute cluster. 

### Getting started 
`julia --project=. -e 'using Pkg; Pkg.instantiate()'`


### Start a run
You can either start a run using
`julia --project=. src/run_main.jl [--problem NAME] [--param VALUE] [--profile fast|default]                            [--seed N] [--no-derivmatch] [--no-collocation]`

E.g. 
`julia --project=. src/run_main.jl --problem rober --no-collocation`

An explanation of the flags can be found using `julia --project=. src/run_main.jl --help` for all flags.

One can also run the script by running 
`./src/scripts/local/run_main.sh`

If you have problems with permissions, do `chmod +x ./src/scripts/local/run_main.sh` first. 
You can also edit the parameters directly in the shell file (so you don't have to write out the parameters in the terminal.)


### Database
Both methods of running the script produce a folder in `data/runs/`. The `run_info.toml` file should contain information about the run. 
There is a "database" of sorts keeping track of all the runs, in `analysis/runs_index.csv`. This can be updated by running
`python3 analysis/build_runs_index.py`




