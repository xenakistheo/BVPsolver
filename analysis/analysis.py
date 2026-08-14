import pandas as pd

pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)

DB_PATH = "analysis/runs_index.csv"

DB = pd.read_csv(DB_PATH, index_col=0)

PROBLEMS = ['vanderpol', 'pollu', 'rober', 'orego', 'hires', 'davis-skodje']

METRIC = "E_trajectory_test"
# METRIC = "E_trajectory_train"
# METRIC = "E_VF_test"
# METRIC = "E_spec_train"

AGG_FUNC = "min"  # "min" or "mean"

# remove rows without E_trajectory_train
DB = DB[DB['E_trajectory_train'].notna()]

# PROBLEM = PROBLEMS[1]
print()
for (i, PROBLEM) in enumerate(PROBLEMS): 
    print(f"Analyzing problem {i+1}/{len(PROBLEMS)}: {PROBLEM}")
    sub = DB[DB["problem"] == PROBLEM].copy()
    sub["config"] = sub["pretraining"] + "+" + sub["training"]

    table = sub.pivot_table(
        index="model", columns="config", values=METRIC, aggfunc=AGG_FUNC
    )

    print(f"min {METRIC} across seeds for {PROBLEM}")
    print(table)
    best_model, best_config = table.stack().idxmin()
    print(f"Best configuration: model={best_model}, config={best_config}")
    print()

print()