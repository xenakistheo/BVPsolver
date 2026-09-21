
# This file provides data for the main results of the paper. 
# It reads through the runs_index.csv

import math
import pandas as pd

#python3 analysis/analysis.py

LATEXIFY = False

pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)

DB_PATH = "analysis/runs_index.csv"

DB = pd.read_csv(DB_PATH, index_col=0)

PROBLEMS = ['vanderpol', 'pollu', 'rober', 'orego', 'hires', 'davis-skodje']

METRIC = "E_trajectory_test"
# METRIC = "E_trajectory_train"
# METRIC = "E_VF_test"
METRIC = "E_spec_train"

AGG_FUNC = "min"  # "min" or "mean"

# Set both to inspect the best run_dir/seed for a specific model+config across
# all problems, e.g. INSPECT_MODEL = "GELU-scaled"; INSPECT_CONFIG = "derivmatch+shooting"
# INSPECT_MODEL = "GELU-scaled"
# INSPECT_CONFIG = "derivmatch+shooting"
INSPECT_MODEL = None
INSPECT_CONFIG = None

# column config -> (header label, group: "single" or "combined")
CONFIG_COLUMNS = [
    ("derivmatch+none", "D", "single"),
    ("none+collocation", "C", "single"),
    ("none+shooting", "S", "single"),
    ("derivmatch+collocation", "D+C", "combined"),
    ("derivmatch+shooting", "D+S", "combined"),
]

MODEL_ORDER = ["GELU-scaled", "mlp", "stiff"]
MODEL_DISPLAY = {"GELU-scaled": "GELU-scaled", "mlp": "Baseline MLP", "stiff": "StiffNet"}

PROBLEM_DISPLAY = {
    "vanderpol": "Van der Pol",
    "pollu": "POLLU",
    "rober": "ROBER",
    "orego": "OREGO",
    "hires": "HIRES",
    "davis-skodje": "Davis--Skodje",
}

PROBLEM_LABEL = {
    "vanderpol": "vdp",
    "pollu": "pollu",
    "rober": "rober",
    "orego": "orego",
    "hires": "hires",
    "davis-skodje": "ds",
}


def _fmt_cell(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    av = abs(value)
    if av != 0 and (av >= 1e4 or av < 1e-3):
        return f"{value:.3e}"
    return f"{value:.3f}"


def render_latex_table(table, problem, metric, best_model, best_config):
    if "trajectory" in metric:
        quantity = "trajectory error"
    elif "VF" in metric:
        quantity = "vector field error"
    elif "spec" in metric:
        quantity = "spectral error"
    else:
        quantity = "error"
    split = "test" if metric.endswith("test") else "train"

    caption = f"$L_2$ {quantity} on the {PROBLEM_DISPLAY.get(problem, problem)} {split} trajectory"
    if problem == "vanderpol":
        caption += r" ($\mu=100$)"
    label = f"tab:main-{PROBLEM_LABEL.get(problem, problem)}"

    row_labels = [MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER if m in table.index]
    label_width = max(len(l) for l in row_labels) if row_labels else 0

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lccc|cc}",
        r"\toprule",
        r"& \multicolumn{3}{c|}{\textbf{Single}} ",
        r"& \multicolumn{2}{c}{\textbf{Combined}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-6}",
        r"\textbf{Architecture}",
        r"& D & C & S & D+C & D+S \\",
        r"\midrule",
    ]

    for model in MODEL_ORDER:
        if model not in table.index:
            continue
        cells = []
        for config, _header, _group in CONFIG_COLUMNS:
            value = table.loc[model, config] if config in table.columns else float("nan")
            cell = _fmt_cell(value)
            if model == best_model and config == best_config:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        disp = MODEL_DISPLAY.get(model, model).ljust(label_width)
        lines.append(f"{disp} & " + " & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)

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

    best_model, best_config = table.stack().idxmin()

    print(f"min {METRIC} across seeds for {PROBLEM}")
    if LATEXIFY:
        print(render_latex_table(table, PROBLEM, METRIC, best_model, best_config))
    else:
        print(table)
    print(f"Best configuration: model={best_model}, config={best_config}")
    if not LATEXIFY:
        best_rows = sub[(sub["model"] == best_model) & (sub["config"] == best_config)]
        best_run_dir = best_rows[METRIC].idxmin()
        best_seed = best_rows.loc[best_run_dir, "seed"]
        print(f"run_dir={best_run_dir}, seed={best_seed}")
    print()

if INSPECT_MODEL is not None and INSPECT_CONFIG is not None:
    print(f"Best {METRIC} for model={INSPECT_MODEL}, config={INSPECT_CONFIG} by problem")
    for PROBLEM in PROBLEMS:
        sub = DB[DB["problem"] == PROBLEM].copy()
        sub["config"] = sub["pretraining"] + "+" + sub["training"]
        sel = sub[(sub["model"] == INSPECT_MODEL) & (sub["config"] == INSPECT_CONFIG)]
        sel = sel[sel[METRIC].notna()]
        if sel.empty:
            print(f"{PROBLEM:14s} -- no runs")
            continue
        run_dir = sel[METRIC].idxmin()
        seed = sel.loc[run_dir, "seed"]
        print(f"{PROBLEM:14s} {METRIC}={sel.loc[run_dir, METRIC]:.4e}  seed={seed}  run_dir={run_dir}")
    print()

print()