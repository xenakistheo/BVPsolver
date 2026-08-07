"""Build a CSV index of every run in data/runs.

Scans each run_* subfolder for a run_info.toml file and writes one CSV row
per run, with one column per param key found (union across all runs; nested
tables such as [cfg] are flattened with a dotted prefix; missing values are
left blank). Run with:

    python3 analysis/build_runs_index.py
"""
from __future__ import annotations

import argparse
import csv
import os

import toml

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ANALYSIS_DIR, ".."))
DEFAULT_RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs")
DEFAULT_OUTPUT = os.path.join(ANALYSIS_DIR, "runs_index.csv")

ID_COLUMN = "run_dir"


def find_runs(runs_dir: str) -> list[str]:
    return sorted(
        entry.name
        for entry in os.scandir(runs_dir)
        if entry.is_dir() and os.path.isfile(os.path.join(entry.path, "run_info.toml"))
    )


def flatten(params: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def load_rows(runs_dir: str, run_names: list[str]) -> list[dict]:
    rows = []
    for name in run_names:
        info_path = os.path.join(runs_dir, name, "run_info.toml")
        with open(info_path, "r", encoding="utf-8") as f:
            params = toml.load(f)
        rows.append({ID_COLUMN: name, **flatten(params)})
    return rows


def write_csv(rows: list[dict], output_path: str) -> list[str]:
    columns = [ID_COLUMN]
    seen = {ID_COLUMN}
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    return columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    run_names = find_runs(args.runs_dir)
    if not run_names:
        print(f"No runs with run_info.toml found under {args.runs_dir}")
        return

    rows = load_rows(args.runs_dir, run_names)
    columns = write_csv(rows, args.output)

    print(f"Indexed {len(rows)} runs into {args.output}")
    print(f"Columns ({len(columns)}): {', '.join(columns)}")


if __name__ == "__main__":
    main()
