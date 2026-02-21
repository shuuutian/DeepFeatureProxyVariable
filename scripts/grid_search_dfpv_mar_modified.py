from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ate import generate_test_data_ate
from src.experiment import experiments


DEFAULT_CONFIG = Path("configs/dfpv_mar_modified_grid_stage1.json")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _parse_model_key(model_key: str) -> Dict[str, Any]:
    if model_key == "one":
        return {}
    params: Dict[str, Any] = {}
    for token in model_key.split("-"):
        if ":" not in token:
            continue
        k, v = token.split(":", 1)
        try:
            if "." in v or "e" in v.lower():
                params[k] = float(v)
            else:
                params[k] = int(v)
        except ValueError:
            params[k] = v
    return params


def _collect_setting_stats(
    run_root: Path,
    ate_true: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result_file in sorted(run_root.rglob("result.csv")):
        result_dir = result_file.parent
        rel_parts = result_file.relative_to(run_root).parts
        if len(rel_parts) == 2:
            data_key, model_key = rel_parts[0], "one"
        elif len(rel_parts) >= 3:
            data_key, model_key = rel_parts[0], rel_parts[1]
        else:
            data_key, model_key = "unknown", "unknown"

        losses = np.loadtxt(result_file)
        losses = np.atleast_1d(losses).astype(float)

        biases: List[float] = []
        for pred_file in sorted(result_dir.glob("*.pred.txt")):
            pred = np.loadtxt(pred_file)
            pred = np.atleast_1d(pred).astype(float)
            ate_hat = float(np.mean(pred))
            biases.append(ate_hat - ate_true)
        bias_arr = np.asarray(biases, dtype=float) if biases else np.array([], dtype=float)

        row: Dict[str, Any] = {
            "data_key": data_key,
            "model_key": model_key,
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses, ddof=0)),
            "num_runs": int(losses.size),
            "mean_bias": float(np.mean(bias_arr)) if bias_arr.size else np.nan,
            "std_bias": float(np.std(bias_arr, ddof=0)) if bias_arr.size else np.nan,
            "mean_abs_bias": float(np.mean(np.abs(bias_arr))) if bias_arr.size else np.nan,
            "num_pred_files": int(bias_arr.size),
        }
        row.update(_parse_model_key(model_key))
        rows.append(row)
    rows.sort(key=lambda x: x["mean_loss"])
    return rows


def _write_summary_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    if not rows:
        raise ValueError("No result.csv files found to summarize.")
    fixed_cols = [
        "rank",
        "data_key",
        "model_key",
        "mean_loss",
        "std_loss",
        "num_runs",
        "mean_bias",
        "std_bias",
        "mean_abs_bias",
        "num_pred_files",
    ]
    dynamic_cols = sorted(set().union(*(set(r.keys()) for r in rows)) - set(fixed_cols))
    cols = fixed_cols + dynamic_cols

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            out = dict(row)
            out["rank"] = i
            writer.writerow(out)


def _write_topk_json(rows: List[Dict[str, Any]], out_json: Path, k: int = 5) -> None:
    top_rows = rows[:k]
    with out_json.open("w") as f:
        json.dump(top_rows, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and analyze hyperparameter grid search for dfpv_mar_modified"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--dump-root", type=Path, default=Path("dumps"))
    parser.add_argument(
        "--analyze-only",
        type=Path,
        default=None,
        help="Skip training and analyze an existing run directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.analyze_only is None:
        config = _load_json(args.config)
        ts = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        run_root = args.dump_root.joinpath(f"grid_dfpv_mar_modified_{ts}")
        os.makedirs(run_root, exist_ok=False)
        print(f"Run directory: {run_root}")
        print("Running grid search...")
        experiments(config, run_root, num_cpus=args.num_cpus, num_gpu=None)
    else:
        run_root = args.analyze_only
        config = _load_json(args.config)
        print(f"Analyze-only mode. Run directory: {run_root}")

    test_data = generate_test_data_ate(config["data"])
    if test_data is None or test_data.structural is None:
        raise ValueError("Cannot compute ATE_true because test structural data is unavailable.")
    ate_true = float(np.mean(test_data.structural))

    print("Analyzing results...")
    rows = _collect_setting_stats(run_root, ate_true=ate_true)
    summary_csv = run_root.joinpath("grid_summary.csv")
    top_json = run_root.joinpath("top5_by_mean_loss.json")
    _write_summary_csv(rows, summary_csv)
    _write_topk_json(rows, top_json, k=5)

    best = rows[0]
    print(f"Summary written to: {summary_csv}")
    print(f"Top-5 written to: {top_json}")
    print(
        "Best setting: "
        f"model_key={best['model_key']}, mean_loss={best['mean_loss']:.6f}, "
        f"mean_abs_bias={best['mean_abs_bias']:.6f}"
    )


if __name__ == "__main__":
    main()

