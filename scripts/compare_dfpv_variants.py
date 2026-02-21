from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple

import numpy as np
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment import experiments
from src.data.ate import generate_test_data_ate


DEFAULT_CONFIGS = {
    "dfpv": Path("configs/dfpv_mar_baseline.json"),
    "dfpv_mar_modified": Path("configs/dfpv_mar_modified.json"),
    "dfpv_mar_naive": Path("configs/dfpv_mar_naive.json"),
}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _collect_results(result_root: Path) -> np.ndarray:
    values: List[np.ndarray] = []
    for result_file in sorted(result_root.rglob("result.csv")):
        arr = np.loadtxt(result_file)
        arr = np.atleast_1d(arr).astype(float)
        values.append(arr)

    if not values:
        raise FileNotFoundError(f"No result.csv found under {result_root}")

    return np.concatenate(values)


def _run_one(
    scenario_name: str,
    cfg_path: Path,
    run_root: Path,
    n_repeat_override: int | None,
    num_cpus: int,
) -> Tuple[float, float, int]:
    cfg = _load_json(cfg_path)
    if n_repeat_override is not None:
        cfg["n_repeat"] = n_repeat_override

    scenario_dir = run_root.joinpath(scenario_name)
    os.makedirs(scenario_dir, exist_ok=False)

    experiments(cfg, scenario_dir, num_cpus=num_cpus, num_gpu=None)

    vals = _collect_results(scenario_dir)
    return float(vals.mean()), float(vals.std(ddof=0)), int(vals.size)


def _compute_biases_from_predictions(
    scenario_dir: Path,
    data_config: Dict[str, Any],
) -> np.ndarray:
    test_data = generate_test_data_ate(data_config=data_config)
    if test_data is None or test_data.structural is None:
        raise ValueError(f"Test structural data unavailable for {data_config['name']}.")

    ate_true = float(np.mean(test_data.structural))
    biases: List[float] = []
    for pred_file in sorted(scenario_dir.rglob("*.pred.txt")):
        pred = np.loadtxt(pred_file)
        pred = np.atleast_1d(pred).astype(float)
        ate_hat = float(np.mean(pred))
        biases.append(ate_hat - ate_true)

    if not biases:
        raise FileNotFoundError(f"No *.pred.txt found under {scenario_dir}")
    return np.asarray(biases, dtype=float)


def _plot_bias_distribution(
    run_root: Path,
    biases_by_label: Dict[str, np.ndarray],
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting. Install dependencies and rerun."
        ) from exc

    color_map = {
        "Modified DFPV": "#4C72B0",
        "Naive DFPV": "#DD8452",
        "Oracle DFPV": "#55A868",
    }

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(9, 5.6))
    all_vals = np.concatenate([v for v in biases_by_label.values()])
    x_min, x_max = float(all_vals.min()), float(all_vals.max())
    pad = 0.1 * max(1e-8, x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 500)

    for label, vals in biases_by_label.items():
        color = color_map.get(label, None)
        ax.hist(vals, bins=18, density=True, alpha=0.22, edgecolor=color, linewidth=1.6, color=color, label=label)
        if vals.size >= 2 and np.std(vals) > 0:
            kde = gaussian_kde(vals)
            ax.plot(x_grid, kde(x_grid), color=color, linewidth=2.0)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Bias = ATE_hat - ATE_true (distribution)")
    ax.set_xlabel("Bias")
    ax.set_ylabel("Density")
    ax.legend(title="Method")
    fig.tight_layout()

    out_path = run_root.joinpath("bias_distribution.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and compare DFPV variants: dfpv, dfpv_mar_modified, dfpv_mar_naive"
    )
    parser.add_argument("--dfpv-config", type=Path, default=DEFAULT_CONFIGS["dfpv"])
    parser.add_argument("--mar-modified-config", type=Path, default=DEFAULT_CONFIGS["dfpv_mar_modified"])
    parser.add_argument("--mar-naive-config", type=Path, default=DEFAULT_CONFIGS["dfpv_mar_naive"])
    parser.add_argument("--n-repeat", type=int, default=None, help="Override n_repeat in all 3 configs")
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--dump-root", type=Path, default=Path("dumps"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ts = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    run_root = args.dump_root.joinpath(f"compare_dfpv_variants_{ts}")
    os.makedirs(run_root, exist_ok=False)

    scenario_cfgs = [
        ("dfpv", args.dfpv_config),
        ("dfpv_mar_modified", args.mar_modified_config),
        ("dfpv_mar_naive", args.mar_naive_config),
    ]

    summary_rows: List[Tuple[str, float, float, int, str, str]] = []
    scenario_notes = {
        "dfpv": "oracle_full_observed",
        "dfpv_mar_modified": "mar_method",
        "dfpv_mar_naive": "mar_complete_case",
    }
    scenario_labels = {
        "dfpv": "Oracle DFPV",
        "dfpv_mar_modified": "Modified DFPV",
        "dfpv_mar_naive": "Naive DFPV",
    }
    biases_by_label: Dict[str, np.ndarray] = {}

    print("Running tasks:")
    print("1. Run dfpv (oracle/full-observed baseline)")
    print("2. Run dfpv_mar_modified")
    print("3. Run dfpv_mar_naive")
    print("4. Aggregate result.csv files and write comparison summary")
    print(f"Output root: {run_root}")

    for scenario_name, cfg_path in scenario_cfgs:
        cfg = _load_json(cfg_path)
        mean, std, n = _run_one(
            scenario_name=scenario_name,
            cfg_path=cfg_path,
            run_root=run_root,
            n_repeat_override=args.n_repeat,
            num_cpus=args.num_cpus,
        )
        summary_rows.append((scenario_name, mean, std, n, str(cfg_path), scenario_notes[scenario_name]))
        biases_by_label[scenario_labels[scenario_name]] = _compute_biases_from_predictions(
            scenario_dir=run_root.joinpath(scenario_name),
            data_config=cfg["data"],
        )
        print(f"[{scenario_name}] n={n}, mean={mean:.6f}, std={std:.6f}")

    summary_csv = run_root.joinpath("comparison_summary.csv")
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "mean_loss", "std_loss", "num_runs", "config_path", "comparison_note"])
        writer.writerows(summary_rows)

    bias_csv = run_root.joinpath("bias_distribution_values.csv")
    with bias_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "bias"])
        for method, vals in biases_by_label.items():
            for v in vals:
                writer.writerow([method, float(v)])

    bias_fig = _plot_bias_distribution(run_root, biases_by_label)

    print(f"Summary written to: {summary_csv}")
    print(f"Bias values written to: {bias_csv}")
    print(f"Bias plot written to: {bias_fig}")


if __name__ == "__main__":
    main()
