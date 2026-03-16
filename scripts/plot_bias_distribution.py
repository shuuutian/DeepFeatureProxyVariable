"""
Replot bias distribution from an existing run directory.

Reads bias_distribution_values.csv (method, bias) from a directory produced by
compare_dfpv_variants.py and saves bias_distribution.png in the same directory.
Use this when you want to change the plot style or regenerate the figure without
rerunning the experiments.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import gaussian_kde


def _load_bias_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """Load bias_distribution_values.csv into method -> bias array."""
    biases_by_label: Dict[str, list] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["method", "bias"]:
            raise ValueError(
                f"Expected columns ['method', 'bias'], got {reader.fieldnames}"
            )
        for row in reader:
            method = row["method"].strip()
            bias = float(row["bias"])
            biases_by_label.setdefault(method, []).append(bias)

    return {k: np.asarray(v, dtype=float) for k, v in biases_by_label.items()}


def _plot_bias_distribution(
    out_dir: Path,
    biases_by_label: Dict[str, np.ndarray],
    index_range: tuple[int, int] | None = None,
) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install dependencies and rerun."
        ) from exc

    color_map = {
        "Modified DFPV": "#B221E2",
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
        ax.hist(
            vals,
            bins=18,
            density=True,
            alpha=0.22,
            edgecolor=color,
            linewidth=1.6,
            color=color,
            label=label,
        )
        if vals.size >= 2 and np.std(vals) > 0:
            kde = gaussian_kde(vals)
            ax.plot(x_grid, kde(x_grid), color=color, linewidth=2.0)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    title = "Bias = ATE_hat - ATE_true (distribution)"
    if index_range is not None:
        title += f" [indices {index_range[0]}:{index_range[1]}]"
    ax.set_title(title)
    ax.set_xlabel("Bias")
    ax.set_ylabel("Density")
    ax.legend(title="Method")
    fig.tight_layout()

    if index_range is not None:
        out_name = f"bias_distribution_{index_range[0]}_{index_range[1]}.png"
    else:
        out_name = "bias_distribution.png"
    out_path = out_dir.joinpath(out_name)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot bias distribution from existing bias_distribution_values.csv"
    )
    parser.add_argument(
        "dir",
        type=Path,
        help="Directory containing bias_distribution_values.csv",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to CSV file (default: <dir>/bias_distribution_values.csv)",
    )
    parser.add_argument(
        "--index-range",
        type=int,
        nargs=2,
        metavar=("START", "STOP"),
        default=None,
        help="Per-method index range (like range(START, STOP)). E.g. 100 200 uses indices 100..199 for each method.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {run_dir}")

    csv_path = args.csv.resolve() if args.csv else run_dir / "bias_distribution_values.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    biases_by_label = _load_bias_csv(csv_path)
    if not biases_by_label:
        raise ValueError(f"No data in {csv_path}")

    index_range = None
    if args.index_range is not None:
        start, stop = args.index_range[0], args.index_range[1]
        if start < 0 or stop < start:
            raise ValueError(f"Invalid --index-range: start={start}, stop={stop}")
        index_range = (start, stop)
        biases_by_label = {
            method: arr[start:stop]
            for method, arr in biases_by_label.items()
        }
        n_per = next(len(v) for v in biases_by_label.values())
        if n_per == 0:
            raise ValueError(
                f"Index range [{start}:{stop}] yields no points for at least one method."
            )
        print(f"Using per-method index range [{start}:{stop}] -> {n_per} points each")

    out_path = _plot_bias_distribution(run_dir, biases_by_label, index_range=index_range)
    print(f"Bias plot written to: {out_path}")


if __name__ == "__main__":
    main()
