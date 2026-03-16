"""
Simple experiment script to study how the MAR threshold affects the empirical
missing rate of the outcome proxy W in the demand DGP.

Usage (from repo root):

    python scripts/inspect_mar_missing_rate.py \
        --n-sample 10000 \
        --mar-alpha 1.6 \
        --min-threshold -1.5 \
        --max-threshold 1.5 \
        --n-thresholds 13 \
        --n-seeds 10

This will print a table of (threshold, avg_missing_rate, std_missing_rate)
over the specified random seeds.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np

from src.data.ate import generate_train_data_ate_mar


def _estimate_missing_rate_for_threshold(
    mar_threshold: float,
    n_sample: int,
    mar_alpha_value: float,
    n_seeds: int,
    base_seed: int,
) -> Tuple[float, float]:
    """Estimate mean and std missing rate over multiple seeds for one threshold."""
    rates = []
    for k in range(n_seeds):
        seed = base_seed + k
        data_config = {
            "name": "demand",
            "n_sample": n_sample,
            "mar_threshold": mar_threshold,
            "mar_alpha_value": mar_alpha_value,
            "preprocess": "Identity",
        }
        train_data = generate_train_data_ate_mar(data_config=data_config, rand_seed=seed)
        delta = train_data.delta_w  # 1 = observed, 0 = missing
        # delta_w has shape (n, 1)
        observed_rate = float(delta.mean())
        missing_rate = 1.0 - observed_rate
        rates.append(missing_rate)

    arr = np.asarray(rates, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect how mar_threshold affects the empirical missing rate "
        "of W in the demand DGP.",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=10000,
        help="Number of training samples to generate per run (default: 10000).",
    )
    parser.add_argument(
        "--mar-alpha",
        type=float,
        default=1.6,
        help="mar_alpha_value used in the MAR score (default: 1.6).",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=-1.5,
        help="Minimum mar_threshold value in the sweep (default: -1.5).",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=1.5,
        help="Maximum mar_threshold value in the sweep (default: 1.5).",
    )
    parser.add_argument(
        "--n-thresholds",
        type=int,
        default=13,
        help="Number of thresholds to evaluate between min and max (default: 13).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of random seeds per threshold (default: 10).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed; seeds are base_seed + k (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    thresholds = np.linspace(
        args.min_threshold, args.max_threshold, args.n_thresholds
    ).tolist()

    print(
        "mar_threshold\tmean_missing_rate\tstd_missing_rate\t"
        f"(n_sample={args.n_sample}, mar_alpha={args.mar_alpha}, n_seeds={args.n_seeds})"
    )
    for th in thresholds:
        mean_miss, std_miss = _estimate_missing_rate_for_threshold(
            mar_threshold=th,
            n_sample=args.n_sample,
            mar_alpha_value=args.mar_alpha,
            n_seeds=args.n_seeds,
            base_seed=args.base_seed,
        )
        print(f"{th:.4f}\t{mean_miss:.4f}\t{std_miss:.4f}")


if __name__ == "__main__":
    main()

