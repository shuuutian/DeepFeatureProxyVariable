from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

BASELINE_CFG = REPO_ROOT / "configs" / "dfpv_mar_baseline.json"
MODIFIED_CFG = REPO_ROOT / "configs" / "dfpv_mar_modified_NEW.json"
NAIVE_CFG = REPO_ROOT / "configs" / "dfpv_mar_naive.json"

# Grid values
N_SAMPLES = [1000, 3000]
MAR_THRESHOLDS = [-1, -1.5, -2.5]
MAR_ALPHAS = [1.0]

TMP_DIR = REPO_ROOT / "configs" / "grid_tmp"


def _load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=4)


def main() -> None:
    baseline_base = _load_json(BASELINE_CFG)
    modified_base = _load_json(MODIFIED_CFG)
    naive_base = _load_json(NAIVE_CFG)

    for n_sample, mar_threshold, mar_alpha in itertools.product(
        N_SAMPLES, MAR_THRESHOLDS, MAR_ALPHAS
    ):
        suffix = f"n{n_sample}_th{mar_threshold}_a{mar_alpha}".replace(".", "p")

        baseline_cfg_path = TMP_DIR / f"dfpv_baseline_{suffix}.json"
        modified_cfg_path = TMP_DIR / f"dfpv_mar_modified_{suffix}.json"
        naive_cfg_path = TMP_DIR / f"dfpv_mar_naive_{suffix}.json"

        # Clone base configurations
        baseline_cfg = json.loads(json.dumps(baseline_base))
        modified_cfg = json.loads(json.dumps(modified_base))
        naive_cfg = json.loads(json.dumps(naive_base))

        # Set n_sample for all three configs
        for cfg in (baseline_cfg, modified_cfg, naive_cfg):
            cfg.setdefault("data", {})
            cfg["data"]["n_sample"] = n_sample

        # Set MAR-specific parameters for the MAR-based methods
        for cfg in (modified_cfg, naive_cfg):
            cfg["data"]["mar_threshold"] = mar_threshold
            cfg["data"]["mar_alpha_value"] = mar_alpha

        _save_json(baseline_cfg, baseline_cfg_path)
        _save_json(modified_cfg, modified_cfg_path)
        _save_json(naive_cfg, naive_cfg_path)

        dump_root = REPO_ROOT / "dumps" / f"grid_{suffix}"

        print(
            f"Running grid point: n_sample={n_sample}, "
            f"mar_threshold={mar_threshold}, mar_alpha_value={mar_alpha}"
        )

        subprocess.run(
            [
                "python",
                "scripts/compare_dfpv_variants.py",
                "--dfpv-config",
                str(baseline_cfg_path),
                "--mar-modified-config",
                str(modified_cfg_path),
                "--mar-naive-config",
                str(naive_cfg_path),
                "--dump-root",
                str(dump_root),
            ],
            check=True,
            cwd=REPO_ROOT,
        )


if __name__ == "__main__":
    main()

