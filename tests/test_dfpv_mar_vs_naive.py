import numpy as np
import torch
from pathlib import Path

from src.data.ate import (
    generate_train_data_ate_mar,
    generate_test_data_ate,
    get_preprocessor_ate,
)
from src.data.ate.data_class import PVTrainDataSet, PVTestDataSetTorch
from src.models.DFPV.trainer import DFPVTrainer
from src.models.DFPV.trainer_mar import dfpv_experiments_mar


def run_naive_vs_mar(
    random_seed: int = 42,
    n_sample: int = 500,
    missing_rate: float = 0.5,
) -> tuple[float, float]:
    """
    Compare:
      1) Naive baseline: original DFPV trained only on complete cases (delta_W=1)
      2) MAR-aware: DFPV_mar using all data with DR correction
    on the synthetic KPV environment.

    Returns:
        (oos_loss_naive, oos_loss_mar)
    """
    # Shared data/model configs
    data_config = {
        "name": "kpv",
        "n_sample": n_sample,
        "missing_rate": missing_rate,
        "preprocess": "ScaleAll",
    }

    model_param = {
        "lam1": 0.01,
        "lam2": 0.01,
        "stage1_iter": 10,
        "stage2_iter": 1,
        "n_epoch": 30,
        "split_ratio": 0.5,
        "treatment_weight_decay": 0.1,
        "treatment_proxy_weight_decay": 0.1,
        "outcome_proxy_weight_decay": 0.1,
        "backdoor_weight_decay": 0.0,
        "n_folds": 3,
        "nuisance_n_epochs": 20,
        "propensity_clip": 1e-3,
    }

    # -------------------------
    # MAR-aware DFPV (DFPV_mar)
    # -------------------------
    mar_dump_dir = Path("tmp_mar_test")
    mar_dump_dir.mkdir(parents=True, exist_ok=True)

    oos_loss_mar = dfpv_experiments_mar(
        data_config=data_config,
        model_param=model_param,
        one_mdl_dump_dir=mar_dump_dir,
        random_seed=random_seed,
        verbose=0,
    )

    # -------------------------
    # Naive baseline: original DFPV on complete cases only
    # -------------------------
    from src.data.ate import generate_train_data_ate_mar  # re-import to avoid circularity

    train_data_mar = generate_train_data_ate_mar(
        data_config=data_config,
        rand_seed=random_seed,
    )
    test_data_org = generate_test_data_ate(data_config)

    # Restrict to rows with observed W (delta_w == 1)
    delta = train_data_mar.delta_w.squeeze()
    mask = delta == 1.0

    backdoor_obs = (
        train_data_mar.backdoor[mask] if train_data_mar.backdoor is not None else None
    )
    train_complete = PVTrainDataSet(
        treatment=train_data_mar.treatment[mask],
        treatment_proxy=train_data_mar.treatment_proxy[mask],
        outcome_proxy=train_data_mar.outcome_proxy[mask],
        outcome=train_data_mar.outcome[mask],
        backdoor=backdoor_obs,
    )

    preprocessor = get_preprocessor_ate(data_config["preprocess"])
    train_complete_p = preprocessor.preprocess_for_train(train_complete)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)

    torch.manual_seed(random_seed)
    trainer = DFPVTrainer(data_config, model_param, gpu_flg=False)
    mdl = trainer.train(train_complete_p, verbose=0)

    test_data_t = PVTestDataSetTorch.from_numpy(test_data)
    if trainer.gpu_flg:
        test_data_t = test_data_t.to_gpu()

    with torch.no_grad():
        pred = mdl.predict_t(test_data_t.treatment).cpu().numpy()
    pred = preprocessor.postprocess_for_prediction(pred)

    if test_data_org.structural is not None:
        oos_loss_naive = float(np.mean((pred - test_data_org.structural) ** 2))
    else:
        oos_loss_naive = float("nan")

    return oos_loss_naive, float(oos_loss_mar)


if __name__ == "__main__":
    naive, mar = run_naive_vs_mar()
    print(f"Naive DFPV (complete cases only) OOS loss: {naive:.6f}")
    print(f"DFPV_mar (MAR-aware) OOS loss:          {mar:.6f}")
    if np.isfinite(naive) and np.isfinite(mar):
        if mar < naive:
            print("DFPV_mar performs better (lower loss) than the naive complete-case DFPV.")
        else:
            print("Naive complete-case DFPV performed better in this run (consider more seeds/repeats).")

