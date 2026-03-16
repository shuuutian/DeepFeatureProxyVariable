import numpy as np
import torch

from src.data.ate import (
    generate_train_data_ate_mar,
    generate_test_data_ate,
    get_preprocessor_ate,
)
from src.data.ate.data_class import PVTrainDataSet, PVTestDataSetTorch
from src.data.ate.data_class_mar import PVTrainDataSetMAR, PVTrainDataSetMARTorch
from src.models.DFPV.trainer import DFPVTrainer
from src.models.DFPV.trainer_mar import DFPVTrainerMAR


def run_naive_vs_mar(
    random_seed: int = 42,
    n_sample: int = 10000,
    mar_threshold: float = 0.0,
) -> tuple[float, float]:
    """
    Compare, on the synthetic demand environment:
      1) Naive baseline: original DFPV on complete cases only (delta_W=1)
      2) MAR-aware: DFPV_mar using all data with DR correction

    Both methods share the *same* underlying MAR dataset: we generate one
    PVTrainDataSetMAR and:
      - feed only complete cases to original DFPV
      - feed all samples (with delta_W) to DFPVTrainerMAR

    Returns:
        (oos_loss_naive, oos_loss_mar)
    """
    # Data and model configs
    data_config = {
        "name": "demand",
        "n_sample": n_sample,
        "mar_threshold": mar_threshold,
        "preprocess": "Identity",
    }

    naive_params = {
        "lam1": 0.01,
        "lam2": 0.01,
        "stage1_iter": 10,
        "stage2_iter": 10,
        "n_epoch": 30,
        "split_ratio": 0.5,
        "treatment_weight_decay": 0.1,
        "treatment_proxy_weight_decay": 0.1,
        "outcome_proxy_weight_decay": 0.1,
        "backdoor_weight_decay": 0.0,
    }

    mar_params = {
        **naive_params,
        "n_folds": 3,
        "nuisance_n_epochs": 20,
        "propensity_clip": 1e-3,
    }

    # ------------------------------------------------------------------
    # Shared MAR dataset (one draw) + test set
    # ------------------------------------------------------------------
    torch.manual_seed(random_seed)
    train_data_mar: PVTrainDataSetMAR = generate_train_data_ate_mar(
        data_config=data_config,
        rand_seed=random_seed,
    )
    test_data_org = generate_test_data_ate(data_config)

    preprocessor = get_preprocessor_ate(data_config["preprocess"])

    # ------------------------------------------------------------------
    # Naive baseline: original DFPV on complete cases only
    # ------------------------------------------------------------------
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

    train_complete_p = preprocessor.preprocess_for_train(train_complete)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)

    trainer_naive = DFPVTrainer(data_config, naive_params, gpu_flg=False)
    mdl_naive = trainer_naive.train(train_complete_p, verbose=0)

    test_data_t = PVTestDataSetTorch.from_numpy(test_data)
    if trainer_naive.gpu_flg:
        test_data_t = test_data_t.to_gpu()
    with torch.no_grad():
        pred_naive = mdl_naive.predict_t(test_data_t.treatment).cpu().numpy()

    pred_naive = preprocessor.postprocess_for_prediction(pred_naive)
    oos_loss_naive = float(np.mean((pred_naive - test_data_org.structural) ** 2))

    # ------------------------------------------------------------------
    # MAR-aware DFPV: DFPVTrainerMAR on all samples with delta_W
    # ------------------------------------------------------------------
    train_mar_p = preprocessor.preprocess_for_train_mar(train_data_mar)
    train_mar_t = PVTrainDataSetMARTorch.from_numpy(train_mar_p)

    trainer_mar = DFPVTrainerMAR(
        data_configs=data_config,
        train_params=mar_params,
        gpu_flg=False,
        random_seed=random_seed,
    )
    mdl_mar = trainer_mar.train(train_mar_p, verbose=0)

    test_treatment_t = torch.tensor(test_data.treatment, dtype=torch.float32)
    train_proxy_t = train_mar_t.treatment_proxy
    train_backdoor_t = train_mar_t.backdoor

    with torch.no_grad():
        pred_mar = (
            mdl_mar.predict_t(
                treatment=test_treatment_t,
                train_treatment_proxy=train_proxy_t,
                train_backdoor=train_backdoor_t,
            )
            .cpu()
            .numpy()
        )

    pred_mar = preprocessor.postprocess_for_prediction(pred_mar)
    oos_loss_mar = float(np.mean((pred_mar - test_data_org.structural) ** 2))

    # Also return the full ATE curves for inspection/printing:
    # test_data_org.treatment: grid of intervention prices
    # test_data_org.structural: true β(a) on that grid
    return (
        oos_loss_naive,
        oos_loss_mar,
        test_data_org.treatment.squeeze(),
        test_data_org.structural.squeeze(),
        pred_naive.squeeze(),
        pred_mar.squeeze(),
    )


if __name__ == "__main__":
    (
        naive_loss,
        mar_loss,
        a_grid,
        beta_true,
        beta_naive,
        beta_mar,
    ) = run_naive_vs_mar()

    print(f"Naive DFPV (complete cases only) OOS loss: {naive_loss:.6f}")
    print(f"DFPV_mar (MAR-aware) OOS loss:          {mar_loss:.6f}")

    print("\nATE curves on intervention grid (price a):")
    print("a\ttrue_beta(a)\tnaive_hat(a)\tmar_hat(a)")
    for a, bt, bn, bm in zip(a_grid, beta_true, beta_naive, beta_mar):
        print(f"{float(a):.3f}\t{float(bt):.6f}\t{float(bn):.6f}\t{float(bm):.6f}")

    if np.isfinite(naive_loss) and np.isfinite(mar_loss):
        if mar_loss < naive_loss:
            print("\nDFPV_mar performs better (lower loss) than the naive complete-case DFPV.")
        else:
            print("\nNaive complete-case DFPV performed better in this run (consider more seeds/repeats).")

