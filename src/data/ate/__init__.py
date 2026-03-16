from typing import Dict, Any, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.ate.preprocess import get_preprocessor_ate
from src.data.ate.kpv_experiment_sim import generate_train_kpv_experiment, generate_test_kpv_experiment
from src.data.ate.deaner_experiment import generate_train_deaner_experiment, generate_test_deaner_experiment
from src.data.ate.demand_pv import generate_test_demand_pv, generate_train_demand_pv
from src.data.ate.dsprite import generate_train_dsprite, generate_test_dsprite
from src.data.ate.dsprite_ver2 import generate_train_dsprite_ver2, generate_test_dsprite_ver2
from src.data.ate.data_class import PVTestDataSet, PVTrainDataSet
from src.data.ate.data_class_mar import PVTrainDataSetMAR
from src.data.ate.cevae_experiment import generate_train_cevae_experiment, generate_test_cevae_experiment


def generate_train_data_ate(data_config: Dict[str, Any], rand_seed: int) -> PVTrainDataSet:
    data_name = data_config["name"]
    if data_name == "kpv":
        return generate_train_kpv_experiment(seed=rand_seed, **data_config)
    elif data_name == "demand":
        return generate_train_demand_pv(seed=rand_seed, **data_config)
    elif data_name == "dsprite_org":
        return generate_train_dsprite(rand_seed=rand_seed, **data_config)
    elif data_name == "dsprite":
        return generate_train_dsprite_ver2(rand_seed=rand_seed, **data_config)
    elif data_name == "cevae":
        return generate_train_cevae_experiment(rand_seed=rand_seed, **data_config)
    elif data_name == "deaner":
        return generate_train_deaner_experiment(seed=rand_seed, **data_config)
    else:
        raise ValueError(f"data name {data_name} is not valid")


def generate_test_data_ate(data_config: Dict[str, Any]) -> Optional[PVTestDataSet]:
    data_name = data_config["name"]
    if data_name == "kpv":
        return generate_test_kpv_experiment()
    elif data_name == "demand":
        return generate_test_demand_pv()
    elif data_name == "dsprite_org":
        return generate_test_dsprite()
    elif data_name == "dsprite":
        return generate_test_dsprite_ver2()
    elif data_name == "cevae":
        return generate_test_cevae_experiment()
    elif data_name == "deaner":
        return generate_test_deaner_experiment(id=data_config["id"])
    else:
        raise ValueError(f"data name {data_name} is not valid")


def generate_train_data_ate_mar(data_config: Dict[str, Any], rand_seed: int) -> PVTrainDataSetMAR:
    """Generate training data with MAR missingness indicator delta_W.

    Wraps the standard generator, then applies a deterministic threshold-based
    MAR mechanism to mask W:
        score_i = standardised(L+) @ alpha
        delta_W_i = 1 if score_i <= mar_threshold else 0

    Args:
        data_config: must contain "name" and optionally "mar_threshold" (default 0.0)
        rand_seed:   random seed for data generation

    Returns:
        PVTrainDataSetMAR with delta_w field; outcome_proxy contains zeros for missing rows
    """
    mar_threshold: float = data_config.get("mar_threshold", 0.0)

    base_data: PVTrainDataSet = generate_train_data_ate(data_config=data_config, rand_seed=rand_seed)

    n = base_data.treatment.shape[0]

    # Build L+ = (A, Z, X, Y) for MAR mechanism
    parts = [base_data.treatment, base_data.treatment_proxy, base_data.outcome]
    if base_data.backdoor is not None:
        parts.append(base_data.backdoor)
    L_plus = np.concatenate(parts, axis=1)  # (n, d_L+)

    L_std = (L_plus - L_plus.mean(axis=0)) / (L_plus.std(axis=0) + 1e-8)
    mar_alpha_value: float = data_config.get("mar_alpha_value", 1.6)
    alpha = np.full(L_std.shape[1], mar_alpha_value, dtype=np.float64)
    score = L_std @ alpha  # (n,)

    # Deterministic threshold: observed if score <= threshold, missing otherwise
    missing_mask = score > mar_threshold  # True = missing
    delta_w = (~missing_mask).astype(np.float32)  # 1=observed, 0=missing

    outcome_proxy_mar = base_data.outcome_proxy.copy()
    outcome_proxy_mar[missing_mask] = 0.0

    return PVTrainDataSetMAR(
        treatment=base_data.treatment,
        treatment_proxy=base_data.treatment_proxy,
        outcome_proxy=outcome_proxy_mar,
        outcome=base_data.outcome,
        backdoor=base_data.backdoor,
        delta_w=delta_w[:, np.newaxis],
    )


def standardise(data: PVTrainDataSet) -> Tuple[PVTrainDataSet, Dict[str, StandardScaler]]:
    treatment_proxy_scaler = StandardScaler()
    treatment_proxy_s = treatment_proxy_scaler.fit_transform(data.treatment_proxy)

    treatment_scaler = StandardScaler()
    treatment_s = treatment_scaler.fit_transform(data.treatment)

    outcome_scaler = StandardScaler()
    outcome_s = outcome_scaler.fit_transform(data.outcome)

    outcome_proxy_scaler = StandardScaler()
    outcome_proxy_s = outcome_proxy_scaler.fit_transform(data.outcome_proxy)

    backdoor_s = None
    backdoor_scaler = None
    if data.backdoor is not None:
        backdoor_scaler = StandardScaler()
        backdoor_s = backdoor_scaler.fit_transform(data.backdoor)

    train_data = PVTrainDataSet(treatment=treatment_s,
                                treatment_proxy=treatment_proxy_s,
                                outcome_proxy=outcome_proxy_s,
                                outcome=outcome_s,
                                backdoor=backdoor_s)

    scalers = dict(treatment_proxy_scaler=treatment_proxy_scaler,
                   treatment_scaler=treatment_scaler,
                   outcome_proxy_scaler=outcome_proxy_scaler,
                   outcome_scaler=outcome_scaler,
                   backdoor_scaler=backdoor_scaler)

    return train_data, scalers
