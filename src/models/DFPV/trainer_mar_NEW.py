from __future__ import annotations
from typing import Dict, Any, Optional, List
from pathlib import Path
import csv
import logging

import numpy as np
import torch
from torch import nn

from src.models.DFPV.nn_structure import build_extractor
from src.models.DFPV.model_mar_NEW import DFPVModelMAR
from src.models.DFPV.nuisance_models import NuisanceModels
from src.models.DFPV.dr_utils import (
    construct_L_plus,
    compute_dr_pseudo_outcome,
    compute_effective_sample_size,
)
from src.data.ate.data_class_mar import (
    PVTrainDataSetMAR,
    PVTrainDataSetMARTorch,
    create_k_folds,
    get_train_val_split,
)
from src.utils.pytorch_linear_reg_utils import (
    fit_linear,
    linear_reg_pred,
    linear_reg_loss,
)

logger = logging.getLogger()


class DFPVTrainerMAR:
    """Trainer for Modified DFPV under MAR with K-fold cross-fitting.

    Algorithm 5.1 (thesis):
        For each epoch t:
            For each fold k:
                1. Fit nuisance models on folds I_{-k}
                2. Compute φ_DR on fold I_k
                3. Stage-1 update on I_k
                4. Stage-2 update on I_k
        Final fit on all data with final nuisance models.

    Key properties:
    - outcome_proxy_net (θ_W) is pre-trained and updated only in Stage 2
    - Nuisance imputation trained on L=(A,Z,X); propensity on L+=(A,Z,X,Y)
    """

    def __init__(
        self,
        data_configs: Dict[str, Any],
        train_params: Dict[str, Any],
        gpu_flg: bool = False,
        random_seed: int = 42,
    ):
        self.data_config = data_configs
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        self.random_seed = random_seed

        self.lam1: float = train_params["lam1"]
        self.lam2: float = train_params["lam2"]
        self.stage1_iter: int = train_params["stage1_iter"]
        self.stage2_iter: int = train_params["stage2_iter"]
        self.n_epoch: int = train_params["n_epoch"]
        self.n_folds: int = train_params.get("n_folds", 5)
        self.nuisance_n_epochs: int = train_params.get("nuisance_n_epochs", 50)
        self.propensity_clip: float = train_params.get("propensity_clip", 1e-3)
        self.outcome_proxy_pretrain_epochs: int = train_params.get("outcome_proxy_pretrain_epochs", 50)
        self.outcome_proxy_pretrain_lr: float = train_params.get("outcome_proxy_pretrain_lr", 1e-3)
        self.outcome_proxy_weight_decay = train_params["outcome_proxy_weight_decay"]
        self.add_stage1_intercept = True
        self.add_stage2_intercept = True
        self.treatment_weight_decay = train_params["treatment_weight_decay"]
        self.treatment_proxy_weight_decay = train_params["treatment_proxy_weight_decay"]
        self.backdoor_weight_decay = train_params["backdoor_weight_decay"]

        # Build networks — outcome_proxy_net is pre-trained, then optimized in MAR loop.
        networks = build_extractor(data_configs["name"])
        self.treatment_1st_net: nn.Module = networks[0]
        self.treatment_2nd_net: nn.Module = networks[1]
        self.treatment_proxy_net: nn.Module = networks[2]
        self.outcome_proxy_net: nn.Module = networks[3]  # θ_W
        self.backdoor_1st_net: Optional[nn.Module] = networks[4]
        self.backdoor_2nd_net: Optional[nn.Module] = networks[5]

        if self.gpu_flg:
            self.treatment_1st_net.to("cuda:0")
            self.treatment_2nd_net.to("cuda:0")
            self.treatment_proxy_net.to("cuda:0")
            self.outcome_proxy_net.to("cuda:0")
            if self.backdoor_1st_net is not None:
                self.backdoor_1st_net.to("cuda:0")
                self.backdoor_2nd_net.to("cuda:0")

        self.treatment_1st_opt = torch.optim.Adam(
            self.treatment_1st_net.parameters(), weight_decay=self.treatment_weight_decay
        )
        self.treatment_2nd_opt = torch.optim.Adam(
            self.treatment_2nd_net.parameters(), weight_decay=self.treatment_weight_decay
        )
        self.treatment_proxy_opt = torch.optim.Adam(
            self.treatment_proxy_net.parameters(), weight_decay=self.treatment_proxy_weight_decay
        )
        self.outcome_proxy_opt = torch.optim.Adam(
            self.outcome_proxy_net.parameters(), weight_decay=self.outcome_proxy_weight_decay
        )
        if self.backdoor_1st_net is not None:
            self.backdoor_1st_opt = torch.optim.Adam(
                self.backdoor_1st_net.parameters(), weight_decay=self.backdoor_weight_decay
            )
            self.backdoor_2nd_opt = torch.optim.Adam(
                self.backdoor_2nd_net.parameters(), weight_decay=self.backdoor_weight_decay
            )

        # Nuisance models — initialized lazily in train() once data dims are known
        self.nuisance_models: Optional[NuisanceModels] = None

        # Training history for nuisance models; populated by _fit_nuisance_models.
        # Each entry: {"label": str, "propensity": [float], "imputation": [float]}
        self.nuisance_history: list = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(
        self,
        train_data: PVTrainDataSetMAR,
        verbose: int = 0,
        diagnostic_save_dir: Optional[Path] = None,
    ) -> DFPVModelMAR:
        """Run full training loop and return fitted DFPVModelMAR.

        Args:
            train_data: MAR training dataset.
            verbose: 0=silent, 1=epoch-level, 2=step-level console logging.
            diagnostic_save_dir: If set, save full e_hat/m_hat (and summary) to CSV and
                a log file here. Safe for parallel runs when each run uses a unique path
                (e.g. one_mdl_dump_dir / str(random_seed)).
        """
        train_data_t = PVTrainDataSetMARTorch.from_numpy(train_data)
        if self.gpu_flg:
            train_data_t = train_data_t.to_gpu()

        self._pretrain_outcome_proxy(train_data_t, verbose)

        # Initialise nuisance models now that we know input dimensions
        self.nuisance_models = self._init_nuisance_models(train_data_t)

        fold_indices = create_k_folds(train_data_t, self.n_folds, seed=self.random_seed)

        for t in range(self.n_epoch):
            for k in range(self.n_folds):
                train_fold, val_fold = get_train_val_split(train_data_t, fold_indices, k)

                # Step 1: Fit nuisance on training folds I_{-k}
                self._fit_nuisance_models(train_fold)

                # Step 2: Compute nuisance predictions on I_k for DR pseudo-outcome
                m_hat, e_hat = self._compute_dr_nuisance_predictions(val_fold, verbose)

                # Step 3 & 4: Update Stage-1 and Stage-2 on validation fold I_k
                self.stage1_update_mar(val_fold, m_hat, e_hat, verbose)
                self.stage2_update_mar(val_fold, m_hat, e_hat, verbose)

            if verbose >= 1:
                logger.info(f"Epoch {t} ended")

        # Final cross-fitted phi_DR — satisfies B2: for each i in I_k, nuisance
        # is fitted only on I_{-k}, so phi_DR_i is evaluated out-of-sample.
        # This replaces the previous approach of fitting nuisance on ALL data then
        # evaluating on the same ALL data, which violated the cross-fitting assumption.

        # Cross-fitted φ_DR and nuisance predictions for final fit and diagnostics
        phi_dr_full, m_hat_full, e_hat_full = self._compute_crossfitted_phi_dr(
            train_data_t, fold_indices, verbose
        )

        # Store e_hat and m_hat for diagnostics (e.g. propensity/imputation checks)
        self.diagnostic_e_hat = e_hat_full.detach()
        self.diagnostic_m_hat = m_hat_full.detach()

        # Summary stats for log file and optional console
        e_np = self.diagnostic_e_hat.cpu().numpy().flatten()
        m_np = self.diagnostic_m_hat.cpu().numpy()
        delta_np = train_data_t.delta_w.cpu().numpy().flatten()
        ess = compute_effective_sample_size(
            self.diagnostic_e_hat.squeeze(1),
            train_data_t.delta_w,
            self.propensity_clip,
        )
        summary_lines = [
            "Diagnostic e_hat: mean=%.4f std=%.4f min=%.4f max=%.4f ESS=%.1f"
            % (float(e_np.mean()), float(e_np.std()), float(e_np.min()), float(e_np.max()), ess),
            "Diagnostic m_hat: mean=%.4f std=%.4f shape=%s"
            % (float(m_np.mean()), float(m_np.std()), m_np.shape),
        ]

        # Build model and fit so we can log post-fit diagnostics (mean_mu_hat, stage weights)
        mdl = DFPVModelMAR(
            self.treatment_1st_net,
            self.treatment_2nd_net,
            self.treatment_proxy_net,
            self.outcome_proxy_net,
            self.backdoor_1st_net,
            self.backdoor_2nd_net,
            self.add_stage1_intercept,
            self.add_stage2_intercept,
        )
        mdl.fit_t(train_data_t, phi_dr_full, self.lam1, self.lam2)

        # Compute psi_w = ψ_W(W) for observed cases (target for m_hat)
        obs_mask = delta_np == 1
        obs_mask_t = torch.from_numpy(obs_mask).to(device=train_data_t.outcome_proxy.device)
        with torch.no_grad():
            psi_w_obs = self.outcome_proxy_net(
                train_data_t.outcome_proxy[obs_mask_t]
            ).cpu().numpy()  # (n_obs, d_W)
        stage2_pred_np = mdl.diag_stage2_pred.cpu().numpy().flatten()
        Y_np = train_data_t.outcome.cpu().numpy().flatten()
        stage2_resid_np = Y_np - stage2_pred_np

        # Post-fit diagnostics: phi_DR scale, mean_mu_hat, stage1/stage2 weight norms
        phi_np = phi_dr_full.cpu().numpy()
        summary_lines.append(
            "Diagnostic phi_DR: mean=%.4f std=%.4f min=%.4f max=%.4f max_abs=%.4f"
            % (
                float(phi_np.mean()),
                float(phi_np.std()),
                float(phi_np.min()),
                float(phi_np.max()),
                float(np.abs(phi_np).max()),
            )
        )
        # Diagnostic 3: IPW correction = phi_DR - m_hat (= (delta/e)*( psi_w - m_hat ))
        # For delta=0 this is identically 0; for delta=1 it equals (1/e)*(psi_w - m_hat).
        ipw_corr_np = phi_np - m_np  # (n, d_W)
        summary_lines.append(
            "Diagnostic ipw_correction (phi_DR-m_hat): mean=%.4f std=%.4f max_abs=%.4f"
            % (float(ipw_corr_np.mean()), float(ipw_corr_np.std()), float(np.abs(ipw_corr_np).max()))
        )
        # Diagnostic 2: psi_w - m_hat for observed cases = e_hat * ipw_correction (delta=1 only)
        psi_w_resid_np = e_np[obs_mask, np.newaxis] * ipw_corr_np[obs_mask]  # (n_obs, d_W)
        summary_lines.append(
            "Diagnostic psi_w_resid (delta=1): mean=%.4f std=%.4f max_abs=%.4f n_obs=%d"
            % (
                float(psi_w_resid_np.mean()),
                float(psi_w_resid_np.std()),
                float(np.abs(psi_w_resid_np).max()),
                int(obs_mask.sum()),
            )
        )
        # Stage1 mean comparison: E[phi_DR] vs E[psi_W(W)|delta=1]
        phi_dr_mean = phi_np.mean(axis=0)
        psi_w_mean = psi_w_obs.mean(axis=0)
        summary_lines.append(
            "Diagnostic Stage1 mean: phi_DR_mean=%.4f psi_w_obs_mean=%.4f diff=%.4f"
            % (float(phi_dr_mean.mean()), float(psi_w_mean.mean()), float((phi_dr_mean - psi_w_mean).mean()))
        )
        mmu = mdl.mean_mu_hat.cpu().numpy().flatten()
        summary_lines.append(
            "Diagnostic mean_mu_hat: mean=%.4f std=%.4f min=%.4f max=%.4f"
            % (float(mmu.mean()), float(mmu.std()), float(mmu.min()), float(mmu.max()))
        )
        w1 = mdl.stage1_weight.detach().cpu().numpy()
        w2 = mdl.stage2_weight.detach().cpu().numpy()
        summary_lines.append(
            "Diagnostic stage1_weight: frobenius_norm=%.4f"
            % (float(np.sqrt(np.sum(w1 ** 2))),)
        )
        summary_lines.append(
            "Diagnostic stage2_weight: l2_norm=%.4f"
            % (float(np.sqrt(np.sum(w2 ** 2))),)
        )
        # Blackbox 1: stage1_feature column stats — explains why stage1_weight can be large
        # min/mean/max are computed over non-zero-std columns only (zero-std = intercept products)
        summary_lines.append(
            "Diagnostic stage1_feat col_std (non-zero): min=%.4f mean=%.4f max=%.4f n_zero=%d | col_mean: min=%.4f max=%.4f"
            % (
                mdl.diag_s1feat_col_std_min, mdl.diag_s1feat_col_std_mean, mdl.diag_s1feat_col_std_max,
                mdl.diag_s1feat_n_zero_cols,
                mdl.diag_s1feat_col_mean_min, mdl.diag_s1feat_col_mean_max,
            )
        )
        # Blackbox 2: mu_hat per-sample distribution (not just its mean across d_W)
        summary_lines.append(
            "Diagnostic mu_hat per-sample: std=%.4f min=%.4f max=%.4f"
            % (mdl.diag_mu_hat_std, mdl.diag_mu_hat_min, mdl.diag_mu_hat_max)
        )
        # Blackbox 3: upper-bound on prediction magnitude = ||mean_mu_hat|| x ||stage2_weight||
        mmu_norm = float(np.sqrt(np.sum(mmu ** 2)))
        w2_norm = float(np.sqrt(np.sum(w2 ** 2)))
        summary_lines.append(
            "Diagnostic pred_scale_bound: ||mean_mu_hat||=%.4f x ||stage2_w||=%.4f => %.4f"
            % (mmu_norm, w2_norm, mmu_norm * w2_norm)
        )

        if diagnostic_save_dir is not None:
            diagnostic_save_dir = Path(diagnostic_save_dir)
            diagnostic_save_dir.mkdir(parents=True, exist_ok=True)
            # Per-sample CSV: Stage1 (e_hat, m_hat, psi_w, diffs) and Stage2 (resid)
            d_w = phi_dr_full.shape[1]
            e_diff_np = delta_np - e_np
            # psi_w_all: (n, d_W) — NaN for delta_w=0
            psi_w_all = np.full((len(delta_np), d_w), np.nan)
            psi_w_all[obs_mask] = psi_w_obs
            # m_abs_diff: (n,) — sum_j |psi_w_j - m_hat_j|, NaN for delta_w=0
            m_abs_diff_np = np.full(len(delta_np), np.nan)
            m_abs_diff_np[obs_mask] = np.abs(psi_w_obs - m_np[obs_mask]).sum(axis=1)
            csv_path = diagnostic_save_dir / "diagnostic_nuisance.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                header = (
                    ["sample_idx", "delta_w", "e_hat", "e_diff"]
                    + [f"m_hat_{j}" for j in range(d_w)]
                    + [f"psi_w_{j}" for j in range(d_w)]
                    + ["m_abs_diff", "stage2_resid"]
                )
                writer.writerow(header)
                for i in range(len(delta_np)):
                    row = [i, float(delta_np[i]), float(e_np[i]), float(e_diff_np[i])]
                    row.extend(float(x) for x in m_np[i])
                    row.extend(
                        "" if np.isnan(psi_w_all[i, j]) else float(psi_w_all[i, j])
                        for j in range(d_w)
                    )
                    row.append("" if np.isnan(m_abs_diff_np[i]) else float(m_abs_diff_np[i]))
                    row.append(float(stage2_resid_np[i]))
                    writer.writerow(row)
            # Summary log file (no console dependency; safe in parallel)
            log_path = diagnostic_save_dir / "diagnostic_summary.log"
            with log_path.open("w") as f:
                f.write("\n".join(summary_lines) + "\n")
            if verbose >= 1:
                logger.info("Diagnostics written to %s and %s", csv_path, log_path)

        if verbose >= 1:
            for line in summary_lines:
                logger.info(line)

        return mdl

    def _pretrain_outcome_proxy(
        self,
        data: PVTrainDataSetMARTorch,
        verbose: int = 0,
    ):
        """Pre-train θ_W on complete cases before MAR stage updates.

        Auxiliary objective: predict Y from [ψ_W(W), A, X] on rows with delta_w=1.
        This improves the starting representation used by nuisance imputation and DR
        targets before joint end-to-end optimization.
        """
        if self.outcome_proxy_pretrain_epochs <= 0:
            return

        if data.delta_w.dim() > 1:
            obs_mask = data.delta_w.squeeze(1).bool()
        else:
            obs_mask = data.delta_w.bool()
        n_obs = int(obs_mask.sum().item())
        if n_obs == 0:
            return

        device = next(self.outcome_proxy_net.parameters()).device
        y_obs = data.outcome[obs_mask]
        a_obs = data.treatment[obs_mask]
        x_obs = data.backdoor[obs_mask] if data.backdoor is not None else None

        with torch.no_grad():
            psi_dim = self.outcome_proxy_net(data.outcome_proxy[obs_mask][:1]).shape[1]
        head_in_dim = psi_dim + a_obs.shape[1] + (x_obs.shape[1] if x_obs is not None else 0)
        head = nn.Linear(head_in_dim, y_obs.shape[1]).to(device)

        params = list(self.outcome_proxy_net.parameters()) + list(head.parameters())
        pretrain_opt = torch.optim.Adam(params, lr=self.outcome_proxy_pretrain_lr)
        mse = nn.MSELoss()
        self.outcome_proxy_net.train(True)
        head.train(True)

        for _ in range(self.outcome_proxy_pretrain_epochs):
            pretrain_opt.zero_grad()
            psi_obs = self.outcome_proxy_net(data.outcome_proxy[obs_mask])
            feat_parts = [psi_obs, a_obs]
            if x_obs is not None:
                feat_parts.append(x_obs)
            feat = torch.cat(feat_parts, dim=1)
            y_hat = head(feat)
            loss = mse(y_hat, y_obs)
            loss.backward()
            pretrain_opt.step()

        if verbose >= 1:
            logger.info(
                "outcome_proxy pretrain complete: epochs=%d, observed=%d, final_loss=%.6f",
                self.outcome_proxy_pretrain_epochs,
                n_obs,
                float(loss.detach().item()),
            )

    # ------------------------------------------------------------------
    # Nuisance fitting and DR pseudo-outcome
    # ------------------------------------------------------------------

    def _init_nuisance_models(self, data: PVTrainDataSetMARTorch) -> NuisanceModels:
        """Infer input dimensions from data and build nuisance models."""
        with torch.no_grad():
            psi_w_sample = self.outcome_proxy_net(data.outcome_proxy[:1])
        dim_psi_w = psi_w_sample.shape[1]

        # L = (A, Z, X)
        dim_L = data.treatment.shape[1] + data.treatment_proxy.shape[1]
        if data.backdoor is not None:
            dim_L += data.backdoor.shape[1]

        # L+ = (A, Z, X, Y)
        outcome_dim = data.outcome.shape[1] if data.outcome.dim() > 1 else 1
        dim_L_plus = dim_L + outcome_dim

        return NuisanceModels(
            dim_L_plus=dim_L_plus,
            dim_L=dim_L,
            dim_psi_w=dim_psi_w,
            gpu_flg=self.gpu_flg,
        )

    def _fit_nuisance_models(self, train_fold: PVTrainDataSetMARTorch, reset: bool = True):
        """Fit propensity and imputation on the given data split.

        Propensity: trained on ALL samples in fold using L+=(A,Z,X,Y)
        Imputation: trained on complete cases (delta_w=1) using L=(A,Z,X)
                    with targets psi_{θ_W}(W_i) from current outcome_proxy_net

        Args:
            train_fold: data to fit on (I_{-k} during epoch loop, full data for final fit)
            reset:      if True, re-initialise nuisance networks before fitting.
                        Must be True during the epoch loop (Algorithm 5.1: each fold is an
                        independent fit on I_{-k}).  Set to False for the final all-data fit
                        so that the nuisance networks warm-start from their last state.
        """
        if reset:
            self.nuisance_models.reset_weights()

        L_plus = construct_L_plus(
            train_fold.treatment,
            train_fold.treatment_proxy,
            train_fold.outcome,
            train_fold.backdoor,
        )

        # Extract ψ_{θ_W}(W_i) with current θ_W for complete cases.
        with torch.no_grad():
            psi_w = self.outcome_proxy_net(train_fold.outcome_proxy)

        prop_hist = self.nuisance_models.fit_propensity(
            L_plus=L_plus,
            delta_w=train_fold.delta_w,
            n_epochs=self.nuisance_n_epochs,
        )
        imp_hist = self.nuisance_models.fit_imputation(
            L_plus=L_plus,
            psi_w_targets=psi_w,
            delta_w=train_fold.delta_w,
            n_epochs=self.nuisance_n_epochs,
        )
        label = "final" if not reset else f"fold"
        self.nuisance_history.append({
            "label": label,
            "propensity": prop_hist,
            "imputation": imp_hist,
        })

    def _compute_dr_nuisance_predictions(
        self,
        fold: PVTrainDataSetMARTorch,
        verbose: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute nuisance predictions (m_hat, e_hat) on a fold."""
        L_plus = construct_L_plus(fold.treatment, fold.treatment_proxy, fold.outcome, fold.backdoor)

        with torch.no_grad():
            e_hat = self.nuisance_models.predict_propensity(L_plus)   # (n, 1)
            m_hat = self.nuisance_models.predict_imputation(L_plus)    # (n, d_W)

        if verbose >= 2:
            ess = compute_effective_sample_size(
                e_hat.squeeze(1), fold.delta_w, self.propensity_clip
            )
            logger.info(f"DR ESS: {ess:.1f}")

        return m_hat.detach(), e_hat.detach()

    def _compute_dr_pseudo_outcome(
        self,
        fold: PVTrainDataSetMARTorch,
        verbose: int = 0,
    ) -> torch.Tensor:
        """Compute detached φ_DR for all samples in fold using trained nuisance models."""
        m_hat, e_hat = self._compute_dr_nuisance_predictions(fold, verbose)
        with torch.no_grad():
            psi_w = self.outcome_proxy_net(fold.outcome_proxy)     # (n, d_W)

        phi_dr = compute_dr_pseudo_outcome(
            psi_w=psi_w,
            m_hat=m_hat,
            e_hat=e_hat,
            delta_w=fold.delta_w,
            propensity_clip=self.propensity_clip,
        )
        return phi_dr.detach()

    def _compute_crossfitted_phi_dr(
        self,
        data: PVTrainDataSetMARTorch,
        fold_indices: List[torch.Tensor],
        verbose: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute cross-fitted φ_DR and nuisance (m_hat, e_hat) for every observation.

        For each fold k:
          1. Re-fit nuisance from scratch on I_{-k}  (reset=True)
          2. Evaluate φ_DR, m_hat, e_hat on I_k
          3. Write results back into the full-data tensors by original index

        Returns:
            phi_dr_full: (n, d_W) cross-fitted DR pseudo-outcome
            m_hat_full:  (n, d_W) cross-fitted imputation predictions (for diagnostics)
            e_hat_full:  (n, 1) cross-fitted propensity scores (for diagnostics)
        """
        n = data.treatment.shape[0]
        device = data.treatment.device
        with torch.no_grad():
            psi_w_sample = self.outcome_proxy_net(data.outcome_proxy[:1])
        d_w = psi_w_sample.shape[1]

        phi_dr_full = torch.zeros(n, d_w, device=device)
        m_hat_full = torch.zeros(n, d_w, device=device)
        e_hat_full = torch.zeros(n, 1, device=device)

        for k in range(self.n_folds):
            train_fold, val_fold = get_train_val_split(data, fold_indices, k)
            self._fit_nuisance_models(train_fold, reset=True)
            m_hat, e_hat = self._compute_dr_nuisance_predictions(val_fold, verbose)
            with torch.no_grad():
                psi_w = self.outcome_proxy_net(val_fold.outcome_proxy)
            phi_dr_k = compute_dr_pseudo_outcome(
                psi_w=psi_w,
                m_hat=m_hat,
                e_hat=e_hat,
                delta_w=val_fold.delta_w,
                propensity_clip=self.propensity_clip,
            )
            idx = fold_indices[k]
            phi_dr_full[idx] = phi_dr_k.detach()
            m_hat_full[idx] = m_hat
            e_hat_full[idx] = e_hat

        return phi_dr_full, m_hat_full, e_hat_full

    # ------------------------------------------------------------------
    # Stage update methods
    # ------------------------------------------------------------------

    def stage1_update_mar(
        self,
        fold: PVTrainDataSetMARTorch,
        m_hat: torch.Tensor,
        e_hat: torch.Tensor,
        verbose: int,
    ):
        """Stage-1 gradient update using φ_DR as the regression target.

        Updates: treatment_1st_net, treatment_proxy_net, backdoor_1st_net
        Frozen:  treatment_2nd_net, backdoor_2nd_net, outcome_proxy_net
        Target:  φ_DR(θ_W) = m_hat + (δ/e_hat)(ψ_W(θ_W) - m_hat)
        θ_W is detached (matches original DFPV Stage-1 treatment of ψ_W target).
        """
        self.treatment_1st_net.train(True)
        self.treatment_2nd_net.train(False)
        self.treatment_proxy_net.train(True)
        self.outcome_proxy_net.train(False)
        if self.backdoor_1st_net is not None:
            self.backdoor_1st_net.train(True)
            self.backdoor_2nd_net.train(False)

        for _ in range(self.stage1_iter):
            self.treatment_1st_opt.zero_grad()
            self.treatment_proxy_opt.zero_grad()
            if self.backdoor_1st_net is not None:
                self.backdoor_1st_opt.zero_grad()

            treatment_1st_feature = self.treatment_1st_net(fold.treatment)
            treatment_proxy_feature = self.treatment_proxy_net(fold.treatment_proxy)
            backdoor_1st_feature = None
            if self.backdoor_1st_net is not None:
                backdoor_1st_feature = self.backdoor_1st_net(fold.backdoor)

            feature = DFPVTrainerMAR._augment_stage1(
                treatment_1st_feature,
                treatment_proxy_feature,
                backdoor_1st_feature,
                self.add_stage1_intercept,
            )
            with torch.no_grad():
                psi_w = self.outcome_proxy_net(fold.outcome_proxy)
            phi_dr = compute_dr_pseudo_outcome(
                psi_w=psi_w,
                m_hat=m_hat,
                e_hat=e_hat,
                delta_w=fold.delta_w,
                propensity_clip=self.propensity_clip,
            )
            loss = linear_reg_loss(phi_dr, feature, self.lam1)
            loss.backward()

            if verbose >= 2:
                logger.info(f"stage1_mar learning: {loss.item()}")

            self.treatment_1st_opt.step()
            self.treatment_proxy_opt.step()
            if self.backdoor_1st_net is not None:
                self.backdoor_1st_opt.step()

    def stage2_update_mar(
        self,
        fold: PVTrainDataSetMARTorch,
        m_hat: torch.Tensor,
        e_hat: torch.Tensor,
        verbose: int,
    ):
        """Stage-2 gradient update — identical to Original DFPV stage2_update,
        except outcome_proxy_feature is replaced by psi_tilde.

        Updates: treatment_2nd_net, backdoor_2nd_net, outcome_proxy_net (θ_W)
        Frozen:  treatment_1st_net, treatment_proxy_net, backdoor_1st_net

        θ_W gradient path (single consistent path, no IPW):
          psi_tilde = δ·ψ_W(θ_W) + (1-δ)·m_hat.detach()
          psi_tilde → V_hat → μ_hat → stage2_feature → stage2_weight → loss
        """
        self.treatment_1st_net.train(False)
        self.treatment_2nd_net.train(True)
        self.treatment_proxy_net.train(False)
        self.outcome_proxy_net.train(True)
        if self.backdoor_1st_net is not None:
            self.backdoor_1st_net.train(False)
            self.backdoor_2nd_net.train(True)

        with torch.no_grad():
            treatment_1st_feature = self.treatment_1st_net(fold.treatment)
            treatment_proxy_feature = self.treatment_proxy_net(fold.treatment_proxy)
            backdoor_1st_feature = None
            if self.backdoor_1st_net is not None:
                backdoor_1st_feature = self.backdoor_1st_net(fold.backdoor)
            stage1_feature = DFPVTrainerMAR._augment_stage1(
                treatment_1st_feature,
                treatment_proxy_feature,
                backdoor_1st_feature,
                self.add_stage1_intercept,
            )
        delta_w = fold.delta_w
        if delta_w.dim() == 1:
            delta_w = delta_w.unsqueeze(1)
        delta_w = delta_w.to(dtype=m_hat.dtype)

        for _ in range(self.stage2_iter):
            self.treatment_2nd_opt.zero_grad()
            self.outcome_proxy_opt.zero_grad()
            if self.backdoor_2nd_net is not None:
                self.backdoor_2nd_opt.zero_grad()

            # Single path — identical to Original DFPV stage2_update.
            # Only difference: psi_tilde replaces outcome_proxy_feature_1st.
            psi_w = self.outcome_proxy_net(fold.outcome_proxy)
            psi_tilde = delta_w * psi_w + (1.0 - delta_w) * m_hat.detach()
            stage1_weight = fit_linear(psi_tilde, stage1_feature, self.lam1)
            mu_hat = linear_reg_pred(stage1_feature, stage1_weight)

            treatment_2nd_feature = self.treatment_2nd_net(fold.treatment)
            backdoor_2nd_feature = None
            if self.backdoor_2nd_net is not None:
                backdoor_2nd_feature = self.backdoor_2nd_net(fold.backdoor)

            stage2_feature = DFPVTrainerMAR._augment_stage2(
                mu_hat,
                treatment_2nd_feature,
                backdoor_2nd_feature,
                self.add_stage2_intercept,
            )
            stage2_weight = fit_linear(fold.outcome, stage2_feature, self.lam2)
            pred = linear_reg_pred(stage2_feature, stage2_weight)
            loss = torch.norm(fold.outcome - pred) ** 2 + self.lam2 * torch.norm(stage2_weight) ** 2
            loss.backward()

            if verbose >= 2:
                logger.info(f"stage2_mar learning: {loss.item()}")

            self.treatment_2nd_opt.step()
            self.outcome_proxy_opt.step()
            if self.backdoor_2nd_net is not None:
                self.backdoor_2nd_opt.step()

    # ------------------------------------------------------------------
    # Nuisance diagnostics
    # ------------------------------------------------------------------

    def plot_nuisance_history(self, save_path: Optional[str] = None):
        """Plot per-epoch training loss of propensity and imputation models.

        Fold runs are shown as faint lines; the final all-data fit is highlighted
        in bold so over/under-fitting is easy to spot:
          - Loss still decreasing at the last epoch  → underfitting (increase nuisance_n_epochs)
          - Loss plateaued well before the last epoch → converged (epochs may be excessive)
          - High variance across folds               → high nuisance estimation variance

        Args:
            save_path: if given, save figure to this path instead of showing it.
        """
        import matplotlib.pyplot as plt

        fold_runs = [h for h in self.nuisance_history if h["label"] == "fold"]
        final_runs = [h for h in self.nuisance_history if h["label"] == "final"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Nuisance model training history", fontsize=13)

        for ax, key, ylabel in zip(
            axes,
            ("propensity", "imputation"),
            ("BCE loss (propensity)", "MSE loss (imputation)"),
        ):
            for h in fold_runs:
                losses = h[key]
                if losses:
                    ax.plot(losses, color="steelblue", alpha=0.25, linewidth=0.8)
            for h in final_runs:
                losses = h[key]
                if losses:
                    ax.plot(losses, color="black", linewidth=2.0, label="final fit")
                    ax.axhline(losses[-1], color="black", linestyle=":", linewidth=0.8,
                               label=f"final={losses[-1]:.4f}")

            if fold_runs:
                # Dummy line for legend
                ax.plot([], [], color="steelblue", alpha=0.6, linewidth=1.0,
                        label=f"fold runs (n={len(fold_runs)})")

            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, linewidth=0.4, alpha=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Nuisance history plot saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Static helpers (delegate to DFPVModel to avoid duplication)
    # ------------------------------------------------------------------

    @staticmethod
    def _augment_stage1(
        treatment_feature: torch.Tensor,
        treatment_proxy_feature: torch.Tensor,
        backdoor_feature: Optional[torch.Tensor],
        add_intercept: bool,
    ) -> torch.Tensor:
        from src.models.DFPV.model import DFPVModel
        return DFPVModel.augment_stage1_feature(
            treatment_feature, treatment_proxy_feature, backdoor_feature, add_intercept
        )

    @staticmethod
    def _augment_stage2(
        outcome_proxy_feature: torch.Tensor,
        treatment_feature: torch.Tensor,
        backdoor_feature: Optional[torch.Tensor],
        add_intercept: bool,
    ) -> torch.Tensor:
        from src.models.DFPV.model import DFPVModel
        return DFPVModel.augment_stage2_feature(
            outcome_proxy_feature, treatment_feature, backdoor_feature, add_intercept
        )


def dfpv_experiments_mar_modified_NEW(
    data_config: Dict[str, Any],
    model_param: Dict[str, Any],
    one_mdl_dump_dir: Path,
    random_seed: int = 42,
    verbose: int = 0,
) -> float:
    """Run a single Modified DFPV MAR experiment and return out-of-sample MSE.

    Mirrors dfpv_experiments() in trainer.py but uses MAR data generation,
    DFPVTrainerMAR, and the MAR predict signature.

    Args:
        data_config:      dataset config dict (must contain "name" and optionally "mar_threshold")
        model_param:      trainer hyperparameter dict
        one_mdl_dump_dir: directory to save predictions
        random_seed:      controls data generation, fold creation, and torch seed
        verbose:          0=silent, 1=epoch-level, 2=step-level logging
    Returns:
        oos_loss: out-of-sample MSE (or MAE for kpv/deaner) against structural function
    """
    from src.data.ate import generate_train_data_ate_mar, generate_test_data_ate
    from src.data.ate import get_preprocessor_ate

    dump_dir = one_mdl_dump_dir.joinpath(f"{random_seed}")
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    train_data_org = generate_train_data_ate_mar(data_config=data_config, rand_seed=random_seed)
    test_data_org = generate_test_data_ate(data_config=data_config)

    preprocessor = get_preprocessor_ate(data_config.get("preprocess", "Identity"))
    train_data = preprocessor.preprocess_for_train_mar(train_data_org)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)

    torch.manual_seed(random_seed)
    trainer = DFPVTrainerMAR(data_config, model_param, gpu_flg=False, random_seed=random_seed)
    mdl = trainer.train(
        train_data,
        verbose,
        diagnostic_save_dir=dump_dir,
    )


    test_treatment_t = torch.tensor(test_data.treatment, dtype=torch.float32)
    pred: np.ndarray = mdl.predict_t(test_treatment_t).detach().cpu().numpy()
    pred = preprocessor.postprocess_for_prediction(pred)

    oos_loss = 0.0
    if test_data_org is not None and test_data_org.structural is not None:
        oos_loss = float(np.mean((pred - test_data_org.structural) ** 2))
        if data_config["name"] in ["kpv", "deaner"]:
            oos_loss = float(np.mean(np.abs(pred - test_data_org.structural)))

    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)
    return oos_loss
