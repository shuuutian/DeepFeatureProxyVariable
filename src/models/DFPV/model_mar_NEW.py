from typing import Optional
import torch
from torch import nn
import numpy as np

from src.models.DFPV.model import DFPVModel
from src.utils.pytorch_linear_reg_utils import fit_linear, linear_reg_pred, add_const_col, outer_prod
from src.data.ate.data_class_mar import PVTrainDataSetMARTorch


class DFPVModelMAR:
    """Modified DFPV model for MAR outcome proxy (DR Stage-1, LS Stage-2).

    Key differences from DFPVModel:
    - Stage 1 target is φ_DR (doubly robust pseudo-outcome), not ψ_W(W)
    - θ_W (outcome_proxy_net) is optimized from Stage 2 (Stage 1 treats ψ_W target as detached)
    - Stage 2 uses μ̂(L) = V̂·(φ_A(A)⊗φ_Z(Z)⊗φ_X(X))
    - ATE prediction uses μ̂(a, Z_i, X_i) with A set to intervention value a

    Reference: Thesis Section 5.1.3, Algorithm 5.1
    """

    stage1_weight: torch.Tensor   # V̂
    stage2_weight: torch.Tensor   # û
    mean_mu_hat: torch.Tensor
    mean_joint_mu_hat_x: Optional[torch.Tensor]  # M̂ = (1/n) Σ μ̂_i ⊗ φ_X(X_i)

    def __init__(
        self,
        treatment_1st_net: nn.Module,
        treatment_2nd_net: nn.Module,
        treatment_proxy_net: nn.Module,
        outcome_proxy_net: nn.Module,
        backdoor_1st_net: Optional[nn.Module],
        backdoor_2nd_net: Optional[nn.Module],
        add_stage1_intercept: bool,
        add_stage2_intercept: bool,
    ):
        self.treatment_1st_net = treatment_1st_net
        self.treatment_2nd_net = treatment_2nd_net
        self.treatment_proxy_net = treatment_proxy_net
        self.outcome_proxy_net = outcome_proxy_net
        self.backdoor_1st_net = backdoor_1st_net
        self.backdoor_2nd_net = backdoor_2nd_net
        self.add_stage1_intercept = add_stage1_intercept
        self.add_stage2_intercept = add_stage2_intercept

    @staticmethod
    def fit_2sls_mar(
        # Stage-1 features for fitting V̂
        treatment_1st_feature: torch.Tensor,       # φ_{A(1)}(A)
        treatment_proxy_feature: torch.Tensor,     # φ_Z(Z)
        backdoor_1st_feature: Optional[torch.Tensor],  # φ_{X(1)}(X)
        phi_dr: torch.Tensor,                      # DR pseudo-outcome target (n, d_W)
        # Stage-2 features for fitting û
        treatment_2nd_feature: torch.Tensor,       # ψ_{A(2)}(A)
        backdoor_2nd_feature: Optional[torch.Tensor],  # φ_{X(2)}(X)
        outcome: torch.Tensor,                     # Y
        lam1: float,
        lam2: float,
        add_stage1_intercept: bool,
        add_stage2_intercept: bool,
    ) -> dict:
        """Two-stage least squares for modified DFPV.

        Stage 1: Regress φ_DR on (A,Z,X) features → solve for V̂
        Stage 2: Regress Y on (A, μ̂(L), X) where μ̂(L)=V̂·Φ(A,Z,X) → solve for û
        """
        # Stage 1: fit V̂ by regressing phi_DR on (A,Z,X) features
        stage1_feature = DFPVModel.augment_stage1_feature(
            treatment_feature=treatment_1st_feature,
            treatment_proxy_feature=treatment_proxy_feature,
            backdoor_feature=backdoor_1st_feature,
            add_stage1_intercept=add_stage1_intercept,
        )
        # Capture column stats BEFORE normalisation so diagnostics reflect the raw feature scale.
        col_stds = stage1_feature.std(dim=0)    # (d_feat,) — per-column std
        col_means = stage1_feature.mean(dim=0)  # (d_feat,) — per-column mean
        # Column-wise normalisation: prevents small-scale features from inflating stage1_weight.
        # Columns with near-zero std (e.g. intercept-product columns) are left unchanged.
        feat_std = col_stds.unsqueeze(0)  # reuse already computed stds
        feat_std = torch.where(feat_std < 1e-6, torch.ones_like(feat_std), feat_std)
        stage1_feature_norm = stage1_feature / feat_std
        stage1_weight = fit_linear(phi_dr, stage1_feature_norm, lam1)
        mu_hat = linear_reg_pred(stage1_feature_norm, stage1_weight)

        # Stage 2: regress Y on (ψ_{A(2)}(A) ⊗ μ̂(L) ⊗ φ_{X(2)}(X))
        stage2_feature = DFPVModel.augment_stage2_feature(
            predicted_outcome_proxy_feature=mu_hat,
            treatment_feature=treatment_2nd_feature,
            backdoor_feature=backdoor_2nd_feature,
            add_stage2_intercept=add_stage2_intercept,
        )
        stage2_weight = fit_linear(outcome, stage2_feature, lam2)
        pred = linear_reg_pred(stage2_feature, stage2_weight)
        stage2_loss = torch.norm(outcome - pred) ** 2 + lam2 * torch.norm(stage2_weight) ** 2

        return dict(
            stage1_weight=stage1_weight,
            mu_hat=mu_hat,
            stage2_weight=stage2_weight,
            stage2_loss=stage2_loss,
            pred=pred,
            col_stds=col_stds,
            col_means=col_means,
        )

    def fit_t(
        self,
        train_data_t: PVTrainDataSetMARTorch,
        phi_dr: torch.Tensor,
        lam1: float,
        lam2: float,
    ):
        """Final model fitting using full training data and precomputed φ_DR.

        Args:
            train_data_t: full training data (all samples)
            phi_dr: (n, d_W) DR pseudo-outcome precomputed for all samples
            lam1, lam2: ridge penalties
        """
        with torch.no_grad():
            treatment_1st_feature = self.treatment_1st_net(train_data_t.treatment)
            treatment_proxy_feature = self.treatment_proxy_net(train_data_t.treatment_proxy)
            treatment_2nd_feature = self.treatment_2nd_net(train_data_t.treatment)

            backdoor_1st_feature = None
            backdoor_2nd_feature = None
            if self.backdoor_1st_net is not None:
                backdoor_1st_feature = self.backdoor_1st_net(train_data_t.backdoor)
                backdoor_2nd_feature = self.backdoor_2nd_net(train_data_t.backdoor)

        res = self.fit_2sls_mar(
            treatment_1st_feature=treatment_1st_feature,
            treatment_proxy_feature=treatment_proxy_feature,
            backdoor_1st_feature=backdoor_1st_feature,
            phi_dr=phi_dr,
            treatment_2nd_feature=treatment_2nd_feature,
            backdoor_2nd_feature=backdoor_2nd_feature,
            outcome=train_data_t.outcome,
            lam1=lam1,
            lam2=lam2,
            add_stage1_intercept=self.add_stage1_intercept,
            add_stage2_intercept=self.add_stage2_intercept,
        )

        self.stage1_weight = res["stage1_weight"]
        self.stage2_weight = res["stage2_weight"]
        mu_hat = res["mu_hat"]
        self.diag_stage2_pred = res["pred"]

        # Diagnostic attributes: stage1 feature column stats and mu_hat per-sample stats
        col_stds_np = res["col_stds"].detach().cpu().numpy()
        col_means_np = res["col_means"].detach().cpu().numpy()
        # Exclude zero-std columns (intercept-product columns are always std=0 due to
        # the Kronecker-product structure with add_const_col).  Reporting their min would
        # always show 0 and obscure the true scale of the trainable feature columns.
        nz_mask = col_stds_np > 1e-6
        col_stds_nz = col_stds_np[nz_mask]
        self.diag_s1feat_n_zero_cols = int((~nz_mask).sum())
        self.diag_s1feat_col_std_min = float(col_stds_nz.min()) if len(col_stds_nz) else 0.0
        self.diag_s1feat_col_std_mean = float(col_stds_nz.mean()) if len(col_stds_nz) else 0.0
        self.diag_s1feat_col_std_max = float(col_stds_nz.max()) if len(col_stds_nz) else 0.0
        self.diag_s1feat_col_mean_min = float(col_means_np.min())
        self.diag_s1feat_col_mean_max = float(col_means_np.max())
        mu_hat_np = mu_hat.detach().cpu().numpy()
        self.diag_mu_hat_std = float(mu_hat_np.std())
        self.diag_mu_hat_min = float(mu_hat_np.min())
        self.diag_mu_hat_max = float(mu_hat_np.max())

        # Store mean(μ̂) for the no-backdoor fallback path.
        self.mean_mu_hat = mu_hat.mean(dim=0, keepdim=True)  # (1, d_W)

        # Store the joint mean M̂ = (1/n) Σ_i μ̂_i ⊗ φ_X(X_i) instead of the
        # product of separate means; μ̂_i and φ_X(X_i) are generally correlated.
        # Intercepts are incorporated into each component before the Kronecker product
        # to match exactly how Stage-2 training features were constructed.
        self.mean_joint_mu_hat_x = None
        if backdoor_2nd_feature is not None:
            mu_hat_aug = add_const_col(mu_hat) if self.add_stage2_intercept else mu_hat
            bdoor_aug = add_const_col(backdoor_2nd_feature) if self.add_stage2_intercept else backdoor_2nd_feature
            joint = torch.flatten(outer_prod(mu_hat_aug, bdoor_aug), start_dim=1)
            self.mean_joint_mu_hat_x = joint.mean(dim=0, keepdim=True)  # (1, d_W[+1]*d_X[+1])

    def predict_t(
        self,
        treatment: torch.Tensor,
    ) -> torch.Tensor:
        """Predict counterfactual mean β̂(a) for each treatment value in batch.

        Mirrors original DFPVModel.predict_t exactly: the "outcome proxy" slot is
        filled with the fixed empirical mean of μ̂(A_i, Z_i, X_i) over training data
        (stored as self.mean_mu_hat), NOT re-evaluated at the test treatment value a.

        Re-evaluating Stage-1 with a_test would inject a spurious extra dependence
        on a through φ_A(a_test), creating a different function class from what
        Stage-2 was trained to invert and producing a monotone extrapolation artefact.

        Args:
            treatment: (n_test, d_a) — intervention values a to evaluate
        """
        n_test = treatment.shape[0]
        treatment_feature = self.treatment_2nd_net(treatment)  # ψ_A(a)

        if self.mean_joint_mu_hat_x is not None:
            # Prediction: ψ_A_aug(a) ⊗ M̂
            # M̂ = (1/n) Σ_i μ̂_aug,i ⊗ φ_X_aug(X_i) already incorporates intercepts.
            if self.add_stage2_intercept:
                treatment_feature = add_const_col(treatment_feature)
            mean_joint_mat = self.mean_joint_mu_hat_x.expand(n_test, -1)
            feature = torch.flatten(outer_prod(treatment_feature, mean_joint_mat), start_dim=1)
        else:
            mean_mu_hat_mat = self.mean_mu_hat.expand(n_test, -1)
            feature = DFPVModel.augment_stage2_feature(
                predicted_outcome_proxy_feature=mean_mu_hat_mat,
                treatment_feature=treatment_feature,
                backdoor_feature=None,
                add_stage2_intercept=self.add_stage2_intercept,
            )
        return linear_reg_pred(feature, self.stage2_weight)

    def predict(self, treatment: np.ndarray) -> np.ndarray:
        treatment_t = torch.tensor(treatment, dtype=torch.float32)
        with torch.no_grad():
            return self.predict_t(treatment_t).detach().numpy()
