from typing import Optional
import torch
from torch import nn
import numpy as np

from src.models.DFPV.model import DFPVModel
from src.utils.pytorch_linear_reg_utils import fit_linear, linear_reg_pred, add_const_col
from src.data.ate.data_class_mar import PVTrainDataSetMARTorch


class DFPVModelMAR:
    """Modified DFPV model for MAR outcome proxy (DR Stage-1, LS Stage-2).

    Key differences from DFPVModel:
    - Stage 1 target is φ_DR (doubly robust pseudo-outcome), not ψ_W(W)
    - θ_W (outcome_proxy_net) is NEVER updated — frozen at random init throughout
    - Stage 2 uses μ̂(L) = V̂·(φ_A(A)⊗φ_Z(Z)⊗φ_X(X)) instead of ψ_W(W)
    - ATE prediction uses μ̂(a, Z_i, X_i) with A set to intervention value a

    Reference: Thesis Section 5.1.3, Algorithm 5.1
    """

    stage1_weight: torch.Tensor   # V̂
    stage2_weight: torch.Tensor   # û
    mean_backdoor_feature: Optional[torch.Tensor]

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
        self.outcome_proxy_net = outcome_proxy_net   # frozen — never updated
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
        Stage 2: Regress Y on (A, μ̂(L), X) features → solve for û

        phi_dr serves as both Stage-1 target AND as μ̂(L) input to Stage 2.
        This is valid because phi_dr already is the best estimate of μ(L) = E[ψ_W(W)|L].
        """
        # Stage 1: fit V̂ by regressing phi_DR on (A,Z,X) features
        stage1_feature = DFPVModel.augment_stage1_feature(
            treatment_feature=treatment_1st_feature,
            treatment_proxy_feature=treatment_proxy_feature,
            backdoor_feature=backdoor_1st_feature,
            add_stage1_intercept=add_stage1_intercept,
        )
        stage1_weight = fit_linear(phi_dr, stage1_feature, lam1)

        # Compute μ̂(L) = V̂ · stage1_feature  (thesis Eq. mu_hat_L)
        mu_hat = linear_reg_pred(stage1_feature, stage1_weight)  # (n, d_W)

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
            stage2_weight=stage2_weight,
            stage2_loss=stage2_loss,
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

        self.mean_backdoor_feature = None
        if backdoor_2nd_feature is not None:
            self.mean_backdoor_feature = torch.mean(backdoor_2nd_feature, dim=0, keepdim=True)

    def predict_t(
        self,
        treatment: torch.Tensor,
        train_treatment_proxy: torch.Tensor,
        train_backdoor: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Predict counterfactual mean β̂(a) for each treatment value in batch.

        Implements thesis Eq. beta_plugin:
            β̂(a) = (1/n) Σ_i û^T( ψ_{A(2)}(a) ⊗ μ̂(a, Z_i, X_i) ⊗ φ_{X(2)}(X_i) )

        Note: A is set to the intervention value `a` both in Stage-1 (inside μ̂)
        and in Stage-2 (ψ_{A(2)}(a)). This is the counterfactual evaluation.

        Args:
            treatment:            (n_test, d_a)  — intervention values a to evaluate
            train_treatment_proxy:(n_train, d_z)  — Z from TRAINING data (marginalise over)
            train_backdoor:       (n_train, d_x) or None — X from TRAINING data

        For each test treatment a_j, β̂(a_j) is the average over all training (Z_i, X_i).
        """
        n_test = treatment.shape[0]
        n_train = train_treatment_proxy.shape[0]

        preds = []
        for j in range(n_test):
            a_j = treatment[j:j+1].expand(n_train, -1)  # (n_train, d_a)

            # Stage-1 features with A=a_j (counterfactual)
            a_j_1st = self.treatment_1st_net(a_j)
            z_feat = self.treatment_proxy_net(train_treatment_proxy)
            x_1st_feat = None
            x_2nd_feat = None
            if self.backdoor_1st_net is not None:
                x_1st_feat = self.backdoor_1st_net(train_backdoor)
                x_2nd_feat = self.backdoor_2nd_net(train_backdoor)

            s1_feature = DFPVModel.augment_stage1_feature(
                treatment_feature=a_j_1st,
                treatment_proxy_feature=z_feat,
                backdoor_feature=x_1st_feat,
                add_stage1_intercept=self.add_stage1_intercept,
            )
            mu_hat = linear_reg_pred(s1_feature, self.stage1_weight)  # (n_train, d_W)

            # Stage-2 features with A=a_j
            a_j_2nd = self.treatment_2nd_net(a_j)
            s2_feature = DFPVModel.augment_stage2_feature(
                predicted_outcome_proxy_feature=mu_hat,
                treatment_feature=a_j_2nd,
                backdoor_feature=x_2nd_feat,
                add_stage2_intercept=self.add_stage2_intercept,
            )
            # Average over training samples → scalar β̂(a_j)
            mean_feature = s2_feature.mean(dim=0, keepdim=True)  # (1, feature_dim)
            pred_j = linear_reg_pred(mean_feature, self.stage2_weight)  # (1, 1) or (1,)
            preds.append(pred_j)

        return torch.cat(preds, dim=0)  # (n_test, 1) or (n_test,)

    def predict(
        self,
        treatment: np.ndarray,
        train_treatment_proxy: np.ndarray,
        train_backdoor: Optional[np.ndarray],
    ) -> np.ndarray:
        treatment_t = torch.tensor(treatment, dtype=torch.float32)
        proxy_t = torch.tensor(train_treatment_proxy, dtype=torch.float32)
        backdoor_t = None
        if train_backdoor is not None:
            backdoor_t = torch.tensor(train_backdoor, dtype=torch.float32)
        with torch.no_grad():
            return self.predict_t(treatment_t, proxy_t, backdoor_t).detach().numpy()
