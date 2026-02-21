from typing import Optional
import torch
from torch import nn


class NuisanceModels:
    """Propensity score and imputation models for Modified DFPV under MAR.

    Propensity model: e(L+) = Pr(delta_W=1 | A, Z, X, Y)  -- input is L+ = (A,Z,X,Y)
    Imputation model: m_psi(L) = E[psi_W(W) | A, Z, X]    -- input is L  = (A,Z,X)

    Per Algorithm 5.1 (thesis):
    - Propensity is trained on ALL samples using logistic loss.
    - Imputation is trained on complete cases only (delta_W=1) using MSE loss,
      with targets psi_{theta_W}(W_i) extracted using the FROZEN outcome_proxy_net.
    """

    def __init__(
        self,
        dim_L_plus: int,
        dim_L: int,
        dim_psi_w: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        gpu_flg: bool = False,
    ):
        self.gpu_flg = gpu_flg

        # Store architecture args so reset_weights() can rebuild identical networks
        self._dim_L_plus = dim_L_plus
        self._dim_L = dim_L
        self._dim_psi_w = dim_psi_w
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._dropout = dropout

        self.propensity_net = self._build_classifier(dim_L_plus, hidden_dim, n_layers, dropout)
        self.imputation_net = self._build_regressor(dim_L, dim_psi_w, hidden_dim, n_layers, dropout)

        if gpu_flg:
            self.propensity_net.to("cuda:0")
            self.imputation_net.to("cuda:0")

        self.propensity_opt = torch.optim.Adam(self.propensity_net.parameters(), lr=1e-3)
        self.imputation_opt = torch.optim.Adam(self.imputation_net.parameters(), lr=1e-3)

    @staticmethod
    def _build_classifier(input_dim: int, hidden_dim: int, n_layers: int, dropout: float) -> nn.Module:
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        return nn.Sequential(*layers)

    @staticmethod
    def _build_regressor(input_dim: int, output_dim: int, hidden_dim: int, n_layers: int, dropout: float) -> nn.Module:
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, output_dim)]
        return nn.Sequential(*layers)

    def fit_propensity(
        self,
        L_plus: torch.Tensor,
        delta_w: torch.Tensor,
        n_epochs: int = 50,
    ):
        """Train propensity model on all samples.

        Args:
            L_plus:  (n, dim_L_plus) — concatenation of (A, Z, X, Y)
            delta_w: (n,) or (n,1)   — binary missingness indicator
            n_epochs: training epochs
        """
        if delta_w.dim() == 1:
            delta_w = delta_w.unsqueeze(1)
        bce = nn.BCELoss()
        self.propensity_net.train(True)
        for _ in range(n_epochs):
            self.propensity_opt.zero_grad()
            e_hat = self.propensity_net(L_plus)
            loss = bce(e_hat, delta_w)
            loss.backward()
            self.propensity_opt.step()
        self.propensity_net.train(False)

    def fit_imputation(
        self,
        L: torch.Tensor,
        psi_w_targets: torch.Tensor,
        delta_w: torch.Tensor,
        n_epochs: int = 50,
    ):
        """Train imputation model on complete cases only.

        Args:
            L:             (n, dim_L)   — concatenation of (A, Z, X)
            psi_w_targets: (n, dim_psi_w) — ψ_{θ_W}(W_i); ignored where delta_w=0
            delta_w:       (n,) or (n,1)  — missingness indicator
            n_epochs: training epochs
        """
        if delta_w.dim() > 1:
            delta_w_mask = delta_w.squeeze(1).bool()
        else:
            delta_w_mask = delta_w.bool()

        L_obs = L[delta_w_mask]
        targets_obs = psi_w_targets[delta_w_mask]

        if L_obs.shape[0] == 0:
            return  # no complete cases in this fold, skip

        mse = nn.MSELoss()
        self.imputation_net.train(True)
        for _ in range(n_epochs):
            self.imputation_opt.zero_grad()
            m_hat = self.imputation_net(L_obs)
            loss = mse(m_hat, targets_obs)
            loss.backward()
            self.imputation_opt.step()
        self.imputation_net.train(False)

    def predict_propensity(self, L_plus: torch.Tensor) -> torch.Tensor:
        """Return ê(L+) in (0,1). Shape: (n, 1)."""
        self.propensity_net.train(False)
        with torch.no_grad():
            return self.propensity_net(L_plus)

    def predict_imputation(self, L: torch.Tensor) -> torch.Tensor:
        """Return m̂_ψ(L) in R^{d_W}. Shape: (n, dim_psi_w)."""
        self.imputation_net.train(False)
        with torch.no_grad():
            return self.imputation_net(L)

    def reset_weights(self):
        """Re-initialise nuisance networks and optimisers from scratch.

        Required by Algorithm 5.1 (thesis): each fold's nuisance models must be
        trained independently on I_{-k} without any information carried over from
        previous folds or epochs.  Calling this before each fold's fit ensures the
        cross-fitting independence guarantee is satisfied.
        """
        device = next(self.propensity_net.parameters()).device

        self.propensity_net = self._build_classifier(
            self._dim_L_plus, self._hidden_dim, self._n_layers, self._dropout
        ).to(device)
        self.imputation_net = self._build_regressor(
            self._dim_L, self._dim_psi_w, self._hidden_dim, self._n_layers, self._dropout
        ).to(device)

        self.propensity_opt = torch.optim.Adam(self.propensity_net.parameters(), lr=1e-3)
        self.imputation_opt = torch.optim.Adam(self.imputation_net.parameters(), lr=1e-3)

    def clone_weights(self) -> "NuisanceModels":
        """Return a deep copy of current nuisance model weights (for saving fold estimates)."""
        import copy
        return copy.deepcopy(self)
