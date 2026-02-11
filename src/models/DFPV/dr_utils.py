from typing import Optional
import torch


def construct_L_plus(
    treatment: torch.Tensor,
    treatment_proxy: torch.Tensor,
    outcome: torch.Tensor,
    backdoor: Optional[torch.Tensor],
) -> torch.Tensor:
    """Concatenate (A, Z, X, Y) into L+ for propensity model input.

    Args:
        treatment: (n, d_a)
        treatment_proxy: (n, d_z)
        outcome: (n, 1) or (n,)
        backdoor: (n, d_x) or None

    Returns:
        L_plus: (n, d_a + d_z + d_x + 1)
    """
    if outcome.dim() == 1:
        outcome = outcome.unsqueeze(1)
    parts = [treatment, treatment_proxy, outcome]
    if backdoor is not None:
        parts.append(backdoor)
    return torch.cat(parts, dim=1)


def construct_L(
    treatment: torch.Tensor,
    treatment_proxy: torch.Tensor,
    backdoor: Optional[torch.Tensor],
) -> torch.Tensor:
    """Concatenate (A, Z, X) into L for imputation model input.

    Args:
        treatment: (n, d_a)
        treatment_proxy: (n, d_z)
        backdoor: (n, d_x) or None

    Returns:
        L: (n, d_a + d_z + d_x)
    """
    parts = [treatment, treatment_proxy]
    if backdoor is not None:
        parts.append(backdoor)
    return torch.cat(parts, dim=1)


def compute_dr_pseudo_outcome(
    psi_w: torch.Tensor,
    m_hat: torch.Tensor,
    e_hat: torch.Tensor,
    delta_w: torch.Tensor,
    propensity_clip: float = 1e-3,
) -> torch.Tensor:
    """Compute DR pseudo-outcome phi_DR (Thesis Eq. 5.12 / Algorithm 5.1).

    phi_DR_i = m_hat(L_i) + (delta_w_i / e_hat(L+_i)) * (psi_w(W_i) - m_hat(L_i))

    For missing cases (delta_w=0): correction term is zero, so phi_DR = m_hat.
    For observed cases (delta_w=1): IPW-weighted residual is added.

    Args:
        psi_w:   (n, d_W) — ψ_{θ_W}(W_i); values at missing entries are ignored
        m_hat:   (n, d_W) — imputation model prediction m̂_ψ(L_i)
        e_hat:   (n, 1) or (n,) — propensity score ê(L+_i), clipped inside
        delta_w: (n, 1) or (n,) — missingness indicator (1=observed)
        propensity_clip: clip e_hat to [clip, 1-clip] to avoid division by zero

    Returns:
        phi_dr: (n, d_W)
    """
    e_hat = e_hat.clamp(min=propensity_clip, max=1.0 - propensity_clip)

    if e_hat.dim() == 1:
        e_hat = e_hat.unsqueeze(1)
    if delta_w.dim() == 1:
        delta_w = delta_w.unsqueeze(1)

    # IPW correction; zero for missing cases since delta_w=0
    correction = (delta_w / e_hat) * (psi_w - m_hat)
    return m_hat + correction


def compute_effective_sample_size(
    e_hat: torch.Tensor,
    delta_w: torch.Tensor,
    propensity_clip: float = 1e-3,
) -> float:
    """Compute ESS = (sum delta/e)^2 / sum (delta/e)^2 as a diagnostic.

    Low ESS indicates extreme propensity weights.
    """
    e_clipped = e_hat.clamp(min=propensity_clip, max=1.0 - propensity_clip)
    if e_clipped.dim() > 1:
        e_clipped = e_clipped.squeeze(1)
    if delta_w.dim() > 1:
        delta_w = delta_w.squeeze(1)
    weights = delta_w / e_clipped
    numerator = weights.sum() ** 2
    denominator = (weights ** 2).sum()
    if denominator.item() == 0:
        return 0.0
    return (numerator / denominator).item()
