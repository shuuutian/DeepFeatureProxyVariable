from typing import NamedTuple, Optional, List, Tuple
import numpy as np
import torch


class PVTrainDataSetMAR(NamedTuple):
    treatment: np.ndarray          # A
    treatment_proxy: np.ndarray    # Z
    outcome_proxy: np.ndarray      # W (NaN or zeros for missing entries)
    outcome: np.ndarray            # Y
    backdoor: Optional[np.ndarray] # X
    delta_w: np.ndarray            # binary indicator: 1=observed, 0=missing


class PVTrainDataSetMARTorch(NamedTuple):
    treatment: torch.Tensor
    treatment_proxy: torch.Tensor
    outcome_proxy: torch.Tensor
    outcome: torch.Tensor
    backdoor: Optional[torch.Tensor]
    delta_w: torch.Tensor

    @classmethod
    def from_numpy(cls, data: PVTrainDataSetMAR):
        backdoor = None
        if data.backdoor is not None:
            backdoor = torch.tensor(data.backdoor, dtype=torch.float32)
        return PVTrainDataSetMARTorch(
            treatment=torch.tensor(data.treatment, dtype=torch.float32),
            treatment_proxy=torch.tensor(data.treatment_proxy, dtype=torch.float32),
            outcome_proxy=torch.tensor(data.outcome_proxy, dtype=torch.float32),
            outcome=torch.tensor(data.outcome, dtype=torch.float32),
            backdoor=backdoor,
            delta_w=torch.tensor(data.delta_w, dtype=torch.float32),
        )

    def to_gpu(self):
        backdoor = None
        if self.backdoor is not None:
            backdoor = self.backdoor.cuda()
        return PVTrainDataSetMARTorch(
            treatment=self.treatment.cuda(),
            treatment_proxy=self.treatment_proxy.cuda(),
            outcome_proxy=self.outcome_proxy.cuda(),
            outcome=self.outcome.cuda(),
            backdoor=backdoor,
            delta_w=self.delta_w.cuda(),
        )

    def subset(self, idx: torch.Tensor) -> "PVTrainDataSetMARTorch":
        backdoor = None
        if self.backdoor is not None:
            backdoor = self.backdoor[idx]
        return PVTrainDataSetMARTorch(
            treatment=self.treatment[idx],
            treatment_proxy=self.treatment_proxy[idx],
            outcome_proxy=self.outcome_proxy[idx],
            outcome=self.outcome[idx],
            backdoor=backdoor,
            delta_w=self.delta_w[idx],
        )


def create_k_folds(
    data: PVTrainDataSetMARTorch,
    n_folds: int,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """Return list of index tensors, one per fold.

    Args:
        data: training data
        n_folds: number of folds K
        seed: random seed for reproducible fold assignment (pass trainer's random_seed)
    """
    n = data.treatment.shape[0]
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    fold_indices = []
    fold_size = n // n_folds
    for k in range(n_folds):
        start = k * fold_size
        end = start + fold_size if k < n_folds - 1 else n
        fold_indices.append(perm[start:end])
    return fold_indices


def get_train_val_split(
    data: PVTrainDataSetMARTorch,
    fold_indices: List[torch.Tensor],
    val_fold: int,
) -> Tuple[PVTrainDataSetMARTorch, PVTrainDataSetMARTorch]:
    """Return (train_data, val_data) where train is all folds except val_fold."""
    val_idx = fold_indices[val_fold]
    train_idx = torch.cat([fold_indices[k] for k in range(len(fold_indices)) if k != val_fold])
    return data.subset(train_idx), data.subset(val_idx)
