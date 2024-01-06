from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ApplyTransform(Dataset):
    """Apply transform to dataset"""

    def __init__(self, dataset: Dataset, transform: nn.Module):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.dataset)
