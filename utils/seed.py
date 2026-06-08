"""
Random seed helpers for reproducible training runs.
"""

from __future__ import annotations

import random
from typing import Optional

import torch

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


def set_random_seed(
    seed: Optional[int],
    *,
    cuda_device: Optional[int] = None,
    seed_cuda: bool = True,
) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    if seed is None:
        return

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

    # Seed the CPU generator without implicitly touching every CUDA device.
    torch.default_generator.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        if cuda_device is None:
            torch.cuda.manual_seed_all(seed)
        else:
            with torch.cuda.device(cuda_device):
                torch.cuda.manual_seed(seed)
