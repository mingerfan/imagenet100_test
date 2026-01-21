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


def set_random_seed(seed: Optional[int]) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    if seed is None:
        return

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
