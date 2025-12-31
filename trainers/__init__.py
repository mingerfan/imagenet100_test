"""
训练器模块
"""
from .base_trainer import Trainer
from .multi_gpu_manager import MultiGPUManager

__all__ = ['Trainer', 'MultiGPUManager']