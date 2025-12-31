"""
模型模块
"""
from .registry import MODEL_REGISTRY, get_model, register_model
from .resnet import resnet18, resnet34, resnet50

__all__ = ['MODEL_REGISTRY', 'get_model', 'register_model', 'resnet18', 'resnet34', 'resnet50']