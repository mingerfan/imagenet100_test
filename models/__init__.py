"""
模型模块
"""
from .registry import MODEL_REGISTRY, get_model, register_model
from .resnet import resnet18, resnet34, resnet50
from .efficientnet import efficientnet_b0

# 导入 gate_net 模块以注册所有变体
from . import gate_net
from . import json_registered

__all__ = ['MODEL_REGISTRY', 'get_model', 'register_model', 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0']
