"""
模型模块
"""
from .registry import MODEL_REGISTRY, get_model, register_model
from .resnet import resnet18, resnet34, resnet50
from .resnet20 import resnet20
from .resnet32 import resnet32
from .resnet44 import resnet44
from .resnet56 import resnet56
from .resnet110 import resnet110
from .efficientnet import efficientnet_b0

# 导入 gate_net 模块以注册所有变体
from . import gate_net
from . import json_registered

__all__ = [
    'MODEL_REGISTRY',
    'get_model',
    'register_model',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet20',
    'resnet32',
    'resnet44',
    'resnet56',
    'resnet110',
    'efficientnet_b0',
]
