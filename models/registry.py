"""
模型注册器
用于统一管理和创建模型
"""

import torch.nn as nn
from typing import Dict, Type, Any


class ModelRegistry:
    """模型注册器类"""
    
    def __init__(self):
        self._registry: Dict[str, Type[nn.Module]] = {}
    
    def register(self, name: str):
        """
        装饰器，用于注册模型
        
        Args:
            name: 模型名称
        """
        def decorator(cls):
            if name in self._registry:
                raise ValueError(f"模型 '{name}' 已经注册")
            self._registry[name] = cls
            return cls
        return decorator
    
    def get(self, name: str, **kwargs) -> nn.Module:
        """
        获取模型实例
        
        Args:
            name: 模型名称
            **kwargs: 模型参数
        
        Returns:
            模型实例
        """
        if name not in self._registry:
            raise ValueError(f"未找到模型 '{name}'，可用的模型: {list(self._registry.keys())}")
        return self._registry[name](**kwargs)
    
    def list_models(self) -> list:
        """
        列出所有已注册的模型
        
        Returns:
            模型名称列表
        """
        return list(self._registry.keys())


# 全局模型注册器实例
MODEL_REGISTRY = ModelRegistry()


def register_model(name: str):
    """
    注册模型的装饰器函数
    
    Args:
        name: 模型名称
    """
    return MODEL_REGISTRY.register(name)


def get_model(name: str, **kwargs) -> nn.Module:
    """
    获取模型实例
    
    Args:
        name: 模型名称
        **kwargs: 模型参数
    
    Returns:
        模型实例
    """
    return MODEL_REGISTRY.get(name, **kwargs)