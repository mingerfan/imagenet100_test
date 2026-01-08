"""操作处理器注册表 - 使用工厂模式替代长的 if-elif 链"""

from typing import Callable, Dict, Any
from torch import nn
import torch
import operator


class OperationHandlerRegistry:
    """操作处理器注册表，管理各种操作的处理函数

    消除：
    - _handle_module() 中的 15+ 个 isinstance 检查
    - _handle_function() 中的 8+ 个 if-elif 分支
    - _handle_method() 中的 5+ 个 if-elif 分支
    """

    def __init__(self):
        """初始化注册表"""
        self._module_handlers: Dict[type, str] = {}  # 模块类型 -> 处理方法名
        self._function_handlers: Dict[Any, str] = {}  # 函数 -> 处理方法名
        self._method_handlers: Dict[str, str] = {}  # 方法名 -> 处理方法名
        self._register_all()

    def _register_all(self):
        """注册所有操作处理器"""
        self._register_module_handlers()
        self._register_function_handlers()
        self._register_method_handlers()

    def _register_module_handlers(self):
        """注册模块处理器"""
        # 核心算子
        self._module_handlers[nn.Conv2d] = 'conv_statistics'
        self._module_handlers[nn.Linear] = 'linear_statistics'

        # 激活函数
        self._module_handlers[nn.ReLU] = 'relu_statistics'
        self._module_handlers[nn.ReLU6] = 'relu6_statistics'
        self._module_handlers[nn.SiLU] = 'swish_statistics'
        self._module_handlers[nn.Sigmoid] = 'sigmoid_statistics'

        # 池化
        self._module_handlers[nn.MaxPool2d] = 'maxpool_statistics'
        self._module_handlers[nn.AvgPool2d] = 'avepool_statistics'
        self._module_handlers[nn.AdaptiveAvgPool2d] = 'adaptiveavepool2d_statistics'

        # Fused 算子（跳过）
        self._module_handlers[nn.BatchNorm2d] = 'pass_through_statistics'
        self._module_handlers[nn.BatchNorm1d] = 'pass_through_statistics'
        self._module_handlers[nn.Dropout] = 'pass_through_statistics'
        self._module_handlers[nn.Identity] = 'pass_through_statistics'

    def _register_function_handlers(self):
        """注册函数处理器"""
        # 算术操作
        self._function_handlers[torch.add] = 'add_statistics'
        self._function_handlers[operator.add] = 'add_statistics'

        self._function_handlers[torch.mul] = 'mul_statistics'
        self._function_handlers[operator.mul] = 'mul_statistics'

        # 激活函数
        self._function_handlers[torch.relu] = 'relu_statistics'
        self._function_handlers[torch.nn.functional.relu] = 'relu_statistics'

        self._function_handlers[torch.sigmoid] = 'sigmoid_statistics'
        self._function_handlers[torch.nn.functional.sigmoid] = 'sigmoid_statistics'

        # 其他操作
        self._function_handlers[torch.flatten] = 'pass_through_statistics'
        self._function_handlers[torch.cat] = 'cat_statistics'
        self._function_handlers[torch.nn.functional.adaptive_avg_pool2d] = 'adaptiveavepool2d_statistics'
        self._function_handlers[getattr] = 'pass_through_statistics'

    def _register_method_handlers(self):
        """注册方法处理器"""
        # 形状变化操作
        self._method_handlers['flatten'] = 'pass_through_statistics'
        self._method_handlers['view'] = 'pass_through_statistics'
        self._method_handlers['reshape'] = 'pass_through_statistics'
        self._method_handlers['contiguous'] = 'pass_through_statistics'

    def register_module(self, module_type: type, handler_name: str) -> None:
        """注册自定义模块处理器

        Args:
            module_type: 模块类型
            handler_name: 处理方法的名称
        """
        self._module_handlers[module_type] = handler_name

    def register_function(self, func: Any, handler_name: str) -> None:
        """注册自定义函数处理器

        Args:
            func: 函数对象
            handler_name: 处理方法的名称
        """
        self._function_handlers[func] = handler_name

    def register_method(self, method_name: str, handler_name: str) -> None:
        """注册自定义方法处理器

        Args:
            method_name: 方法名称
            handler_name: 处理方法的名称
        """
        self._method_handlers[method_name] = handler_name

    def get_module_handler(self, module: nn.Module) -> str:
        """获取模块的处理器名称

        Args:
            module: PyTorch 模块实例

        Returns:
            处理方法的名称，如果不存在则返回 None
        """
        return self._module_handlers.get(type(module))

    def get_function_handler(self, func: Any) -> str:
        """获取函数的处理器名称

        Args:
            func: 函数对象

        Returns:
            处理方法的名称，如果不存在则返回 None
        """
        return self._function_handlers.get(func)

    def get_method_handler(self, method_name: str) -> str:
        """获取方法的处理器名称

        Args:
            method_name: 方法名称

        Returns:
            处理方法的名称，如果不存在则返回 None
        """
        return self._method_handlers.get(method_name)

    def get_all_registered_modules(self) -> Dict[type, str]:
        """获取所有注册的模块处理器

        Returns:
            模块类型 -> 处理方法名的字典
        """
        return dict(self._module_handlers)

    def get_all_registered_functions(self) -> Dict[Any, str]:
        """获取所有注册的函数处理器

        Returns:
            函数 -> 处理方法名的字典
        """
        return dict(self._function_handlers)

    def get_all_registered_methods(self) -> Dict[str, str]:
        """获取所有注册的方法处理器

        Returns:
            方法名 -> 处理方法名的字典
        """
        return dict(self._method_handlers)
