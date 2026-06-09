"""激活函数配置表 - 消除重复的激活函数统计方法"""

from typing import Dict, Any


# 激活函数配置：定义每个激活函数的计算成本参数
ACTIVATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    'relu': {
        'depth_delta': 15,
        'mul_both_factor': 33,
        'mul_single_factor': 33,
    },
    'relu6': {
        'depth_delta': 15,
        'mul_both_factor': 33,
        'mul_single_factor': 33,
    },
    'learnable_swish': {
        'depth_delta': 8,
        'mul_both_factor': 16,
        'mul_single_factor': 16,
    },
    'swish': {
        'depth_delta': 7,
        'mul_both_factor': 16,
        'mul_single_factor': 16,
    },
    'learnable_relu': {
        'depth_delta': 15,
        'mul_both_factor': 33,
        'mul_single_factor': 33,
    },
    'sigmoid': {
        'depth_delta': 7,
        'mul_both_factor': 15,
        'mul_single_factor': 15,
    },
    'poly4': {
        'depth_delta': 3,
        'mul_both_factor': 8,
        'mul_single_factor': 8,
    },
    'poly4_herpn': {
        'depth_delta': 3,
        'mul_both_factor': 8,
        'mul_single_factor': 8,
    },
    'hermitepoly4': {
        'depth_delta': 3,
        'mul_both_factor': 8,
        'mul_single_factor': 8,
    },
    'swish_herpn': {
        'depth_delta': 1,
        'mul_both_factor': 4,
        'mul_single_factor': 4,
    },
}


def get_activation_config(activation_type: str) -> Dict[str, Any]:
    """获取激活函数配置

    Args:
        activation_type: 激活函数类型名称

    Returns:
        包含计算成本参数的配置字典

    Raises:
        ValueError: 如果激活函数类型不在配置中
    """
    if activation_type not in ACTIVATION_CONFIGS:
        raise ValueError(f"未知的激活函数类型: {activation_type}")
    return ACTIVATION_CONFIGS[activation_type]
