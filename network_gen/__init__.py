"""
FHE-NAS 网络生成模块

提供神经网络架构搜索的网络生成功能。
"""

from .search_space import (
    SearchSpace,
    StrideEncoder,
    ChannelCalculator,
    # BLOCK_TYPES,
    ACTIVATION_TYPES,
    STEM_CONFIGS,
    SECOND_DOWNSAMPLE_CONFIGS,
    UNIFIED_BLOCKS,
    StemConfig,
    SecondDownsampleConfig,
    # BlockSpec,
    ActivationSpec,
    UnifiedBlockSpec,
)

from .network_config import (
    NetworkConfig,
    BlockConfig,
    NetworkConfigBatch,
)

from .network_generator import (
    RandomNetworkGenerator,
    NetworkBuilder,
    GeneratedNetwork,
    create_network,
    create_random_network,
)

__all__ = [
    # search_space
    "SearchSpace",
    "StrideEncoder",
    "ChannelCalculator",
    # "BLOCK_TYPES",
    "ACTIVATION_TYPES",
    "STEM_CONFIGS",
    "SECOND_DOWNSAMPLE_CONFIGS",
    "UNIFIED_BLOCKS",
    "StemConfig",
    "SecondDownsampleConfig",
    "BlockSpec",
    "ActivationSpec",
    "UnifiedBlockSpec",
    # network_config
    "NetworkConfig",
    "BlockConfig",
    "NetworkConfigBatch",
    # network_generator
    "RandomNetworkGenerator",
    "NetworkBuilder",
    "GeneratedNetwork",
    "create_network",
    "create_random_network",
]
