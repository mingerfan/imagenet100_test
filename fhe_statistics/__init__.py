"""
FHE Statistics - 全同态加密网络统计分析工具

主要功能：
1. FHE操作统计：计算rotation、multiplication、rescale等操作
2. Boot优化：使用动态规划优化boot插入位置
3. 网络延迟分析：分析各个网络的FHE延迟
4. 批量横向比较：支持YAML配置的批量网络对比

快速开始：
    from fhe_statistics import FheInfo, analyze_model

    # 分析单个模型
    model = YourModel()
    fhe_info = analyze_model(model, "MyModel", output_folder="results")

    # 批量分析
    from fhe_statistics.batch_analyzer import BatchAnalyzer
    analyzer = BatchAnalyzer("config.yaml")
    analyzer.run()
"""

from .statistics_fn import (
    FheInfo,
    analyze_model,
    compare_networks,
)
from .fhelipe_statistics_fn import (
    FhelipeInfo,
    analyze_model_fhelipe,
)
from .orion_statistics_fn import (
    OrionInfo,
    analyze_model_orion,
)

from .boot_optimizer import BootOptimizer, NodeInfo

__version__ = "1.0.0"
__all__ = [
    "FheInfo",
    "analyze_model",
    "compare_networks",
    "FhelipeInfo",
    "analyze_model_fhelipe",
    "OrionInfo",
    "analyze_model_orion",
    "BootOptimizer",
    "NodeInfo",
]
