"""深度 Binning 工具类 - 统一深度分组逻辑"""

from typing import List, Dict, Tuple
from collections import defaultdict


class DepthBinner:
    """深度分组工具，用于将节点按深度分组到不同的 bin 中

    消除 3 处重复的 binning 逻辑：
    - get_depth_histogram_data()
    - get_depth_flops_distribution()
    - get_depth_parameter_distribution()
    """

    def __init__(
        self,
        bin_size: int = 1,
        max_depth: int = None,
        auto_bin: bool = True,
    ):
        """初始化 DepthBinner

        Args:
            bin_size: 每个 bin 的深度范围大小
            max_depth: 最大深度值（用于预分配 bin）
            auto_bin: 是否自动根据数据调整 bin 大小
        """
        self.bin_size = bin_size
        self.max_depth = max_depth
        self.auto_bin = auto_bin
        self.bins: Dict[int, List] = defaultdict(list)

    def get_bin_index(self, depth: int) -> int:
        """获取指定深度所属的 bin 索引

        Args:
            depth: 节点深度

        Returns:
            bin 索引（0-based）
        """
        if depth < 0:
            return -1
        return depth // self.bin_size

    def add_item(self, depth: int, item: any) -> None:
        """将一个项添加到对应的 bin

        Args:
            depth: 项的深度
            item: 要添加的项
        """
        bin_idx = self.get_bin_index(depth)
        self.bins[bin_idx].append(item)

    def get_bins(self) -> Dict[int, List]:
        """获取所有 bin

        Returns:
            bin_idx -> items 的字典
        """
        return dict(self.bins)

    def get_bin_range(self, bin_idx: int) -> Tuple[int, int]:
        """获取指定 bin 的深度范围

        Args:
            bin_idx: bin 索引

        Returns:
            (min_depth, max_depth) 元组（包含边界）
        """
        min_depth = bin_idx * self.bin_size
        max_depth = min_depth + self.bin_size - 1
        return min_depth, max_depth

    def get_bin_label(self, bin_idx: int) -> str:
        """获取 bin 的标签字符串

        Args:
            bin_idx: bin 索引

        Returns:
            可读的标签，如 "0-4" 或 "5-9"
        """
        min_depth, max_depth = self.get_bin_range(bin_idx)
        return f"{min_depth}-{max_depth}"

    def clear(self) -> None:
        """清空所有 bin"""
        self.bins.clear()

    def get_summary(self) -> Dict[str, any]:
        """获取 binning 的汇总信息

        Returns:
            包含统计信息的字典
        """
        if not self.bins:
            return {}

        bin_indices = sorted(self.bins.keys())
        return {
            'num_bins': len(bin_indices),
            'bin_range': (bin_indices[0], bin_indices[-1]),
            'items_per_bin': {idx: len(self.bins[idx]) for idx in bin_indices},
            'total_items': sum(len(items) for items in self.bins.values()),
        }


class DepthMetricsCollector:
    """深度度量收集器 - 按深度统计指标

    用于统一以下逻辑：
    - get_shallow_layer_metrics() 中的深度分组
    - get_depth_flops_distribution() 中的深度分组
    - get_depth_parameter_distribution() 中的深度分组
    """

    def __init__(self, bin_size: int = 1):
        """初始化收集器

        Args:
            bin_size: 每个 bin 的深度范围
        """
        self.binner = DepthBinner(bin_size=bin_size)
        self.metrics: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {
                'latency': 0,
                'flops': 0,
                'parameters': 0,
                'count': 0,
            }
        )

    def add_node_metrics(
        self,
        depth: int,
        latency: float = 0,
        flops: int = 0,
        parameters: int = 0,
    ) -> None:
        """添加节点度量

        Args:
            depth: 节点深度
            latency: 延迟
            flops: FLOPs 数量
            parameters: 参数数量
        """
        bin_idx = self.binner.get_bin_index(depth)

        self.metrics[bin_idx]['latency'] += latency
        self.metrics[bin_idx]['flops'] += flops
        self.metrics[bin_idx]['parameters'] += parameters
        self.metrics[bin_idx]['count'] += 1

    def get_metrics_by_bin(self) -> Dict[int, Dict[str, float]]:
        """按 bin 获取汇总的度量

        Returns:
            bin_idx -> metrics 字典
        """
        return dict(self.metrics)

    def get_metrics_as_lists(
        self,
    ) -> Tuple[List[str], List[float], List[int], List[int]]:
        """以列表形式获取度量，便于绘图

        Returns:
            (labels, latencies, flops, parameters) 元组
        """
        bin_indices = sorted(self.metrics.keys())
        labels = [self.binner.get_bin_label(idx) for idx in bin_indices]
        latencies = [self.metrics[idx]['latency'] for idx in bin_indices]
        flops = [self.metrics[idx]['flops'] for idx in bin_indices]
        parameters = [self.metrics[idx]['parameters'] for idx in bin_indices]

        return labels, latencies, flops, parameters

    def clear(self) -> None:
        """清空所有数据"""
        self.binner.clear()
        self.metrics.clear()
