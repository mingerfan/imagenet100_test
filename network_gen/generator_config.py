"""
网络生成器配置系统

支持通过YAML配置文件控制网络生成的各个方面，包括：
- 数据集特定的配置（分辨率、类别数等）
- 搜索空间约束（允许的block类型、stem配置等）
- 特殊约束（如CIFAR-10的前几层不降分辨率）
- 输出目录结构
"""

import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str                    # 数据集名称 (imagenet, cifar10, etc.)
    num_classes: int             # 分类数量
    input_size: int              # 输入图像大小

    def __repr__(self):
        return f"Dataset({self.name}, {self.input_size}x{self.input_size}, {self.num_classes}classes)"


@dataclass
class StemConstraints:
    """Stem层约束"""
    enabled: bool = True                          # 是否启用stem层
    allowed_codes: Optional[List[int]] = None     # 允许的stem配置编码 [0-3]
    custom_config: Optional[Dict[str, Any]] = None # 自定义stem配置（用于特殊数据集）

    def is_code_allowed(self, code: int) -> bool:
        """检查code是否允许"""
        if self.allowed_codes is None:
            return True
        return code in self.allowed_codes


@dataclass
class SecondDownsampleConstraints:
    """第二次降分辨率约束"""
    enabled: bool = True                          # 是否启用第二次降分辨率
    allowed_codes: Optional[List[int]] = None     # 允许的配置编码 [0-4]

    def is_code_allowed(self, code: int) -> bool:
        """检查code是否允许"""
        if not self.enabled:
            return False
        if self.allowed_codes is None:
            return True
        return code in self.allowed_codes


@dataclass
class LayerConstraint:
    """单层约束"""
    position: int                                 # 层位置（从0开始）
    allowed_block_ids: Optional[List[int]] = None # 允许的block ID
    stride: Optional[int] = None                  # 强制的stride值（如果指定）

    def __repr__(self):
        parts = [f"pos={self.position}"]
        if self.allowed_block_ids is not None:
            parts.append(f"blocks={self.allowed_block_ids}")
        if self.stride is not None:
            parts.append(f"stride={self.stride}")
        return f"LayerConstraint({', '.join(parts)})"


@dataclass
class BlockConstraints:
    """Block约束"""
    allowed_block_ids: Optional[List[int]] = None              # 全局允许的block ID [0-23]
    first_layers_constraints: Optional[List[LayerConstraint]] = None  # 前几层的特殊约束

    def is_block_id_allowed(self, block_id: int, position: Optional[int] = None) -> bool:
        """
        检查block_id在指定位置是否允许

        Args:
            block_id: block ID [0-23]
            position: 层位置（可选）

        Returns:
            是否允许
        """
        # 首先检查位置特定约束
        if position is not None and self.first_layers_constraints:
            for constraint in self.first_layers_constraints:
                if constraint.position == position:
                    if constraint.allowed_block_ids is not None:
                        return block_id in constraint.allowed_block_ids

        # 然后检查全局约束
        if self.allowed_block_ids is None:
            return True
        return block_id in self.allowed_block_ids

    def get_stride_for_position(self, position: int) -> Optional[int]:
        """
        获取指定位置的强制stride值

        Args:
            position: 层位置

        Returns:
            stride值，如果没有约束则返回None
        """
        if self.first_layers_constraints:
            for constraint in self.first_layers_constraints:
                if constraint.position == position and constraint.stride is not None:
                    return constraint.stride
        return None


@dataclass
class StrideConstraints:
    """Stride约束"""
    allowed_block_counts: Optional[List[int]] = None  # 允许的block数量
    num_strides: int = 3                              # 降分辨率次数

    def is_block_count_allowed(self, count: int) -> bool:
        """检查block数量是否允许"""
        if self.allowed_block_counts is None:
            return True
        return count in self.allowed_block_counts


@dataclass
class CTPolicyConstraints:
    """CT策略约束"""
    allowed: List[str] = field(default_factory=lambda: ["keep", "half"])  # 允许的策略

    def is_policy_allowed(self, policy: str) -> bool:
        """检查策略是否允许"""
        return policy in self.allowed


@dataclass
class SearchSpaceConstraints:
    """搜索空间约束"""
    ct_slots: int = 32768                                   # CT槽位数
    initial_ct_count: int = 1                               # 初始CT数量
    stem: StemConstraints = field(default_factory=StemConstraints)
    second_downsample: SecondDownsampleConstraints = field(
        default_factory=SecondDownsampleConstraints
    )
    blocks: BlockConstraints = field(default_factory=BlockConstraints)
    stride: StrideConstraints = field(default_factory=StrideConstraints)
    ct_policies: CTPolicyConstraints = field(default_factory=CTPolicyConstraints)


@dataclass
class OutputConfig:
    """输出配置"""
    base_dir: str = "generated_networks"  # 基础输出目录
    save_format: str = "json"             # 保存格式 (json)


@dataclass
class GeneratorConfig:
    """
    网络生成器完整配置

    示例YAML配置：
    ```yaml
    name: "imagenet_224"
    description: "ImageNet-100 with 224x224 input"

    dataset:
      name: "imagenet"
      num_classes: 100
      input_size: 224

    search_space:
      ct_slots: 32768
      initial_ct_count: 1
      stem:
        enabled: true
        allowed_codes: [0, 1, 2, 3]
      second_downsample:
        enabled: true
        allowed_codes: [0, 1, 2, 3, 4]
      blocks:
        allowed_block_ids: null
        first_layers_constraints: null
      stride:
        allowed_block_counts: [4, 6, 8, 10, 12, 14, 16]
        num_strides: 3
      ct_policies:
        allowed: ["keep", "half"]

    output:
      base_dir: "generated_networks/imagenet_224"
      save_format: "json"
    ```
    """
    name: str                                      # 配置名称
    description: str = ""                          # 描述
    dataset: DatasetConfig = None                  # 数据集配置
    search_space: SearchSpaceConstraints = field(
        default_factory=SearchSpaceConstraints
    )
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self):
        """后处理：确保必要字段存在"""
        if self.dataset is None:
            # 默认ImageNet配置
            self.dataset = DatasetConfig(
                name="imagenet",
                num_classes=100,
                input_size=224
            )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GeneratorConfig":
        """
        从YAML文件加载配置

        Args:
            yaml_path: YAML配置文件路径

        Returns:
            GeneratorConfig实例
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratorConfig":
        """从字典创建配置"""
        # 解析dataset
        dataset_data = data.get("dataset", {})
        dataset = DatasetConfig(**dataset_data)

        # 解析search_space
        search_space_data = data.get("search_space", {})
        search_space = cls._parse_search_space(search_space_data)

        # 解析output
        output_data = data.get("output", {})
        output = OutputConfig(**output_data)

        return cls(
            name=data.get("name", "default"),
            description=data.get("description", ""),
            dataset=dataset,
            search_space=search_space,
            output=output,
        )

    @classmethod
    def _parse_search_space(cls, data: Dict[str, Any]) -> SearchSpaceConstraints:
        """解析搜索空间约束"""
        # 解析stem
        stem_data = data.get("stem", {})
        stem = StemConstraints(
            enabled=stem_data.get("enabled", True),
            allowed_codes=stem_data.get("allowed_codes"),
            custom_config=stem_data.get("custom_config"),
        )

        # 解析second_downsample
        second_ds_data = data.get("second_downsample", {})
        second_downsample = SecondDownsampleConstraints(
            enabled=second_ds_data.get("enabled", True),
            allowed_codes=second_ds_data.get("allowed_codes"),
        )

        # 解析blocks
        blocks_data = data.get("blocks", {})
        first_layers_constraints = None
        if "first_layers_constraints" in blocks_data and blocks_data["first_layers_constraints"]:
            # 导入parse_block_ids用于解析block名称
            from .search_space import parse_block_ids

            first_layers_constraints = []
            for constraint in blocks_data["first_layers_constraints"]:
                # 解析allowed_block_ids（可能包含名称）
                allowed_block_ids = constraint.get("allowed_block_ids")
                if allowed_block_ids is not None:
                    allowed_block_ids = parse_block_ids(allowed_block_ids)

                first_layers_constraints.append(LayerConstraint(
                    position=constraint["position"],
                    allowed_block_ids=allowed_block_ids,
                    stride=constraint.get("stride"),
                ))

        # 解析全局allowed_block_ids（支持名称）
        from .search_space import parse_block_ids
        allowed_block_ids = blocks_data.get("allowed_block_ids")
        if allowed_block_ids is not None:
            allowed_block_ids = parse_block_ids(allowed_block_ids)

        blocks = BlockConstraints(
            allowed_block_ids=allowed_block_ids,
            first_layers_constraints=first_layers_constraints,
        )

        # 解析stride
        stride_data = data.get("stride", {})
        stride = StrideConstraints(
            allowed_block_counts=stride_data.get("allowed_block_counts"),
            num_strides=stride_data.get("num_strides", 3),
        )

        # 解析ct_policies
        ct_policies_data = data.get("ct_policies", {})
        ct_policies = CTPolicyConstraints(
            allowed=ct_policies_data.get("allowed", ["keep", "half"])
        )

        return SearchSpaceConstraints(
            ct_slots=data.get("ct_slots", 32768),
            initial_ct_count=data.get("initial_ct_count", 1),
            stem=stem,
            second_downsample=second_downsample,
            blocks=blocks,
            stride=stride,
            ct_policies=ct_policies,
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于保存）"""
        return {
            "name": self.name,
            "description": self.description,
            "dataset": {
                "name": self.dataset.name,
                "num_classes": self.dataset.num_classes,
                "input_size": self.dataset.input_size,
            },
            "search_space": self._search_space_to_dict(),
            "output": {
                "base_dir": self.output.base_dir,
                "save_format": self.output.save_format,
            },
        }

    def _search_space_to_dict(self) -> Dict[str, Any]:
        """将搜索空间转换为字典"""
        result = {
            "ct_slots": self.search_space.ct_slots,
            "initial_ct_count": self.search_space.initial_ct_count,
            "stem": {
                "enabled": self.search_space.stem.enabled,
                "allowed_codes": self.search_space.stem.allowed_codes,
            },
            "second_downsample": {
                "enabled": self.search_space.second_downsample.enabled,
                "allowed_codes": self.search_space.second_downsample.allowed_codes,
            },
            "blocks": {
                "allowed_block_ids": self.search_space.blocks.allowed_block_ids,
            },
            "stride": {
                "allowed_block_counts": self.search_space.stride.allowed_block_counts,
                "num_strides": self.search_space.stride.num_strides,
            },
            "ct_policies": {
                "allowed": self.search_space.ct_policies.allowed,
            },
        }

        # 添加first_layers_constraints
        if self.search_space.blocks.first_layers_constraints:
            result["blocks"]["first_layers_constraints"] = [
                {
                    "position": c.position,
                    "allowed_block_ids": c.allowed_block_ids,
                    "stride": c.stride,
                }
                for c in self.search_space.blocks.first_layers_constraints
            ]

        return result

    def save(self, yaml_path: str):
        """保存配置到YAML文件"""
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def get_output_dir(self) -> Path:
        """获取输出目录路径"""
        return Path(self.output.base_dir)

    def summary(self) -> str:
        """生成配置摘要"""
        lines = [
            "=" * 60,
            f"生成器配置: {self.name}",
            f"描述: {self.description}",
            "=" * 60,
            f"\n数据集: {self.dataset}",
            "\n搜索空间:",
            f"  CT槽位数: {self.search_space.ct_slots}",
            f"  初始CT数量: {self.search_space.initial_ct_count}",
            "\nStem层:",
            f"  启用: {self.search_space.stem.enabled}",
            f"  允许的配置: {self.search_space.stem.allowed_codes or 'All'}",
            "\n第二次降分辨率:",
            f"  启用: {self.search_space.second_downsample.enabled}",
            f"  允许的配置: {self.search_space.second_downsample.allowed_codes or 'All'}",
            "\nBlock约束:",
            f"  允许的Block ID: {self.search_space.blocks.allowed_block_ids or 'All'}",
        ]

        if self.search_space.blocks.first_layers_constraints:
            lines.append(f"  前几层约束:")
            for c in self.search_space.blocks.first_layers_constraints:
                lines.append(f"    - {c}")

        lines.extend([
            "\nStride约束:",
            f"  允许的Block数量: {self.search_space.stride.allowed_block_counts or 'All'}",
            f"  降分辨率次数: {self.search_space.stride.num_strides}",
            "\nCT策略:",
            f"  允许的策略: {self.search_space.ct_policies.allowed}",
            "\n输出:",
            f"  目录: {self.output.base_dir}",
            f"  格式: {self.output.save_format}",
            "=" * 60,
        ])

        return "\n".join(lines)


def create_default_imagenet_config() -> GeneratorConfig:
    """创建默认的ImageNet配置"""
    return GeneratorConfig(
        name="imagenet_224",
        description="ImageNet-100 with 224x224 input - default configuration",
        dataset=DatasetConfig(
            name="imagenet",
            num_classes=100,
            input_size=224,
        ),
        search_space=SearchSpaceConstraints(
            ct_slots=32768,
            initial_ct_count=1,
            stem=StemConstraints(enabled=True, allowed_codes=None),
            second_downsample=SecondDownsampleConstraints(enabled=True, allowed_codes=None),
            blocks=BlockConstraints(allowed_block_ids=None, first_layers_constraints=None),
            stride=StrideConstraints(allowed_block_counts=None, num_strides=3),
            ct_policies=CTPolicyConstraints(allowed=["keep", "half"]),
        ),
        output=OutputConfig(
            base_dir="generated_networks/imagenet_224",
            save_format="json",
        ),
    )


def create_default_cifar10_config() -> GeneratorConfig:
    """创建默认的CIFAR-10配置"""
    return GeneratorConfig(
        name="cifar10_32",
        description="CIFAR-10 with 32x32 input - less aggressive downsampling",
        dataset=DatasetConfig(
            name="cifar10",
            num_classes=10,
            input_size=32,
        ),
        search_space=SearchSpaceConstraints(
            ct_slots=32768,
            initial_ct_count=1,
            stem=StemConstraints(
                enabled=False,  # CIFAR-10不需要激进的stem
                allowed_codes=None,
            ),
            second_downsample=SecondDownsampleConstraints(
                enabled=False,  # 小分辨率不需要第二次降采样
                allowed_codes=None,
            ),
            blocks=BlockConstraints(
                allowed_block_ids=None,
                # 前两层不降分辨率
                first_layers_constraints=[
                    LayerConstraint(position=0, stride=1),
                    LayerConstraint(position=1, stride=1),
                ],
            ),
            stride=StrideConstraints(
                allowed_block_counts=[6, 8, 10, 12],  # CIFAR适合较少的block
                num_strides=2,  # 只降2次分辨率（从32到8）
            ),
            ct_policies=CTPolicyConstraints(allowed=["keep", "half"]),
        ),
        output=OutputConfig(
            base_dir="generated_networks/cifar10_32",
            save_format="json",
        ),
    )


if __name__ == "__main__":
    # 测试配置系统
    print("创建默认ImageNet配置")
    imagenet_cfg = create_default_imagenet_config()
    print(imagenet_cfg.summary())

    print("\n" + "=" * 60)
    print("创建默认CIFAR-10配置")
    cifar10_cfg = create_default_cifar10_config()
    print(cifar10_cfg.summary())

    # 测试保存和加载
    print("\n" + "=" * 60)
    print("测试保存和加载")
    test_path = "/tmp/test_config.yaml"
    imagenet_cfg.save(test_path)
    print(f"配置已保存到: {test_path}")

    loaded_cfg = GeneratorConfig.from_yaml(test_path)
    print("配置已加载")
    print(loaded_cfg.summary())


class ConfigManager:
    """
    配置管理器

    负责：
    - 管理不同配置的输出目录
    - 保存网络配置到相应目录
    - 加载和管理批量配置
    """

    def __init__(self, config: GeneratorConfig):
        """
        Args:
            config: 生成器配置
        """
        self.config = config
        self.output_dir = Path(config.output.base_dir)

    def ensure_output_dir(self):
        """确保输出目录存在"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_config_path(self, config_name: str) -> Path:
        """
        获取单个网络配置的保存路径

        Args:
            config_name: 配置名称

        Returns:
            配置文件路径
        """
        return self.output_dir / f"{config_name}.json"

    def get_batch_path(self, batch_name: str) -> Path:
        """
        获取批量配置的保存路径

        Args:
            batch_name: 批次名称

        Returns:
            批量配置文件路径
        """
        return self.output_dir / f"batch_{batch_name}.json"

    def save_network_config(self, network_config, overwrite: bool = False):
        """
        保存单个网络配置

        Args:
            network_config: NetworkConfig实例
            overwrite: 是否覆盖已存在的文件
        """
        from .network_config import NetworkConfig

        if not isinstance(network_config, NetworkConfig):
            raise TypeError("network_config must be a NetworkConfig instance")

        self.ensure_output_dir()
        path = self.get_config_path(network_config.name)

        if path.exists() and not overwrite:
            raise FileExistsError(f"Config file already exists: {path}")

        network_config.save(str(path))
        return path

    def save_batch(self, batch, overwrite: bool = False):
        """
        保存批量配置

        Args:
            batch: NetworkConfigBatch实例
            overwrite: 是否覆盖已存在的文件

        Returns:
            保存路径
        """
        from .network_config import NetworkConfigBatch

        if not isinstance(batch, NetworkConfigBatch):
            raise TypeError("batch must be a NetworkConfigBatch instance")

        self.ensure_output_dir()
        path = self.get_batch_path(batch.batch_name)

        if path.exists() and not overwrite:
            raise FileExistsError(f"Batch file already exists: {path}")

        batch.save(str(path))
        return path

    def list_configs(self) -> List[str]:
        """
        列出所有已保存的网络配置

        Returns:
            配置名称列表
        """
        if not self.output_dir.exists():
            return []

        config_files = self.output_dir.glob("*.json")
        # 排除batch文件
        return [f.stem for f in config_files if not f.stem.startswith("batch_")]

    def list_batches(self) -> List[str]:
        """
        列出所有已保存的批量配置

        Returns:
            批次名称列表
        """
        if not self.output_dir.exists():
            return []

        batch_files = self.output_dir.glob("batch_*.json")
        # 移除"batch_"前缀
        return [f.stem.replace("batch_", "") for f in batch_files]

    def load_network_config(self, config_name: str):
        """
        加载单个网络配置

        Args:
            config_name: 配置名称

        Returns:
            NetworkConfig实例
        """
        from .network_config import NetworkConfig

        path = self.get_config_path(config_name)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        return NetworkConfig.load(str(path))

    def load_batch(self, batch_name: str):
        """
        加载批量配置

        Args:
            batch_name: 批次名称

        Returns:
            NetworkConfigBatch实例
        """
        from .network_config import NetworkConfigBatch

        path = self.get_batch_path(batch_name)
        if not path.exists():
            raise FileNotFoundError(f"Batch file not found: {path}")

        return NetworkConfigBatch.load(str(path))

    def summary(self) -> str:
        """生成配置管理器摘要"""
        lines = [
            "=" * 60,
            f"配置管理器: {self.config.name}",
            "=" * 60,
            f"输出目录: {self.output_dir}",
            f"目录存在: {self.output_dir.exists()}",
        ]

        if self.output_dir.exists():
            configs = self.list_configs()
            batches = self.list_batches()
            lines.extend([
                f"\n已保存的网络配置: {len(configs)}",
                f"已保存的批量配置: {len(batches)}",
            ])

            if configs:
                lines.append("\n网络配置:")
                for name in configs[:5]:  # 只显示前5个
                    lines.append(f"  - {name}")
                if len(configs) > 5:
                    lines.append(f"  ... 还有 {len(configs) - 5} 个")

            if batches:
                lines.append("\n批量配置:")
                for name in batches:
                    lines.append(f"  - {name}")

        lines.append("=" * 60)
        return "\n".join(lines)
