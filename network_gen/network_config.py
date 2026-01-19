"""
网络配置类

定义网络架构的完整配置，支持序列化和反序列化。
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .search_space import (
    SearchSpace,
    StrideEncoder,
    ChannelCalculator,
    UNIFIED_BLOCKS,
    STEM_CONFIGS,
    SECOND_DOWNSAMPLE_CONFIGS,
    StemConfig,
    SecondDownsampleConfig,
    UnifiedBlockSpec,
)


@dataclass
class BlockConfig:
    """单个Block的配置"""
    block_id: int            # 统一Block ID [0-23]
    in_channels: int
    out_channels: int
    stride: int

    @property
    def spec(self) -> UnifiedBlockSpec:
        """获取Block规格"""
        return UNIFIED_BLOCKS[self.block_id]

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def block_class(self):
        return self.spec.block_class

    @property
    def activation_class(self):
        return self.spec.activation_class

    @property
    def factor(self) -> Optional[float]:
        return self.spec.factor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "stride": self.stride,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BlockConfig":
        return cls(**d)


@dataclass
class NetworkConfig:
    """
    完整的网络配置

    包含：
    - stem配置
    - 第二次降分辨率配置
    - body部分的所有block配置
    - 分层选择信息
    - 元信息（用于追踪和复现）
    """

    # Stem层配置编码 [0-3]
    stem_code: int

    # 第二次降分辨率配置编码 [0-5]
    second_ds_code: int

    # Body stride编码 [0-1343]
    stride_code: int

    # CT策略列表（3个位置）
    ct_policies: List[str]

    # Block choices per block (length == num_blocks)
    block_choices: List[int]

    # Each block config (matches block_choices)
    blocks: List[BlockConfig]

    # 初始CT数量
    initial_ct_count: int = 1

    # 输入通道数（stem输出通道数）
    stem_out_channels: int = 64

    # 分类数
    num_classes: int = 100

    # 元信息
    name: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if self.blocks:
            block_ids = [b.block_id for b in self.blocks]
            if self.block_choices != block_ids:
                self.block_choices = block_ids
        if self.name is None:
            self.name = self.generate_name()

    def generate_name(self) -> str:
        """生成配置的唯一名称"""
        config_str = f"{self.stem_code}_{self.second_ds_code}_{self.stride_code}"
        config_str += f"_{''.join(self.ct_policies)}"
        config_str += f"_{'_'.join(map(str, self.block_choices))}"
        hash_val = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"net_{hash_val}"

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def num_choices(self) -> int:
        """Number of block choices (per block)."""
        return len(self.block_choices)

    @property
    def stem_config(self) -> StemConfig:
        return STEM_CONFIGS[self.stem_code]

    @property
    def second_ds_config(self) -> SecondDownsampleConfig:
        return SECOND_DOWNSAMPLE_CONFIGS[self.second_ds_code]

    def get_stride_positions(self) -> List[int]:
        """获取stride=2的位置"""
        return [i for i, b in enumerate(self.blocks) if b.stride == 2]

    def summary(self) -> str:
        """生成配置摘要"""
        lines = [
            "=" * 60,
            f"网络配置: {self.name}",
            "=" * 60,
            f"Stem: {self.stem_config}",
            f"第二次降分辨率: {self.second_ds_config}",
            f"Block数量: {self.num_blocks}",
            f"Block choices: {self.num_choices}",
            f"Stride位置: {self.get_stride_positions()}",
            f"CT策略: {self.ct_policies}",
            "",
            f"Block选择: {self.block_choices}",
            "",
            "Blocks:",
        ]

        for i, block in enumerate(self.blocks):
            stride_mark = " [S2]" if block.stride == 2 else ""
            lines.append(
                f"  [{i:2d}] {block.name:25s} "
                f"in={block.in_channels:4d} out={block.out_channels:4d}{stride_mark}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "stem_code": self.stem_code,
            "second_ds_code": self.second_ds_code,
            "stride_code": self.stride_code,
            "ct_policies": self.ct_policies,
            "block_choices": self.block_choices,
            "blocks": [b.to_dict() for b in self.blocks],
            "initial_ct_count": self.initial_ct_count,
            "stem_out_channels": self.stem_out_channels,
            "num_classes": self.num_classes,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NetworkConfig":
        """从字典创建"""
        blocks = [BlockConfig.from_dict(b) for b in d["blocks"]]
        return cls(
            stem_code=d["stem_code"],
            second_ds_code=d["second_ds_code"],
            stride_code=d["stride_code"],
            ct_policies=d["ct_policies"],
            block_choices=d["block_choices"],
            blocks=blocks,
            initial_ct_count=d.get("initial_ct_count", 1),
            stem_out_channels=d.get("stem_out_channels", 64),
            num_classes=d.get("num_classes", 100),
            name=d.get("name"),
            description=d.get("description"),
            created_at=d.get("created_at"),
        )

    def save(self, path: str):
        """保存配置到JSON文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "NetworkConfig":
        """从JSON文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)


@dataclass
class NetworkConfigBatch:
    """批量网络配置"""
    configs: List[NetworkConfig]
    batch_name: str = "batch"
    description: str = ""

    def __len__(self):
        return len(self.configs)

    def __iter__(self):
        return iter(self.configs)

    def __getitem__(self, idx):
        return self.configs[idx]

    def save(self, path: str):
        """保存批量配置"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "batch_name": self.batch_name,
            "description": self.description,
            "num_configs": len(self.configs),
            "configs": [c.to_dict() for c in self.configs],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "NetworkConfigBatch":
        """加载批量配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        configs = [NetworkConfig.from_dict(c) for c in data["configs"]]
        return cls(
            configs=configs,
            batch_name=data.get("batch_name", "batch"),
            description=data.get("description", ""),
        )

    def summary(self) -> str:
        """批量配置摘要"""
        lines = [
            "=" * 60,
            f"批量配置: {self.batch_name}",
            f"描述: {self.description}",
            f"配置数量: {len(self.configs)}",
            "=" * 60,
        ]

        # 统计
        block_count_dist = {}
        stem_dist = {}
        second_ds_dist = {}

        for cfg in self.configs:
            n = cfg.num_blocks
            block_count_dist[n] = block_count_dist.get(n, 0) + 1
            stem_dist[cfg.stem_code] = stem_dist.get(cfg.stem_code, 0) + 1
            second_ds_dist[cfg.second_ds_code] = second_ds_dist.get(cfg.second_ds_code, 0) + 1

        lines.append("\nBlock数量分布:")
        for n in sorted(block_count_dist.keys()):
            lines.append(f"  {n} blocks: {block_count_dist[n]}")

        lines.append("\nStem配置分布:")
        for code in sorted(stem_dist.keys()):
            lines.append(f"  [{code}] {STEM_CONFIGS[code]}: {stem_dist[code]}")

        lines.append("\n第二次降分辨率分布:")
        for code in sorted(second_ds_dist.keys()):
            lines.append(f"  [{code}] {SECOND_DOWNSAMPLE_CONFIGS[code]}: {second_ds_dist[code]}")

        lines.append("=" * 60)
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    from .search_space import UNIFIED_BLOCKS

    print("24种Block定义:")
    for i, spec in UNIFIED_BLOCKS.items():
        print(f"  [{i:2d}] {spec.name}: {spec.description}")
