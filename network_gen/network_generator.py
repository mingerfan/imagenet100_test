"""
网络生成器

支持随机生成网络配置，并可以将配置转换为PyTorch模型。
使用分层选择策略：前4个block单独选择，后面每2个block共享选择。
"""

import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Type
import sys
import os

import torch
import torch.nn as nn

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .search_space import (
    SearchSpace,
    StrideEncoder,
    ChannelCalculator,
    UNIFIED_BLOCKS,
    STEM_CONFIGS,
    SECOND_DOWNSAMPLE_CONFIGS,
    UnifiedBlockSpec,
)
from .network_config import NetworkConfig, BlockConfig, NetworkConfigBatch
from .generator_config import GeneratorConfig

from models.gate_net_cmp.block_def import (
    BasicBlock,
    BasicSelfGatedBlock,
    FullGatedBasicBlock,
    MBConvBlock,
    SelfGated,
)


class HierarchicalBlockSelector:
    """
    分层Block选择器

    选择策略：
    - 前4个block：每个单独选择（重要位置，细粒度控制）
    - 后面的block：每2个共享选择（降低搜索空间）
    """

    NUM_INDIVIDUAL = 4  # 前4个单独选择
    GROUP_SIZE = 2       # 后面每2个一组
    NUM_BLOCK_TYPES = 22  # 22种统一Block

    def __init__(self):
        pass

    def compute_num_choices(self, num_blocks: int) -> int:
        """
        计算需要的选择位数量

        Args:
            num_blocks: block总数

        Returns:
            选择位数量
        """
        individual = min(self.NUM_INDIVIDUAL, num_blocks)
        remaining = max(0, num_blocks - self.NUM_INDIVIDUAL)
        grouped = (remaining + self.GROUP_SIZE - 1) // self.GROUP_SIZE
        return individual + grouped

    def expand_choices(self, choices: List[int], num_blocks: int) -> List[int]:
        """
        将选择位展开为每个block的选择

        Args:
            choices: 选择位列表（前4个单独，后面每个代表2个block）
            num_blocks: 目标block数量

        Returns:
            每个block的block_id列表
        """
        block_ids = []

        # 前4个单独选择
        for i in range(min(self.NUM_INDIVIDUAL, num_blocks)):
            if i < len(choices):
                block_ids.append(choices[i])
            else:
                block_ids.append(0)  # 默认值

        # 后面每2个共享选择
        remaining = num_blocks - self.NUM_INDIVIDUAL
        if remaining > 0:
            choice_idx = self.NUM_INDIVIDUAL
            for i in range(remaining):
                group_idx = i // self.GROUP_SIZE
                actual_choice_idx = self.NUM_INDIVIDUAL + group_idx
                if actual_choice_idx < len(choices):
                    block_ids.append(choices[actual_choice_idx])
                else:
                    block_ids.append(0)

        return block_ids

    def generate_random_choices(self, num_blocks: int) -> List[int]:
        """
        随机生成选择位

        Args:
            num_blocks: block总数

        Returns:
            随机选择位列表
        """
        num_choices = self.compute_num_choices(num_blocks)
        return [random.randint(0, self.NUM_BLOCK_TYPES - 1) for _ in range(num_choices)]


class RandomNetworkGenerator:
    """
    随机网络生成器

    根据搜索空间定义和配置约束随机生成网络配置。
    使用分层选择策略控制搜索空间大小。
    支持通过GeneratorConfig应用约束。
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        config: Optional[GeneratorConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            search_space: 搜索空间定义，默认使用标准配置
            config: 生成器配置（包含约束），如果提供则优先使用
            seed: 随机种子
        """
        self.config = config

        # 如果提供了config，从config创建search_space
        if config is not None:
            self.search_space = SearchSpace(
                ct_slots=config.search_space.ct_slots,
                input_size=config.dataset.input_size,
                initial_ct_count=config.search_space.initial_ct_count,
            )
        else:
            self.search_space = search_space or SearchSpace()

        self.stride_encoder = StrideEncoder()
        self.channel_calculator = ChannelCalculator(
            ct_slots=self.search_space.ct_slots,
            input_size=self.search_space.input_size,
        )
        self.block_selector = HierarchicalBlockSelector()

        if seed is not None:
            random.seed(seed)

    def generate_random_config(
        self,
        stem_code: Optional[int] = None,
        second_ds_code: Optional[int] = None,
        stride_code: Optional[int] = None,
        block_choices: Optional[List[int]] = None,
        ct_policies: Optional[List[str]] = None,
    ) -> NetworkConfig:
        """
        生成随机网络配置（应用配置约束）

        Args:
            stem_code: 指定stem配置 [0-3]，默认随机（受约束）
            second_ds_code: 指定第二次降分辨率配置 [0-5]，默认随机（受约束）
            stride_code: 指定stride编码 [0-1343]，默认随机（受约束）
            block_choices: 分层Block选择，默认随机（受约束）
            ct_policies: CT策略列表，默认随机（受约束）

        Returns:
            NetworkConfig
        """
        # 1. 选择stem配置（应用约束）
        if stem_code is None:
            stem_code = self._random_stem_code()

        # 2. 选择第二次降分辨率配置（应用约束）
        if second_ds_code is None:
            second_ds_code = self._random_second_ds_code()

        # 3. 选择stride编码（决定block数量和stride位置）（应用约束）
        if stride_code is None:
            stride_code = self._random_stride_code()

        # 解码stride信息
        num_blocks, stride_positions = self.stride_encoder.decode(stride_code)
        strides = self.stride_encoder.get_strides_list(num_blocks, stride_positions)

        # 应用first_layers_constraints（强制某些位置的stride）
        strides = self._apply_stride_constraints(strides)

        # 4. 选择CT策略（应用约束）
        if ct_policies is None:
            ct_policies = self._random_ct_policies()

        # 5. 生成分层Block选择（应用约束）
        if block_choices is None:
            block_choices = self._random_block_choices(num_blocks)

        # 展开为每个block的选择
        block_ids = self.block_selector.expand_choices(block_choices, num_blocks)

        # 应用位置特定的block约束
        block_ids = self._apply_block_constraints(block_ids)

        # 6. 选择初始CT数量并计算通道数
        initial_ct_count = self._random_initial_ct_count()
        channels, feature_sizes, ct_counts = self.channel_calculator.compute_channels_sequence(
            strides=strides,
            ct_policies=ct_policies,
            initial_ct_count=initial_ct_count,
        )

        # 7. 构建block配置列表
        blocks = []
        stem_out_channels = self.channel_calculator.get_initial_channels(
            initial_ct_count
        )

        for i in range(num_blocks):
            # 输入通道数
            if i == 0:
                in_channels = stem_out_channels
            else:
                in_channels = blocks[-1].out_channels

            # 输出通道数
            out_channels = channels[i]

            block_cfg = BlockConfig(
                block_id=block_ids[i],
                in_channels=in_channels,
                out_channels=out_channels,
                stride=strides[i],
            )
            blocks.append(block_cfg)

        # 8. 创建网络配置
        config = NetworkConfig(
            stem_code=stem_code,
            second_ds_code=second_ds_code,
            stride_code=stride_code,
            ct_policies=ct_policies,
            block_choices=block_choices,
            blocks=blocks,
            initial_ct_count=initial_ct_count,
            stem_out_channels=stem_out_channels,
            created_at=datetime.now().isoformat(),
        )

        return config

    def generate_batch(
        self,
        num_configs: int,
        batch_name: str = "random_batch",
        description: str = "",
        unique: bool = True,
    ) -> NetworkConfigBatch:
        """
        批量生成网络配置

        Args:
            num_configs: 生成数量
            batch_name: 批次名称
            description: 描述
            unique: 是否确保配置唯一

        Returns:
            NetworkConfigBatch
        """
        configs = []
        seen_hashes = set()

        attempts = 0
        max_attempts = num_configs * 10

        while len(configs) < num_configs and attempts < max_attempts:
            config = self.generate_random_config()
            attempts += 1

            if unique:
                config_hash = config.name
                if config_hash in seen_hashes:
                    continue
                seen_hashes.add(config_hash)

            configs.append(config)

        if len(configs) < num_configs:
            print(f"警告: 只生成了 {len(configs)}/{num_configs} 个唯一配置")

        return NetworkConfigBatch(
            configs=configs,
            batch_name=batch_name,
            description=description or f"随机生成的 {len(configs)} 个网络配置",
        )

    # ========== 约束应用辅助方法 ==========

    def _resolve_initial_ct_count_range(self) -> Tuple[int, int]:
        """计算初始CT数量范围（结合最小通道数约束）"""
        min_channels = 16
        max_channels = 64
        min_ct_override = None
        max_ct_override = None
        legacy_ct = None

        if self.config is not None:
            min_channels = getattr(self.config.search_space, "initial_min_channels", min_channels)
            max_channels = getattr(self.config.search_space, "initial_max_channels", max_channels)
            min_ct_override = getattr(self.config.search_space, "initial_ct_count_min", None)
            max_ct_override = getattr(self.config.search_space, "initial_ct_count_max", None)
            legacy_ct = getattr(self.config.search_space, "initial_ct_count", None)
        else:
            legacy_ct = getattr(self.search_space, "initial_ct_count", None)

        feature_size = self.channel_calculator.feature_size_after_stem
        min_required_ct = self.channel_calculator.compute_ct_from_channels(
            min_channels,
            feature_size,
        )

        if min_ct_override is not None:
            min_ct = max(min_required_ct, min_ct_override)
        else:
            min_ct = min_required_ct

        if max_ct_override is not None:
            max_ct = max(max_ct_override, min_ct)
        elif max_channels is not None:
            max_from_channels = self.channel_calculator.compute_ct_from_channels(
                max_channels,
                feature_size,
            )
            max_ct = max(max_from_channels, min_ct)
        elif legacy_ct is not None:
            max_ct = max(legacy_ct, min_ct)
        else:
            max_ct = min_ct

        return min_ct, max_ct

    def _random_initial_ct_count(self) -> int:
        """随机选择初始CT数量（满足最小通道数约束）"""
        min_ct, max_ct = self._resolve_initial_ct_count_range()
        if min_ct == max_ct:
            return min_ct
        return random.randint(min_ct, max_ct)

    def _random_stem_code(self) -> int:
        """随机选择stem配置（应用约束）"""
        if self.config is None:
            return random.randint(0, self.search_space.num_stem_configs - 1)

        # 应用约束
        allowed_codes = self.config.search_space.stem.allowed_codes
        if allowed_codes is None:
            allowed_codes = list(range(self.search_space.num_stem_configs))

        if not allowed_codes:
            raise ValueError("No allowed stem codes in config")

        return random.choice(allowed_codes)

    def _random_second_ds_code(self) -> int:
        """随机选择第二次降分辨率配置（应用约束）"""
        if self.config is None:
            return random.randint(0, self.search_space.num_second_ds_configs - 1)

        # 应用约束
        if not self.config.search_space.second_downsample.enabled:
            # 如果禁用，返回默认值（这里可能需要特殊处理）
            return 0  # AvgPool

        allowed_codes = self.config.search_space.second_downsample.allowed_codes
        if allowed_codes is None:
            allowed_codes = list(range(self.search_space.num_second_ds_configs))

        if not allowed_codes:
            raise ValueError("No allowed second downsample codes in config")

        return random.choice(allowed_codes)

    def _random_stride_code(self) -> int:
        """随机选择stride编码（应用约束）"""
        if self.config is None:
            return random.randint(0, self.search_space.num_stride_configs - 1)

        # 应用约束：只选择允许的block数量
        allowed_counts = self.config.search_space.stride.allowed_block_counts
        if allowed_counts is None:
            allowed_counts = self.stride_encoder.BLOCK_COUNTS

        # 收集满足条件的stride codes
        valid_codes = []
        for code in range(self.stride_encoder.total_combinations):
            num_blocks, _ = self.stride_encoder.decode(code)
            if num_blocks in allowed_counts:
                valid_codes.append(code)

        if not valid_codes:
            raise ValueError("No valid stride codes match the allowed block counts")

        return random.choice(valid_codes)

    def _random_ct_policies(self) -> List[str]:
        """随机选择CT策略（应用约束）"""
        if self.config is None:
            return [
                random.choice(self.search_space.ct_policy_options)
                for _ in range(3)
            ]

        # 应用约束
        allowed_policies = self.config.search_space.ct_policies.allowed
        if not allowed_policies:
            raise ValueError("No allowed CT policies in config")

        return [random.choice(allowed_policies) for _ in range(3)]

    def _random_block_choices(self, num_blocks: int) -> List[int]:
        """随机生成block选择（应用约束）"""
        num_choices = self.block_selector.compute_num_choices(num_blocks)

        if self.config is None:
            # 无约束，使用全部22种block
            return [random.randint(0, self.block_selector.NUM_BLOCK_TYPES - 1) for _ in range(num_choices)]

        # 应用全局block约束
        allowed_ids = self.config.search_space.blocks.allowed_block_ids
        if allowed_ids is None:
            allowed_ids = list(range(self.block_selector.NUM_BLOCK_TYPES))

        if not allowed_ids:
            raise ValueError("No allowed block IDs in config")

        return [random.choice(allowed_ids) for _ in range(num_choices)]

    def _apply_stride_constraints(self, strides: List[int]) -> List[int]:
        """应用stride约束（first_layers_constraints）"""
        if self.config is None:
            return strides

        constraints = self.config.search_space.blocks.first_layers_constraints
        if not constraints:
            return strides

        # 复制strides以避免修改原始列表
        strides = strides.copy()

        # 应用每个约束
        for constraint in constraints:
            if constraint.position < len(strides) and constraint.stride is not None:
                strides[constraint.position] = constraint.stride

        return strides

    def _apply_block_constraints(self, block_ids: List[int]) -> List[int]:
        """应用block约束（位置特定的约束）"""
        if self.config is None:
            return block_ids

        constraints = self.config.search_space.blocks.first_layers_constraints
        if not constraints:
            return block_ids

        # 复制block_ids以避免修改原始列表
        block_ids = block_ids.copy()

        # 应用每个约束
        for constraint in constraints:
            if constraint.position < len(block_ids) and constraint.allowed_block_ids is not None:
                # 如果当前block不在允许列表中，随机选择一个允许的block
                if block_ids[constraint.position] not in constraint.allowed_block_ids:
                    block_ids[constraint.position] = random.choice(constraint.allowed_block_ids)

        return block_ids


class NetworkBuilder:
    """
    网络构建器

    将NetworkConfig转换为PyTorch模型。
    """

    def __init__(self):
        pass

    def build_stem(self, config: NetworkConfig) -> nn.Module:
        """构建Stem层"""
        stem_cfg = config.stem_config
        activation_class = stem_cfg.get_activation_class()

        layers = []

        # 7x7 Conv, stride=2
        layers.append(nn.Conv2d(3, config.stem_out_channels, kernel_size=7, stride=2, padding=3, bias=False))
        layers.append(nn.BatchNorm2d(config.stem_out_channels))

        # 可选的SelfGate
        if stem_cfg.use_selfgate:
            layers.append(SelfGated(
                config.stem_out_channels,
                config.stem_out_channels,
                stride=1,
                activation=activation_class,
            ))
        else:
            layers.append(activation_class())

        # MaxPool, stride=2
        # layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        return nn.Sequential(*layers)

    def build_second_downsample(
        self,
        config: NetworkConfig,
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        """构建第二次降分辨率层"""
        ds_cfg = config.second_ds_config

        if ds_cfg.type == "avepool":
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif ds_cfg.type == "none":
            # 不使用第二次降分辨率层（像EfficientNet B0）
            return nn.Identity()
        else:
            activation_class = ds_cfg.get_activation_class()
            layers = []

            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))

            if ds_cfg.use_selfgate:
                layers.append(SelfGated(
                    out_channels,
                    out_channels,
                    stride=1,
                    activation=activation_class,
                ))
            else:
                layers.append(activation_class())

            return nn.Sequential(*layers)

    def build_block(self, block_cfg: BlockConfig) -> nn.Module:
        """构建单个Block"""
        spec = block_cfg.spec
        block_class = spec.block_class
        activation_class = spec.activation_class

        if spec.is_mbconv():
            # MBConv类型（包含MBConv和GatedMBConv）
            return block_class(
                in_channels=block_cfg.in_channels,
                out_channels=block_cfg.out_channels,
                stride=block_cfg.stride,
                expansion_factor=spec.expansion,
                activation=activation_class,
                use_se=spec.use_se,
                use_gated_dw=spec.use_gated_dw,
            )
        elif block_class == BasicSelfGatedBlock:
            # BasicSelfGatedBlock
            return block_class(
                in_channels=block_cfg.in_channels,
                out_channels=block_cfg.out_channels,
                stride=block_cfg.stride,
                activation=activation_class,
                full_gated=False,
            )
        elif block_class == FullGatedBasicBlock:
            # FullGatedBasicBlock（新增）
            return block_class(
                in_channels=block_cfg.in_channels,
                out_channels=block_cfg.out_channels,
                stride=block_cfg.stride,
                activation=activation_class,
            )
        else:
            # BasicBlock
            return block_class(
                in_channels=block_cfg.in_channels,
                out_channels=block_cfg.out_channels,
                stride=block_cfg.stride,
                activation=activation_class,
            )

    def build(self, config: NetworkConfig) -> nn.Module:
        """
        根据配置构建完整网络

        Args:
            config: 网络配置

        Returns:
            nn.Module
        """
        return GeneratedNetwork(config, self)


class GeneratedNetwork(nn.Module):
    """由NetworkConfig生成的网络"""

    def __init__(self, config: NetworkConfig, builder: NetworkBuilder):
        super().__init__()
        self.config = config

        # 构建Stem
        self.stem = builder.build_stem(config)

        self.second = builder.build_second_downsample(config, config.stem_out_channels, config.stem_out_channels)

        # 构建Body（所有blocks）
        self.blocks = nn.ModuleList()
        for block_cfg in config.blocks:
            self.blocks.append(builder.build_block(block_cfg))

        # 输出层
        final_channels = config.blocks[-1].out_channels
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_channels, config.num_classes)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        x = self.second(x)

        # Body blocks
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_config(self) -> NetworkConfig:
        """返回网络配置"""
        return self.config


def create_network(config: NetworkConfig) -> nn.Module:
    """便捷函数：从配置创建网络"""
    builder = NetworkBuilder()
    return builder.build(config)


def create_random_network(
    seed: Optional[int] = None,
    search_space: Optional[SearchSpace] = None,
) -> Tuple[nn.Module, NetworkConfig]:
    """
    便捷函数：随机生成一个网络

    Returns:
        (model, config)
    """
    generator = RandomNetworkGenerator(search_space=search_space, seed=seed)
    config = generator.generate_random_config()
    model = create_network(config)
    return model, config


if __name__ == "__main__":
    # 测试
    print("测试随机网络生成器（分层选择）")
    print("=" * 60)

    generator = RandomNetworkGenerator(seed=42)

    # 生成单个配置
    config = generator.generate_random_config()
    print(config.summary())

    # 测试分层选择器
    print("\n分层选择器测试:")
    selector = HierarchicalBlockSelector()
    for num_blocks in [4, 6, 8, 10, 12, 14, 16]:
        num_choices = selector.compute_num_choices(num_blocks)
        choices = list(range(num_choices))  # 使用0,1,2,...作为测试
        expanded = selector.expand_choices(choices, num_blocks)
        print(f"  {num_blocks:2d} blocks: {num_choices} choices -> {expanded}")

    # 构建网络
    print("\n构建网络...")
    model = create_network(config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    print("\n测试前向传播...")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {y.shape}")

    # 生成批量配置
    print("\n" + "=" * 60)
    print("生成批量配置")
    batch = generator.generate_batch(5, batch_name="test_batch")
    print(batch.summary())
