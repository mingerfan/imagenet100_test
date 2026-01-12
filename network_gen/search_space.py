"""
FHE-NAS 搜索空间定义

设计原则：
1. Block数量: 4, 6, 8, 10, 12, 14, 16（不含stem）
2. 降分辨率位置: 固定3个stride=2的位置，使用组合数编码
3. 通道数: 根据CT（密文槽位）和特征图大小反向计算
4. Stem层: 2×2=4种选择（selfgate与否 × 激活函数）
5. 第二次降分辨率: 5种选择（avepool 或 4种conv组合）
6. 24种预定义Block: 将block类型、激活函数、factor组合成24种独立选择
7. 分层选择策略: 前4个block单独选，后面每2个block共享选择

Stride编码（body部分）:
- 从n个block中选择3个位置进行降分辨率（stride=2）
- C(4,3)=4, C(6,3)=20, C(8,3)=56, C(10,3)=120, C(12,3)=220, C(14,3)=364, C(16,3)=560
- 总共 1344 种组合

搜索空间大小估算:
- Block选择: 24^(4+6) = 24^10 ≈ 6.3e+13
- Stem × SecondDS × Stride × CT策略 = 4 × 5 × 1344 × 8 ≈ 2.2e+5
- 总计: ~1.4e+19 (完整) 或 ~6.3e+13 (仅block选择)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Type, Tuple, Optional
from itertools import combinations
from math import comb
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gate_net_cmp.block_def import (
    BasicBlock,
    BottleneckBlock,
    BasicSelfGatedBlock,
    BottleneckSelfGatedBlock,
    LearnableSwish,
    StablePoly4,
    Swish,
)


# ============ 激活函数定义 ============
@dataclass
class ActivationSpec:
    """激活函数规格定义"""
    name: str
    activation_class: Type
    trainable: bool = False
    description: str = ""


ACTIVATION_TYPES: Dict[str, ActivationSpec] = {
    "swish": ActivationSpec(
        name="swish",
        activation_class=Swish,
        description="Swish/SiLU激活"
    ),
    "learnable_swish": ActivationSpec(
        name="learnable_swish",
        activation_class=LearnableSwish,
        trainable=True,
        description="可学习Swish"
    ),
    "poly4": ActivationSpec(
        name="poly4",
        activation_class=StablePoly4,
        trainable=True,
        description="4阶多项式激活"
    ),
}


# ============ 24种预定义Block ============
@dataclass
class UnifiedBlockSpec:
    """
    统一的Block规格定义

    将block类型、激活函数、factor组合成一个独立的选择单元
    """
    id: int                  # 0-23的唯一ID
    name: str                # 可读名称
    block_class: Type        # Block类
    activation_class: Type   # 激活函数类
    factor: Optional[float]  # Bottleneck的factor（非bottleneck为None）
    description: str = ""

    def is_bottleneck(self) -> bool:
        return self.factor is not None


def _create_unified_blocks() -> Dict[int, UnifiedBlockSpec]:
    """
    创建24种预定义Block

    组合规则：
    - basic × 2激活 = 2种
    - basic_self_gated × 2激活 = 2种
    - bottleneck × 2激活 × 5factor = 10种
    - bottleneck_self_gated × 2激活 × 5factor = 10种
    总计: 2 + 2 + 10 + 10 = 24种
    """
    blocks = {}
    block_id = 0

    activations = [
        ("poly4", StablePoly4),
        ("swish", Swish),
    ]

    # Factor选项（5种，用于bottleneck）
    factors = [0.25, 0.5, 1.0, 1.5, 2.0]

    # Basic blocks (2种)
    for act_name, act_class in activations:
        blocks[block_id] = UnifiedBlockSpec(
            id=block_id,
            name=f"basic_{act_name}",
            block_class=BasicBlock,
            activation_class=act_class,
            factor=None,
            description=f"BasicBlock + {act_name}"
        )
        block_id += 1

    # Basic Self-Gated blocks (2种)
    for act_name, act_class in activations:
        blocks[block_id] = UnifiedBlockSpec(
            id=block_id,
            name=f"basic_sg_{act_name}",
            block_class=BasicSelfGatedBlock,
            activation_class=act_class,
            factor=None,
            description=f"BasicSelfGated + {act_name}"
        )
        block_id += 1

    # Bottleneck blocks (10种)
    for factor in factors:
        for act_name, act_class in activations:
            blocks[block_id] = UnifiedBlockSpec(
                id=block_id,
                name=f"btn_f{factor}_{act_name}",
                block_class=BottleneckBlock,
                activation_class=act_class,
                factor=factor,
                description=f"Bottleneck(f={factor}) + {act_name}"
            )
            block_id += 1

    # Bottleneck Self-Gated blocks (10种)
    for factor in factors:
        for act_name, act_class in activations:
            blocks[block_id] = UnifiedBlockSpec(
                id=block_id,
                name=f"btn_sg_f{factor}_{act_name}",
                block_class=BottleneckSelfGatedBlock,
                activation_class=act_class,
                factor=factor,
                description=f"BottleneckSelfGated(f={factor}) + {act_name}"
            )
            block_id += 1

    assert len(blocks) == 24, f"Expected 24 blocks, got {len(blocks)}"
    return blocks


# 全局的24种Block定义
UNIFIED_BLOCKS: Dict[int, UnifiedBlockSpec] = _create_unified_blocks()


# 为了兼容性保留的旧定义
# @dataclass
# class BlockSpec:
#     """Block规格定义（兼容旧代码）"""
#     name: str
#     block_class: Type
#     requires_factor: bool = False
#     supports_full_gated: bool = False
#     description: str = ""


# BLOCK_TYPES: Dict[str, BlockSpec] = {
#     "basic": BlockSpec(
#         name="basic",
#         block_class=BasicBlock,
#         description="标准BasicBlock，两个3x3卷积"
#     ),
#     "bottleneck": BlockSpec(
#         name="bottleneck",
#         block_class=BottleneckBlock,
#         requires_factor=True,
#         description="Bottleneck结构，1x1->3x3->1x1"
#     ),
#     "basic_self_gated": BlockSpec(
#         name="basic_self_gated",
#         block_class=BasicSelfGatedBlock,
#         supports_full_gated=True,
#         description="自门控BasicBlock"
#     ),
#     "bottleneck_self_gated": BlockSpec(
#         name="bottleneck_self_gated",
#         block_class=BottleneckSelfGatedBlock,
#         requires_factor=True,
#         description="自门控Bottleneck"
#     ),
# }


# ============ Stem层选项 ============
@dataclass
class StemConfig:
    """
    Stem层配置

    Stem层结构: Conv(7x7, stride=2) + BN + Act + Pool(3x3, stride=2)
    选项:
    - use_selfgate: 是否在conv后使用SelfGate
    - activation: 激活函数类型 ("poly4" 或 "swish")

    共 2×2=4 种组合
    """
    use_selfgate: bool
    activation: str  # "poly4" or "swish"

    def get_activation_class(self) -> Type:
        return ACTIVATION_TYPES[self.activation].activation_class

    def __repr__(self):
        sg = "SG" if self.use_selfgate else "NoSG"
        return f"Stem({sg}, {self.activation})"


# 所有可能的Stem配置（编码0-3）
STEM_CONFIGS = [
    StemConfig(use_selfgate=False, activation="poly4"),   # 0
    StemConfig(use_selfgate=False, activation="swish"),   # 1
    StemConfig(use_selfgate=True, activation="poly4"),    # 2
    StemConfig(use_selfgate=True, activation="swish"),    # 3
]


# ============ 第二次降分辨率选项 ============
@dataclass
class SecondDownsampleConfig:
    """
    第二次降分辨率配置

    选项:
    - type: "avepool" 或 "conv"
    - 若为conv，则有与stem相同的4种选择

    共 1+4=5 种组合
    """
    type: str  # "avepool" or "conv"
    use_selfgate: bool = False  # 仅当type="conv"时有效
    activation: str = "poly4"   # 仅当type="conv"时有效

    def get_activation_class(self) -> Optional[Type]:
        if self.type == "avepool":
            return None
        return ACTIVATION_TYPES[self.activation].activation_class

    def __repr__(self):
        if self.type == "avepool":
            return "SecondDS(AvgPool)"
        sg = "SG" if self.use_selfgate else "NoSG"
        return f"SecondDS(Conv, {sg}, {self.activation})"


# 所有可能的第二次降分辨率配置（编码0-4）
SECOND_DOWNSAMPLE_CONFIGS = [
    SecondDownsampleConfig(type="avepool"),                                    # 0
    SecondDownsampleConfig(type="conv", use_selfgate=False, activation="poly4"),  # 1
    SecondDownsampleConfig(type="conv", use_selfgate=False, activation="swish"),  # 2
    SecondDownsampleConfig(type="conv", use_selfgate=True, activation="poly4"),   # 3
    SecondDownsampleConfig(type="conv", use_selfgate=True, activation="swish"),   # 4
]


# ============ Stride 编码系统 ============
class StrideEncoder:
    """
    Body部分的Stride位置编码器

    使用组合数编码网络结构：
    - block数量: 4, 6, 8, 10, 12, 14, 16
    - 每个网络固定3个stride=2位置
    - 编码范围: [0, 1343]
    """

    BLOCK_COUNTS = [4, 6, 8, 10, 12, 14, 16]
    NUM_STRIDES = 3

    def __init__(self):
        self.combinations_per_count = {
            n: comb(n, self.NUM_STRIDES) for n in self.BLOCK_COUNTS
        }

        self.cumulative_offsets = {}
        offset = 0
        for n in self.BLOCK_COUNTS:
            self.cumulative_offsets[n] = offset
            offset += self.combinations_per_count[n]

        self.total_combinations = offset  # 1344

        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """构建编码/解码查找表"""
        self.code_to_config: Dict[int, Tuple[int, Tuple[int, ...]]] = {}
        self.config_to_code: Dict[Tuple[int, Tuple[int, ...]], int] = {}

        code = 0
        for num_blocks in self.BLOCK_COUNTS:
            for positions in combinations(range(num_blocks), self.NUM_STRIDES):
                self.code_to_config[code] = (num_blocks, positions)
                self.config_to_code[(num_blocks, positions)] = code
                code += 1

    def encode(self, num_blocks: int, stride_positions: Tuple[int, ...]) -> int:
        """将配置编码为整数"""
        positions = tuple(sorted(stride_positions))
        if (num_blocks, positions) not in self.config_to_code:
            raise ValueError(f"Invalid config: {num_blocks} blocks, positions {positions}")
        return self.config_to_code[(num_blocks, positions)]

    def decode(self, code: int) -> Tuple[int, Tuple[int, ...]]:
        """将编码解码为配置"""
        if code < 0 or code >= self.total_combinations:
            raise ValueError(f"Code {code} out of range [0, {self.total_combinations})")
        return self.code_to_config[code]

    def get_strides_list(self, num_blocks: int, stride_positions: Tuple[int, ...]) -> List[int]:
        """获取每个block的stride值列表"""
        strides = [1] * num_blocks
        for pos in stride_positions:
            strides[pos] = 2
        return strides

    def summary(self) -> str:
        lines = [
            "Body Stride编码器:",
            f"  Block数量选项: {self.BLOCK_COUNTS}",
            f"  Stride=2位置数: {self.NUM_STRIDES}",
            f"  总编码数: {self.total_combinations}",
            "",
            "  编码范围:",
        ]
        for n in self.BLOCK_COUNTS:
            start = self.cumulative_offsets[n]
            end = start + self.combinations_per_count[n] - 1
            lines.append(f"    {n}blocks: [{start}, {end}]")
        return "\n".join(lines)


# ============ 通道数计算 ============
class ChannelCalculator:
    """
    通道数计算器

    根据CT（密文）数量和特征图大小反向计算通道数。

    核心公式：
    - 一个CT可存储 ct_slots 个数据
    - 特征图大小 H × W × C
    - 需要的CT数量 = ceil(H × W × C / ct_slots)

    反向计算：
    - 给定CT数量和特征图大小，计算通道数
    - C = (ct_count × ct_slots) / (H × W)

    CT策略：
    - "keep": 保持当前的CT数量，根据新的H×W反推通道数
    - "half": CT数量减半，根据新的H×W反推通道数
    """

    def __init__(
        self,
        ct_slots: int = 32768,
        input_size: int = 224,
        stem_downsample: int = 4
    ):
        """
        Args:
            ct_slots: 每个CT的槽位数（典型值: 2^15 = 32768）
            input_size: 输入图像大小（默认224）
            stem_downsample: stem层的降采样因子（默认4）
        """
        self.ct_slots = ct_slots
        self.input_size = input_size
        self.stem_downsample = stem_downsample

        # stem后的特征图大小
        self.feature_size_after_stem = input_size // stem_downsample  # 56

    def compute_channels_from_ct(self, ct_count: int, feature_size: int) -> int:
        """
        根据CT数量和特征图大小计算通道数

        Args:
            ct_count: 使用的CT数量
            feature_size: 特征图边长（假设H=W）

        Returns:
            通道数 C = (ct_count × ct_slots) / (H × W)
            注意：返回偶数以确保与SelfGated模块兼容
        """
        total_slots = ct_count * self.ct_slots
        channels = total_slots // (feature_size * feature_size)
        # 确保通道数是偶数（SelfGated模块要求）
        channels = (channels // 2) * 2
        return max(2, channels)

    def compute_channels_sequence(
        self,
        strides: List[int],
        ct_policies: List[str],
        initial_ct_count: int = 1,
        initial_feature_size: Optional[int] = None
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        计算每个block的通道数、特征图大小和CT数量

        Args:
            strides: 每个block的stride列表
            ct_policies: 每次stride=2时的CT策略
                - "keep": 保持CT数量不变
                - "half": CT数量减半
            initial_ct_count: 初始CT数量
            initial_feature_size: 初始特征图大小

        Returns:
            (channels_list, feature_sizes_list, ct_counts_list)
        """
        if initial_feature_size is None:
            initial_feature_size = self.feature_size_after_stem

        channels = []
        feature_sizes = []
        ct_counts = []

        current_feature_size = initial_feature_size
        current_ct_count = initial_ct_count
        stride_idx = 0

        for stride in strides:
            if stride == 2:
                current_feature_size = current_feature_size // 2
                policy = ct_policies[stride_idx] if stride_idx < len(ct_policies) else "keep"
                stride_idx += 1

                if policy == "half":
                    current_ct_count = max(1, current_ct_count // 2)
                # keep: 保持CT数量不变

            out_channels = self.compute_channels_from_ct(current_ct_count, current_feature_size)

            channels.append(out_channels)
            feature_sizes.append(current_feature_size)
            ct_counts.append(current_ct_count)

        return channels, feature_sizes, ct_counts

    def get_initial_channels(self, ct_count: int = 1) -> int:
        """获取stem后的初始通道数"""
        return self.compute_channels_from_ct(ct_count, self.feature_size_after_stem)


# ============ 搜索空间 ============
@dataclass
class SearchSpace:
    """
    FHE-NAS搜索空间

    编码维度：
    1. stem_code: [0-3] Stem层配置
    2. second_ds_code: [0-4] 第二次降分辨率配置
    3. stride_code: [0-1343] Body部分的block数量和stride位置
    4. block_types: 每个block的类型
    5. activations: 每个block的激活函数
    6. ct_policies: 每次stride=2时的CT策略（keep/half）
    """

    # Block类型选项
    block_types: List[str] = field(default_factory=lambda: list(BLOCK_TYPES.keys()))

    # 激活函数选项
    activation_types: List[str] = field(default_factory=lambda: ["poly4", "swish"])

    # Bottleneck factor选项
    factor_options: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])

    # CT策略选项
    ct_policy_options: List[str] = field(default_factory=lambda: ["keep", "half"])

    # CT槽位数
    ct_slots: int = 32768

    # 输入图像大小
    input_size: int = 224

    # 初始CT数量
    initial_ct_count: int = 1

    def __post_init__(self):
        self.stride_encoder = StrideEncoder()
        self.channel_calculator = ChannelCalculator(
            self.ct_slots, self.input_size
        )

    @property
    def num_stem_configs(self) -> int:
        return len(STEM_CONFIGS)

    @property
    def num_second_ds_configs(self) -> int:
        return len(SECOND_DOWNSAMPLE_CONFIGS)

    @property
    def num_stride_configs(self) -> int:
        return self.stride_encoder.total_combinations

    def get_stem_config(self, code: int) -> StemConfig:
        """获取Stem配置"""
        return STEM_CONFIGS[code]

    def get_second_ds_config(self, code: int) -> SecondDownsampleConfig:
        """获取第二次降分辨率配置"""
        return SECOND_DOWNSAMPLE_CONFIGS[code]

    def get_block_spec(self, block_type: str) -> BlockSpec:
        """获取block规格"""
        return BLOCK_TYPES[block_type]

    def get_activation_class(self, activation_type: str) -> Type:
        """获取激活函数类"""
        return ACTIVATION_TYPES[activation_type].activation_class

    def compute_search_space_size(self) -> int:
        """
        估算搜索空间大小

        分层选择策略:
        - 前4个block: 每个单独选择 (24种)
        - 后面的block: 每2个共享选择 (24种)

        对于平均10个block的网络:
        - 前4个: 24^4
        - 后6个 (3组): 24^3
        - Block选择总数: 24^7

        再乘以其他因素:
        - Stem: 4
        - SecondDS: 5
        - Stride: 1344
        - CT策略: 8
        """
        stem_choices = self.num_stem_configs  # 4
        second_ds_choices = self.num_second_ds_configs  # 5
        stride_choices = self.num_stride_configs  # 1344
        ct_policy_choices = len(self.ct_policy_options) ** 3  # 8

        num_unified_blocks = 24

        # 分层选择: 前4个单独选 + 后面每2个一组
        # 假设平均10个block: 4 + ceil(6/2) = 4 + 3 = 7个选择位
        individual_blocks = 4
        avg_remaining = 6  # 平均剩余block数
        grouped_choices = (avg_remaining + 1) // 2  # 向上取整

        block_selection_choices = num_unified_blocks ** (individual_blocks + grouped_choices)

        total = (
            stem_choices *
            second_ds_choices *
            stride_choices *
            block_selection_choices *
            ct_policy_choices
        )
        return int(total)

    def compute_block_choices_for_network(self, num_blocks: int) -> int:
        """
        计算指定block数量的网络的block选择数

        Args:
            num_blocks: 网络的block数量

        Returns:
            block选择的组合数
        """
        num_unified_blocks = 24
        individual_blocks = min(4, num_blocks)  # 前4个单独选
        remaining = max(0, num_blocks - 4)
        grouped_choices = (remaining + 1) // 2  # 每2个一组

        return num_unified_blocks ** (individual_blocks + grouped_choices)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "FHE-NAS 搜索空间",
            "=" * 60,
            "",
            f"Stem层配置 ({self.num_stem_configs}种):",
        ]
        for i, cfg in enumerate(STEM_CONFIGS):
            lines.append(f"  [{i}] {cfg}")

        lines.append(f"\n第二次降分辨率配置 ({self.num_second_ds_configs}种):")
        for i, cfg in enumerate(SECOND_DOWNSAMPLE_CONFIGS):
            lines.append(f"  [{i}] {cfg}")

        lines.append("")
        lines.append(self.stride_encoder.summary())

        lines.append("\n24种统一Block定义:")
        for i in range(24):
            spec = UNIFIED_BLOCKS[i]
            lines.append(f"  [{i:2d}] {spec.name:25s} {spec.description}")

        lines.append("\n分层选择策略:")
        lines.append("  - 前4个block: 每个单独选择 (24种/位置)")
        lines.append("  - 后面block: 每2个共享选择 (24种/组)")

        lines.append(f"\nCT策略选项: {self.ct_policy_options}")
        lines.append(f"CT槽位数: {self.ct_slots}")
        lines.append(f"初始CT数量: {self.initial_ct_count}")
        lines.append(f"输入图像: {self.input_size}×{self.input_size}")

        # 详细的搜索空间计算
        lines.append("\n搜索空间估算 (平均10个block):")
        lines.append(f"  Block选择: 24^(4+3) = 24^7 ≈ {24**7:.2e}")
        lines.append("  × Stem(4) × SecondDS(5) × Stride(1344) × CT策略(8)")
        lines.append(f"  = {self.compute_search_space_size():.2e}")
        lines.append("=" * 60)

        return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    space = SearchSpace()
    print(space.summary())

    print("\n" + "=" * 60)
    print("测试Stride编码")
    print("=" * 60)
    encoder = StrideEncoder()
    for code in [0, 10, 100, 500, 1000, 1343]:
        num_blocks, positions = encoder.decode(code)
        strides = encoder.get_strides_list(num_blocks, positions)
        print(f"  code={code}: {num_blocks}blocks, positions={positions}")
        print(f"    strides={strides}")

    print("\n" + "=" * 60)
    print("测试通道计算 (CT策略)")
    print("=" * 60)
    calc = ChannelCalculator()
    strides = [1, 2, 1, 2, 1, 2]
    ct_policies = ["keep", "half", "keep"]
    channels, sizes, ct_counts = calc.compute_channels_sequence(strides, ct_policies)
    print(f"  strides: {strides}")
    print(f"  ct_policies: {ct_policies}")
    print()
    for i, (c, s, ct, st) in enumerate(zip(channels, sizes, ct_counts, strides)):
        print(f"  block {i}: stride={st}, feature_size={s}×{s}, ct_count={ct}, channels={c}")
