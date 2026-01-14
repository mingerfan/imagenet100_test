import torch
import torch.nn as nn


class Activation(nn.Module):
    def forward(self, x):
        pass


class Relu(Activation):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)


class LearnableSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.swish = nn.SiLU()

    def forward(self, x):
        return self.swish(x)


class LearnableRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return torch.maximum(self.beta * x, torch.zeros_like(x))


class StablePoly4(nn.Module):
    """
    FHE友好激活函数 (多项式激活)
    f(x) = scale * (ax^4 + bx^3 + cx^2 + dx + e)
    用于: Standard Poly, Gated Poly (Ours)

    改进版本：
    - 使用ReLU进行预热训练（前warmup_epochs个epoch）
    - 平滑过渡到多项式激活
    - 使用更小的初始化值防止梯度爆炸
    - 添加梯度裁剪保护
    - 限制高阶项的范围
    """

    def __init__(self, output_scale=0.1, warmup_epochs=30):
        super().__init__()
        self.output_scale = output_scale
        self.warmup_epochs = warmup_epochs

        # 使用更小的初始化值，防止高阶项导致梯度爆炸
        # a, b, c 初始化为接近0的小值
        self.a = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.c = nn.Parameter(torch.tensor(0.001))  # 二次项保留小值
        self.d = nn.Parameter(torch.tensor(0.5))  # 线性项降为0.5
        self.e = nn.Parameter(torch.tensor(0.0))

        # 梯度裁剪阈值
        self.grad_clip_value = 1.0

        # 当前epoch（用于控制过渡）
        # 使用 register_buffer 确保 current_epoch 被保存到 state_dict 中
        # 这样推理时加载模型后能正确使用多项式激活
        self.register_buffer("current_epoch", torch.tensor(0, dtype=torch.long))

    def set_epoch(self, epoch):
        """设置当前epoch，用于控制ReLU到多项式的过渡"""
        self.current_epoch.fill_(epoch)

    def set_warmup_epochs(self, warmup_epochs):
        """动态设置warmup epoch数量

        Args:
            warmup_epochs: 新的warmup epoch数量
        """
        self.warmup_epochs = warmup_epochs

    def forward(self, x):
        # 数值稳定性保护：限制输入范围
        x_clipped = torch.clamp(x, min=-10.0, max=10.0)

        # 计算ReLU激活（用于预热）
        relu_out = torch.nn.functional.relu(x_clipped)

        # 计算多项式激活
        x2 = x_clipped * x_clipped
        x3 = x2 * x_clipped
        x4 = x3 * x_clipped

        # 对高阶参数进行软约束，防止它们过大
        a_clamped = torch.clamp(self.a, min=-0.01, max=0.01)
        b_clamped = torch.clamp(self.b, min=-0.1, max=0.1)
        c_clamped = torch.clamp(self.c, min=-0.5, max=0.5)

        poly_out = (
            a_clamped * x4
            + b_clamped * x3
            + c_clamped * x2
            + self.d * x_clipped
            + self.e
        )

        # 渐进式过渡：前warmup_epochs个epoch完全使用ReLU，然后平滑过渡到多项式
        # 从 buffer 中获取当前 epoch 值
        epoch = self.current_epoch.item()
        if epoch < self.warmup_epochs:
            # 预热阶段：使用ReLU
            alpha = 0.0
        elif epoch < self.warmup_epochs + 10:
            # 过渡阶段（10个epoch）：从ReLU平滑过渡到多项式
            progress = (epoch - self.warmup_epochs) / 10.0
            alpha = progress
        else:
            # 完全使用多项式
            alpha = 1.0

        # 混合ReLU和多项式输出
        out = (1 - alpha) * relu_out + alpha * poly_out

        return out * self.output_scale


class BasicBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int, activation: Activation
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.act = activation()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        shortcut_input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += self.shortcut(shortcut_input)
        x = self.act(x)

        return x


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        factor: float,
        activation: Activation,
    ):
        super().__init__()

        mid_channels = int(in_channels * factor)

        self.act = activation()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        input_x = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += self.shortcut(input_x)
        x = self.act(x)

        return x


class SelfGated(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int, activation: Activation
    ):
        super().__init__()

        mid_channels = out_channels // 2  # 使用拼接的方式恢复通道数

        self.conv_3x3 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn_3x3 = nn.BatchNorm2d(mid_channels)

        self.conv_gate = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=5, stride=1, padding=2
        )

        self.act = activation()

        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()

    def forward(self, x):
        feat_intrinsic = self.conv_3x3(x)
        feat_intrinsic = self.bn_3x3(feat_intrinsic)

        gate = self.conv_gate(feat_intrinsic)
        gate = self.act(gate)

        feat_generated = feat_intrinsic * gate

        out = torch.cat([feat_intrinsic, feat_generated], dim=1)

        out = self.bn_out(self.conv_out(out))

        return out


class BottleneckSelfGatedBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        factor: float,
        activation: Activation,
    ):
        super().__init__()
        mid_channels = int(in_channels * factor)

        self.act = activation()

        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.selfgate = SelfGated(mid_channels, out_channels, 1, activation)

        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        input_x = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.selfgate(x)

        x += self.shortcut(input_x)

        return x


class BasicSelfGatedBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        activation: Activation,
        full_gated: bool = False,
    ):
        super().__init__()

        self.act = activation()

        self.full_gated = full_gated

        if not full_gated:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = SelfGated(in_channels, out_channels, stride, activation)
            self.bn1 = nn.Identity()

        self.conv2 = SelfGated(out_channels, out_channels, 1, activation)
        self.shortcut = nn.Identity()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        shortcut_input = x
        x = self.conv1(x)
        x = self.bn1(x)
        if not self.full_gated:
            x = self.act(x)

        x = self.conv2(x)

        x += self.shortcut(shortcut_input)

        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力模块

    通过全局池化捕获通道间的相关性，生成通道注意力权重。

    Args:
        channels: 输入通道数
        reduction: 降维比例，默认为4
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale


class GatedDepthwiseConv(nn.Module):
    """门控深度可分离卷积

    关键优化：
    - 输入通道数已经减半（在MBConv的扩展层）
    - 通过拼接恢复到完整通道数
    - 减少通过激活函数的CT数量（FHE优化）

    结构：
    - 主分支：3x3深度卷积 + BN
    - 门控分支：5x5深度卷积 + 激活函数
    - 输出：cat([主分支, 主分支 * gate])

    Args:
        channels: 减半后的通道数
        stride: 步长
        activation: 激活函数类
    """

    def __init__(
        self,
        channels: int,
        stride: int,
        activation: Activation
    ):
        super().__init__()

        # 主分支：3x3深度卷积
        self.dw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=3, stride=stride, padding=1,
            groups=channels, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)

        # 门控分支：5x5深度卷积
        self.gate_conv = nn.Conv2d(
            channels, channels,
            kernel_size=5, stride=1, padding=2,
            groups=channels, bias=False
        )
        self.activation = activation()

    def forward(self, x):
        # 主分支特征
        feat_intrinsic = self.dw_conv(x)
        feat_intrinsic = self.bn(feat_intrinsic)

        # 门控分支
        gate = self.gate_conv(feat_intrinsic)
        gate = self.activation(gate)

        # 生成门控特征
        feat_gated = feat_intrinsic * gate

        # 拼接恢复通道数（类似SelfGated）
        out = torch.cat([feat_intrinsic, feat_gated], dim=1)
        return out  # 输出通道数是输入的2倍


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Block (MBConv)

    关键优化（GatedMBConv模式）：
    - 扩展层输出减半（expanded_channels // 2）
    - GatedDWConv通过拼接恢复通道数
    - 减少通过激活函数的CT数量

    结构：
    [1x1 expansion → BN → Act]
    → 3x3 DWConv (或 GatedDWConv) → BN → Act
    → [SE] (可选)
    → 1x1 projection → BN
    → [+shortcut] (stride=1 && in_ch==out_ch时)

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 步长
        expansion_factor: 扩展因子（1.0或4.0）
        activation: 激活函数类
        use_se: 是否使用SE注意力
        use_gated_dw: 是否使用门控深度卷积
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion_factor: float,
        activation: Activation,
        use_se: bool = False,
        use_gated_dw: bool = False
    ):
        super().__init__()

        self.use_se = use_se
        self.use_gated_dw = use_gated_dw

        # 计算扩展后的通道数
        expanded_channels = int(in_channels * expansion_factor)

        # 1x1扩展层（expansion=1时跳过）
        self.use_expansion = (expansion_factor != 1.0)

        # 关键优化：GatedDWConv模式下，只有当有expansion层时才减半
        # 如果expansion=1（无expansion层），则不减半
        if use_gated_dw and self.use_expansion:
            expansion_out_channels = expanded_channels // 2
        else:
            expansion_out_channels = expanded_channels

        if self.use_expansion:
            self.expand_conv = nn.Conv2d(
                in_channels, expansion_out_channels,
                kernel_size=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(expansion_out_channels)

        # 深度卷积
        if use_gated_dw:
            # GatedDWConv：输入expansion_out_channels，输出2*expansion_out_channels（通过拼接）
            self.dw_conv = GatedDepthwiseConv(
                expansion_out_channels, stride, activation
            )
            self.bn2 = nn.Identity()  # GatedDWConv内部已有BN
            # 拼接后的通道数是输入的2倍
            if self.use_expansion:
                # 有expansion层：expansion_out_channels = expanded_channels // 2
                # 拼接后恢复为 expanded_channels
                self.dw_out_channels = expanded_channels
            else:
                # 无expansion层：expansion_out_channels = in_channels
                # 拼接后变为 2 * in_channels
                self.dw_out_channels = 2 * expansion_out_channels
        else:
            # 标准深度卷积
            self.dw_conv = nn.Conv2d(
                expansion_out_channels, expansion_out_channels,
                kernel_size=3, stride=stride, padding=1,
                groups=expansion_out_channels, bias=False
            )
            self.bn2 = nn.BatchNorm2d(expansion_out_channels)
            self.dw_out_channels = expansion_out_channels

        # SE模块（可选）
        if use_se:
            self.se = SEBlock(self.dw_out_channels)

        # 1x1压缩层
        self.project_conv = nn.Conv2d(
            self.dw_out_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.activation = activation()

        # Shortcut连接
        self.use_shortcut = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x

        # 1. 扩展阶段
        if self.use_expansion:
            out = self.expand_conv(x)
            out = self.bn1(out)
            out = self.activation(out)
        else:
            out = x

        # 2. 深度卷积阶段
        out = self.dw_conv(out)
        if not self.use_gated_dw:
            out = self.bn2(out)
            out = self.activation(out)
        # GatedDWConv内部已经完成BN和激活+拼接

        # 3. SE注意力（可选）
        if self.use_se:
            out = self.se(out)

        # 4. 压缩阶段
        out = self.project_conv(out)
        out = self.bn3(out)

        # 5. Shortcut连接
        if self.use_shortcut:
            out = out + identity

        return out


class FullGatedBasicBlock(nn.Module):
    """两层都使用SelfGated的BasicBlock

    结构：
    SelfGated(stride) → SelfGated(1) → [+shortcut]

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 步长
        activation: 激活函数类
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        activation: Activation
    ):
        super().__init__()

        self.activation = activation()

        # 第一层：SelfGated with stride
        self.conv1 = SelfGated(
            in_channels, out_channels,
            stride, activation
        )

        # 第二层：SelfGated without stride
        self.conv2 = SelfGated(
            out_channels, out_channels,
            1, activation
        )

        # Shortcut
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)

        out = out + identity
        return out
