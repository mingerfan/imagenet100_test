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


class LearnableRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.maximum(self.beta * x, torch.zeros_like(x))

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
        self.c = nn.Parameter(torch.tensor(0.001))   # 二次项保留小值
        self.d = nn.Parameter(torch.tensor(0.5))     # 线性项降为0.5
        self.e = nn.Parameter(torch.tensor(0.0))
        
        # 梯度裁剪阈值
        self.grad_clip_value = 1.0
        
        # 当前epoch（用于控制过渡）
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """设置当前epoch，用于控制ReLU到多项式的过渡"""
        self.current_epoch = epoch

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
        
        poly_out = a_clamped * x4 + b_clamped * x3 + c_clamped * x2 + self.d * x_clipped + self.e
        
        # 渐进式过渡：前warmup_epochs个epoch完全使用ReLU，然后平滑过渡到多项式
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段：使用ReLU
            alpha = 0.0
        elif self.current_epoch < self.warmup_epochs + 10:
            # 过渡阶段（10个epoch）：从ReLU平滑过渡到多项式
            progress = (self.current_epoch - self.warmup_epochs) / 10.0
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