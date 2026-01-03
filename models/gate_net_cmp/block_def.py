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
    """
    def __init__(self, output_scale=0.1):
        super().__init__()
        self.output_scale = output_scale
        # 初始化为近似线性，防止训练初期崩塌
        self.a = nn.Parameter(torch.tensor(0.0)) 
        self.b = nn.Parameter(torch.tensor(0.0))
        self.c = nn.Parameter(torch.tensor(0.0))   
        self.d = nn.Parameter(torch.tensor(1.0)) 
        self.e = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x2 = x * x
        x3 = x2 * x
        x4 = x3 * x
        out = self.a * x4 + self.b * x3 + self.c * x2 + self.d * x + self.e
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
