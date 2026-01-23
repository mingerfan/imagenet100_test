import torch
import torch.nn as nn


def _safe_gated_mul(feat, gate):
    if not torch.is_tensor(feat):
        return feat * gate
    if not feat.is_floating_point():
        return feat * gate
    dtype = feat.dtype
    if dtype in (torch.float16, torch.bfloat16):
        feat_f = feat.float()
        gate_f = gate.float()
        prod = feat_f * gate_f
        max_val = torch.finfo(dtype).max
        prod = torch.nan_to_num(prod, nan=0.0, posinf=max_val, neginf=-max_val)
        prod = torch.clamp(prod, min=-max_val, max=max_val)
        return prod.to(dtype)
    # float32 也需要保护，防止 inf/nan 传播
    prod = feat * gate
    prod = torch.nan_to_num(prod, nan=0.0, posinf=1e6, neginf=-1e6)
    prod = torch.clamp(prod, min=-1e6, max=1e6)
    return prod


def _safe_conv_bn(conv, bn, x):
    if not torch.is_tensor(x):
        return bn(conv(x))
    if not x.is_floating_point():
        return bn(conv(x))
    dtype = x.dtype
    if dtype in (torch.float16, torch.bfloat16):
        conv_weight = getattr(conv, "weight", None)
        bn_weight = getattr(bn, "weight", None)
        conv_dtype = conv_weight.dtype if conv_weight is not None else dtype
        bn_dtype = bn_weight.dtype if bn_weight is not None else conv_dtype
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            if conv_dtype == torch.float32 and bn_dtype == torch.float32:
                y = conv(x.float())
                y = bn(y)
            else:
                x_work = x.to(conv_dtype) if x.dtype != conv_dtype else x
                y = conv(x_work)
                if y.dtype != bn_dtype:
                    y = y.to(bn_dtype)
                y = bn(y)
        max_val = torch.finfo(dtype).max
        y = torch.nan_to_num(y, nan=0.0, posinf=max_val, neginf=-max_val)
        y = torch.clamp(y, min=-max_val, max=max_val)
        return y.to(dtype)
    # float32 也需要保护
    y = bn(conv(x))
    y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    y = torch.clamp(y, min=-1e6, max=1e6)
    return y


def _safe_bn_output(x, max_feature_val=1000.0):
    """
    高效的 BN 输出保护 - 只在必要时保护
    
    fp16 下 BN 输出可能达到 ±14000+，在后续乘法操作中容易溢出。
    本函数采用分层策略，避免不必要的GPU操作开销：
    
    1. fp32: 直接返回（范围足够大）
    2. fp16 + 值域安全: 快速返回（只需一次 max 检查）
    3. fp16 + 值域危险: 执行 clamp 保护
    
    Args:
        x: BN 输出张量
        max_feature_val: 最大特征值（默认1000）
    
    Returns:
        可能被保护的张量
    """
    if not torch.is_tensor(x) or not x.is_floating_point():
        return x
    
    dtype = x.dtype
    
    # fp32 的范围足够大，不需要保护
    if dtype == torch.float32:
        return x
    
    # fp16/bfloat16：检查值域是否安全
    if dtype in (torch.float16, torch.bfloat16):
        # 快速检查：如果最大绝对值 < 阈值的80%，说明很安全，直接返回
        abs_max = x.abs().max()
        if abs_max < max_feature_val * 0.8:  # 安全范围
            return x
        
        # 值域接近边界：执行保护
        # 先 clamp 再处理 NaN，避免双重检查
        x = torch.clamp(x, min=-max_feature_val, max=max_feature_val)
        x = torch.nan_to_num(x, nan=0.0, posinf=max_feature_val, neginf=-max_feature_val)
        return x
    
    return x


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
    - 使用Swish进行预热训练（前warmup_epochs个epoch）
    - 平滑过渡到多项式激活
    - 使用更小的初始化值防止梯度爆炸
    - 添加梯度裁剪保护
    - 限制高阶项的范围
    """

    def __init__(self, output_scale=0.1, warmup_epochs=30):
        super().__init__()
        self.output_scale = output_scale
        self.warmup_epochs = warmup_epochs
        self.warmup_act = nn.SiLU()

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

        orig_dtype = x_clipped.dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            x_work = x_clipped.float()
        else:
            x_work = x_clipped

        # 计算Swish激活（用于预热）
        swish_out = self.warmup_act(x_work)

        # 计算多项式激活
        x2 = x_work * x_work
        x3 = x2 * x_work
        x4 = x3 * x_work

        # 对高阶参数进行软约束，防止它们过大
        a_clamped = torch.clamp(self.a, min=-0.01, max=0.01)
        b_clamped = torch.clamp(self.b, min=-0.1, max=0.1)
        c_clamped = torch.clamp(self.c, min=-0.5, max=0.5)
        d_clamped = torch.clamp(self.d, min=-5.0, max=5.0)
        e_clamped = torch.clamp(self.e, min=-5.0, max=5.0)

        poly_out = (
            a_clamped * x4
            + b_clamped * x3
            + c_clamped * x2
            + d_clamped * x_work
            + e_clamped
        )

        # 渐进式过渡：前warmup_epochs个epoch完全使用Swish，然后平滑过渡到多项式
        # 从 buffer 中获取当前 epoch 值
        epoch = self.current_epoch.item()
        if epoch < self.warmup_epochs:
            # 预热阶段：使用Swish
            alpha = 0.0
        elif epoch < self.warmup_epochs + 10:
            # 过渡阶段（10个epoch）：从Swish平滑过渡到多项式
            progress = (epoch - self.warmup_epochs) / 10.0
            alpha = progress
        else:
            # 完全使用多项式
            alpha = 1.0

        # 混合Swish和多项式输出
        out = (1 - alpha) * swish_out + alpha * poly_out

        out = out * self.output_scale
        
        # 始终检查并处理 NaN/Inf，防止数值不稳定传播
        # 先 clamp 到合理范围，再处理可能的 NaN
        out = torch.clamp(out, min=-100.0, max=100.0)
        out = torch.nan_to_num(out, nan=0.0, posinf=100.0, neginf=-100.0)
        
        if out.dtype != orig_dtype:
            max_val = torch.finfo(orig_dtype).max
            out = torch.clamp(out, min=-max_val, max=max_val)
            out = out.to(orig_dtype)
        return out


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
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
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
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
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
        
        # 用于检测溢出的标志
        self._overflow_warned = False

    def forward(self, x):
        # 输入检测
        if not self._overflow_warned and not torch.isfinite(x).all():
            print(f"\n⚠️ SelfGated输入检测: 输入包含非有限值!")
            print(f"   输入shape: {x.shape}, dtype: {x.dtype}")
            print(f"   NaN数量: {torch.isnan(x).sum().item()}")
            print(f"   Inf数量: {torch.isinf(x).sum().item()}")
            self._overflow_warned = True
        
        feat_intrinsic = self.conv_3x3(x)
        feat_intrinsic = self.bn_3x3(feat_intrinsic)
        
        # 关键保护：限制 BN 输出范围，防止 fp16 溢出
        feat_intrinsic = _safe_bn_output(feat_intrinsic, max_feature_val=1000.0)
        
        # feat_intrinsic检测
        if not self._overflow_warned and not torch.isfinite(feat_intrinsic).all():
            print(f"\n⚠️ SelfGated检测: conv_3x3+bn后产生非有限值!")
            print(f"   feat_intrinsic shape: {feat_intrinsic.shape}, dtype: {feat_intrinsic.dtype}")
            finite_mask = torch.isfinite(feat_intrinsic)
            if finite_mask.any():
                print(f"   有限值范围: [{feat_intrinsic[finite_mask].min().item():.2f}, {feat_intrinsic[finite_mask].max().item():.2f}]")
            print(f"   NaN数量: {torch.isnan(feat_intrinsic).sum().item()}")
            print(f"   Inf数量: {torch.isinf(feat_intrinsic).sum().item()}")
            self._overflow_warned = True

        gate = self.conv_gate(feat_intrinsic)
        
        # gate检测(激活前)
        if not self._overflow_warned and not torch.isfinite(gate).all():
            print(f"\n⚠️ SelfGated检测: conv_gate后产生非有限值!")
            print(f"   gate shape: {gate.shape}, dtype: {gate.dtype}")
            finite_mask = torch.isfinite(gate)
            if finite_mask.any():
                print(f"   有限值范围: [{gate[finite_mask].min().item():.2f}, {gate[finite_mask].max().item():.2f}]")
            print(f"   NaN数量: {torch.isnan(gate).sum().item()}")
            print(f"   Inf数量: {torch.isinf(gate).sum().item()}")
            self._overflow_warned = True
        
        gate = self.act(gate)
        
        # gate检测(激活后)
        if not self._overflow_warned and not torch.isfinite(gate).all():
            print(f"\n⚠️ SelfGated检测: 激活函数后产生非有限值!")
            print(f"   激活函数类型: {type(self.act).__name__}")
            print(f"   gate shape: {gate.shape}, dtype: {gate.dtype}")
            finite_mask = torch.isfinite(gate)
            if finite_mask.any():
                print(f"   有限值范围: [{gate[finite_mask].min().item():.2f}, {gate[finite_mask].max().item():.2f}]")
            print(f"   NaN数量: {torch.isnan(gate).sum().item()}")
            print(f"   Inf数量: {torch.isinf(gate).sum().item()}")
            self._overflow_warned = True

        feat_generated = _safe_gated_mul(feat_intrinsic, gate)
        
        # feat_generated检测
        if not self._overflow_warned and not torch.isfinite(feat_generated).all():
            print(f"\n⚠️ SelfGated检测: 门控乘法后产生非有限值!")
            print(f"   feat_generated shape: {feat_generated.shape}, dtype: {feat_generated.dtype}")
            finite_mask = torch.isfinite(feat_generated)
            if finite_mask.any():
                print(f"   有限值范围: [{feat_generated[finite_mask].min().item():.2f}, {feat_generated[finite_mask].max().item():.2f}]")
            print(f"   NaN数量: {torch.isnan(feat_generated).sum().item()}")
            print(f"   Inf数量: {torch.isinf(feat_generated).sum().item()}")
            self._overflow_warned = True

        out = torch.cat([feat_intrinsic, feat_generated], dim=1)
        
        # concat检测
        if not self._overflow_warned and not torch.isfinite(out).all():
            print(f"\n⚠️ SelfGated检测: concat后产生非有限值!")
            print(f"   out shape: {out.shape}, dtype: {out.dtype}")
            self._overflow_warned = True

        out = _safe_conv_bn(self.conv_out, self.bn_out, out)
        
        # 最终输出检测
        if not self._overflow_warned and not torch.isfinite(out).all():
            print(f"\n⚠️ SelfGated检测: conv_out+bn后产生非有限值!")
            print(f"   最终输出shape: {out.shape}, dtype: {out.dtype}")
            finite_mask = torch.isfinite(out)
            if finite_mask.any():
                print(f"   有限值范围: [{out[finite_mask].min().item():.2f}, {out[finite_mask].max().item():.2f}]")
            print(f"   NaN数量: {torch.isnan(out).sum().item()}")
            print(f"   Inf数量: {torch.isinf(out).sum().item()}")
            self._overflow_warned = True

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
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
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
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False
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
        
        # 用于检测溢出的标志
        self._overflow_warned = False

    def forward(self, x):
        # 输入检测
        if not self._overflow_warned and not torch.isfinite(x).all():
            print(f"\n⚠️ GatedDepthwiseConv输入检测: 输入包含非有限值!")
            print(f"   输入shape: {x.shape}, dtype: {x.dtype}")
            print(f"   NaN数量: {torch.isnan(x).sum().item()}")
            print(f"   Inf数量: {torch.isinf(x).sum().item()}")
            self._overflow_warned = True
        
        # 主分支特征
        feat_intrinsic = self.dw_conv(x)
        feat_intrinsic = self.bn(feat_intrinsic)
        
        # 关键保护：限制 BN 输出范围，防止 fp16 溢出
        feat_intrinsic = _safe_bn_output(feat_intrinsic, max_feature_val=1000.0)
        
        # feat_intrinsic检测
        if not self._overflow_warned and not torch.isfinite(feat_intrinsic).all():
            print(f"\n⚠️ GatedDepthwiseConv检测: dw_conv+bn后产生非有限值!")
            print(f"   feat_intrinsic shape: {feat_intrinsic.shape}, dtype: {feat_intrinsic.dtype}")
            finite_mask = torch.isfinite(feat_intrinsic)
            if finite_mask.any():
                print(f"   有限值范围: [{feat_intrinsic[finite_mask].min().item():.2f}, {feat_intrinsic[finite_mask].max().item():.2f}]")
            print(f"   NaN数量: {torch.isnan(feat_intrinsic).sum().item()}")
            print(f"   Inf数量: {torch.isinf(feat_intrinsic).sum().item()}")
            self._overflow_warned = True

        # 门控分支
        gate = self.gate_conv(feat_intrinsic)
        
        # gate检测(激活前)
        if not self._overflow_warned and not torch.isfinite(gate).all():
            print(f"\n⚠️ GatedDepthwiseConv检测: gate_conv后产生非有限值!")
            print(f"   gate shape: {gate.shape}, dtype: {gate.dtype}")
            finite_mask = torch.isfinite(gate)
            if finite_mask.any():
                print(f"   有限值范围: [{gate[finite_mask].min().item():.2f}, {gate[finite_mask].max().item():.2f}]")
            print(f"   NaN数量: {torch.isnan(gate).sum().item()}")
            print(f"   Inf数量: {torch.isinf(gate).sum().item()}")
            self._overflow_warned = True
        
        gate = self.activation(gate)
        
        # gate检测(激活后)
        if not self._overflow_warned and not torch.isfinite(gate).all():
            print(f"\n⚠️ GatedDepthwiseConv检测: 激活函数后产生非有限值!")
            print(f"   激活函数类型: {type(self.activation).__name__}")
            print(f"   gate shape: {gate.shape}, dtype: {gate.dtype}")
            finite_mask = torch.isfinite(gate)
            if finite_mask.any():
                print(f"   有限值范围: [{gate[finite_mask].min().item():.2f}, {gate[finite_mask].max().item():.2f}]")
            print(f"   NaN数量: {torch.isnan(gate).sum().item()}")
            print(f"   Inf数量: {torch.isinf(gate).sum().item()}")
            self._overflow_warned = True

        # 生成门控特征
        feat_gated = _safe_gated_mul(feat_intrinsic, gate)
        
        # feat_gated检测
        if not self._overflow_warned and not torch.isfinite(feat_gated).all():
            print(f"\n⚠️ GatedDepthwiseConv检测: 门控乘法后产生非有限值!")
            print(f"   feat_gated shape: {feat_gated.shape}, dtype: {feat_gated.dtype}")
            finite_mask = torch.isfinite(feat_gated)
            if finite_mask.any():
                print(f"   有限值范围: [{feat_gated[finite_mask].min().item():.2f}, {feat_gated[finite_mask].max().item():.2f}]")
            print(f"   NaN数量: {torch.isnan(feat_gated).sum().item()}")
            print(f"   Inf数量: {torch.isinf(feat_gated).sum().item()}")
            self._overflow_warned = True

        # 拼接恢复通道数（类似SelfGated）
        out = torch.cat([feat_intrinsic, feat_gated], dim=1)
        
        # 最终输出检测
        if not self._overflow_warned and not torch.isfinite(out).all():
            print(f"\n⚠️ GatedDepthwiseConv检测: concat后产生非有限值!")
            print(f"   最终输出shape: {out.shape}, dtype: {out.dtype}")
            finite_mask = torch.isfinite(out)
            if finite_mask.any():
                print(f"   有限值范围: [{out[finite_mask].min().item():.2f}, {out[finite_mask].max().item():.2f}]")
            print(f"   NaN数量: {torch.isnan(out).sum().item()}")
            print(f"   Inf数量: {torch.isinf(out).sum().item()}")
            self._overflow_warned = True
        
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
        
        # 用于检测溢出的标志
        self._overflow_warned = False

    def forward(self, x):
        identity = x
        
        # 输入检测
        if not self._overflow_warned and not torch.isfinite(x).all():
            print(f"\n⚠️ MBConvBlock输入检测: 输入包含非有限值!")
            print(f"   输入shape: {x.shape}, dtype: {x.dtype}")
            print(f"   NaN数量: {torch.isnan(x).sum().item()}")
            print(f"   Inf数量: {torch.isinf(x).sum().item()}")
            self._overflow_warned = True

        # 1. 扩展阶段
        if self.use_expansion:
            out = self.expand_conv(x)
            out = self.bn1(out)
            
            # 关键保护：限制 BN 输出范围，防止 fp16 溢出
            out = _safe_bn_output(out, max_feature_val=1000.0)
            
            # expansion后检测（关键位置！）
            if not self._overflow_warned and not torch.isfinite(out).all():
                print(f"\n⚠️ MBConvBlock检测: expansion+bn后产生非有限值!")
                print(f"   out shape: {out.shape}, dtype: {out.dtype}")
                finite_mask = torch.isfinite(out)
                if finite_mask.any():
                    print(f"   有限值范围: [{out[finite_mask].min().item():.2f}, {out[finite_mask].max().item():.2f}]")
                print(f"   NaN数量: {torch.isnan(out).sum().item()}")
                print(f"   Inf数量: {torch.isinf(out).sum().item()}")
                self._overflow_warned = True
            
            out = self.activation(out)
            
            # activation后检测
            if not self._overflow_warned and not torch.isfinite(out).all():
                print(f"\n⚠️ MBConvBlock检测: expansion激活后产生非有限值!")
                print(f"   激活函数类型: {type(self.activation).__name__}")
                print(f"   out shape: {out.shape}, dtype: {out.dtype}")
                finite_mask = torch.isfinite(out)
                if finite_mask.any():
                    print(f"   有限值范围: [{out[finite_mask].min().item():.2f}, {out[finite_mask].max().item():.2f}]")
                print(f"   NaN数量: {torch.isnan(out).sum().item()}")
                print(f"   Inf数量: {torch.isinf(out).sum().item()}")
                self._overflow_warned = True
        else:
            out = x

        # 2. 深度卷积阶段
        out = self.dw_conv(out)
        if not self.use_gated_dw:
            out = self.bn2(out)
            # 关键保护：限制 BN 输出范围
            out = _safe_bn_output(out, max_feature_val=1000.0)
            out = self.activation(out)
        # GatedDWConv内部已经完成BN和激活+拼接

        # 3. SE注意力（可选）
        if self.use_se:
            out = self.se(out)

        # 4. 压缩阶段
        out = self.project_conv(out)
        out = self.bn3(out)
        # 关键保护：限制 BN 输出范围
        out = _safe_bn_output(out, max_feature_val=1000.0)

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
