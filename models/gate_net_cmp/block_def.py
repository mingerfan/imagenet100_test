import torch
import torch.nn as nn
import torch.nn.functional as F

# 常量定义
BN_EPS = 1e-3  # 高eps防止验证集统计漂移
GATE_SCALE_INIT = 1e-3  # 门控缩放初始值


try:
    from torch.fx.proxy import Proxy as _FxProxy
except Exception:  # pragma: no cover
    _FxProxy = None


def _is_fx_proxy(x) -> bool:
    return _FxProxy is not None and isinstance(x, _FxProxy)


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


def _is_swish_activation(activation) -> bool:
    if activation is None:
        return False
    if isinstance(activation, str):
        return activation.strip().lower() in ("swish", "silu")
    if isinstance(activation, nn.Module):
        act_cls = activation.__class__
    else:
        act_cls = activation if isinstance(activation, type) else activation.__class__
    if act_cls is nn.SiLU:
        return True
    name = act_cls.__name__.lower()
    return name in ("swish", "learnableswish", "silu", "stablepoly4", "hermitepoly4", "swishherpn")


class Activation(nn.Module):
    def forward(self, x):
        pass


class Relu(Activation):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


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


class SwishHerPN(nn.Module):
    """
    AESPA-style Hermite polynomial activation for Swish.

    Uses degree-2 normalized probabilists' Hermite bases with basis-wise
    BatchNorm. Coefficients are initialized from the standard-normal projection
    of Swish/SiLU and left trainable for proxy training.
    """

    def __init__(
        self,
        degree=2,
        coeff0=0.20662,
        coeff1=0.5,
        coeff2=0.24860,
        eps=BN_EPS,
        trainable_coeffs=True,
    ):
        super().__init__()
        degree = int(degree)
        if degree not in {1, 2}:
            raise ValueError(f"SwishHerPN supports degree 1 or 2, got {degree}")
        self.degree = degree
        self.basis_norm1 = nn.LazyBatchNorm2d(eps=eps, affine=False)
        self.basis_norm2 = nn.LazyBatchNorm2d(eps=eps, affine=False) if degree >= 2 else None

        coeffs = torch.tensor([float(coeff1), float(coeff2)], dtype=torch.float32)
        if trainable_coeffs:
            self.coeffs = nn.Parameter(coeffs)
        else:
            self.register_buffer("coeffs", coeffs)
        self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(float(coeff0), dtype=torch.float32))

    def forward(self, x):
        orig_dtype = x.dtype
        x_work = x.float() if orig_dtype in (torch.float16, torch.bfloat16) else x

        h1 = self.basis_norm1(x_work)
        out = self.coeffs[0].to(dtype=x_work.dtype) * h1
        if self.degree >= 2:
            h2 = (x_work * x_work - 1.0) * (2.0 ** -0.5)
            h2 = torch.nan_to_num(h2, nan=0.0, posinf=1e6, neginf=-1e6)
            h2 = torch.clamp(h2, min=-1e6, max=1e6)
            out = out + self.coeffs[1].to(dtype=x_work.dtype) * self.basis_norm2(h2)

        gamma = self.gamma.to(dtype=x_work.dtype)
        beta = self.beta.to(dtype=x_work.dtype)
        out = gamma * out + beta
        out = torch.nan_to_num(out, nan=0.0, posinf=100.0, neginf=-100.0)
        return out.to(orig_dtype) if out.dtype != orig_dtype else out


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

    def __init__(
        self,
        output_scale=0.1,
        warmup_epochs=30,
        warmup_act="swish",
        in_scale_init=-1.0,
        range_r=2.0,
        deriv_L=3.0,
        enable_range_loss=True,
        enable_deriv_loss=True,
        scale_mode="learned",
        dynamic_scale_momentum=0.99,
        dynamic_scale_eps=1e-6,
        poly_degree=4,
        pat_swish_backward=False,
    ):
        super().__init__()
        self.output_scale = output_scale
        self.warmup_epochs = warmup_epochs
        self.warmup_act = self._build_warmup_act(warmup_act)
        self.warmup_act_name = self._normalize_warmup_act_name(warmup_act)
        self.poly_degree = self._normalize_poly_degree(poly_degree)

        # 使用更小的初始化值，防止高阶项导致梯度爆炸
        # a, b, c 初始化为接近0的小值
        self.a = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.c = nn.Parameter(torch.tensor(0.001))  # 二次项保留小值
        self.d = nn.Parameter(torch.tensor(0.5))  # 线性项降为0.5
        self.e = nn.Parameter(torch.tensor(0.0))

        # 可学习输入缩放（log-parameterization确保为正）
        self.log_in_scale = nn.Parameter(torch.tensor(in_scale_init))

        # 梯度裁剪阈值
        self.grad_clip_value = 1.0

        # 正则化配置与缓存
        self.range_r = range_r
        self.deriv_L = deriv_L
        self.enable_range_loss = enable_range_loss
        self.enable_deriv_loss = enable_deriv_loss
        self.scale_mode = self._normalize_scale_mode(scale_mode)
        self.dynamic_scale_momentum = float(dynamic_scale_momentum)
        self.dynamic_scale_eps = float(dynamic_scale_eps)
        self.pat_swish_backward = bool(pat_swish_backward)
        self.range_loss = torch.tensor(0.0)
        self.deriv_loss = torch.tensor(0.0)
        self.collect_stats = False
        self.last_x_poly_stats = None
        self.last_fprime_stats = None

        # 当前epoch（用于控制过渡）
        # 使用 register_buffer 确保 current_epoch 被保存到 state_dict 中
        # 这样推理时加载模型后能正确使用多项式激活
        self.register_buffer("current_epoch", torch.tensor(0, dtype=torch.long))
        # 细粒度进度（用于 step 级平滑过渡）
        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("steps_per_epoch", torch.tensor(1, dtype=torch.long))
        self.register_buffer("running_absmax", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("static_absmax", torch.tensor(1.0, dtype=torch.float32))
        self.poly_start_epoch = float(warmup_epochs)
        self.poly_transition_epochs = 10.0

    def set_epoch(self, epoch):
        """设置当前epoch，用于控制ReLU到多项式的过渡"""
        self.current_epoch.fill_(epoch)

    def set_epoch_progress(self, epoch, step_idx, steps_per_epoch):
        """设置当前epoch与step进度，用于 step 级平滑过渡"""
        self.current_epoch.fill_(int(epoch))
        self.current_step.fill_(int(step_idx))
        self.steps_per_epoch.fill_(max(1, int(steps_per_epoch)))

    def set_warmup_epochs(self, warmup_epochs):
        """动态设置warmup epoch数量

        Args:
            warmup_epochs: 新的warmup epoch数量
        """
        self.warmup_epochs = warmup_epochs
        self.poly_start_epoch = float(warmup_epochs)

    def set_poly_schedule(self, start_epoch=None, transition_epochs=None):
        """设置多项式分支开始生效的调度。

        start_epoch 为 epoch 小数进度阈值；transition_epochs 为从 warmup
        激活平滑过渡到多项式激活的持续 epoch 数。
        """
        if start_epoch is not None:
            self.poly_start_epoch = float(start_epoch)
        if transition_epochs is not None:
            self.poly_transition_epochs = max(0.0, float(transition_epochs))

    def set_warmup_act(self, warmup_act):
        """动态设置warmup阶段的激活函数"""
        self.warmup_act = self._build_warmup_act(warmup_act)
        self.warmup_act_name = self._normalize_warmup_act_name(warmup_act)

    def set_poly_degree(self, degree):
        """设置多项式最高次数，默认 4；低阶模式屏蔽高阶项。"""
        self.poly_degree = self._normalize_poly_degree(degree)

    def set_pat_swish_backward(self, enabled: bool):
        """Use Swish surrogate gradients for the polynomial branch input."""
        self.pat_swish_backward = bool(enabled)

    def set_collect_stats(self, enabled: bool):
        """是否在 forward 中收集诊断统计"""
        self.collect_stats = bool(enabled)

    def set_range_params(self, range_r=None, enable=None):
        """动态设置输入范围约束参数"""
        if range_r is not None:
            self.range_r = float(range_r)
        if enable is not None:
            self.enable_range_loss = bool(enable)

    def set_deriv_params(self, deriv_L=None, enable=None):
        """动态设置导数约束参数"""
        if deriv_L is not None:
            self.deriv_L = float(deriv_L)
        if enable is not None:
            self.enable_deriv_loss = bool(enable)

    def set_scale_mode(self, mode=None, momentum=None, eps=None, static_absmax=None):
        """设置 StablePoly4 输入缩放模式。"""
        if mode is not None:
            self.scale_mode = self._normalize_scale_mode(mode)
        if momentum is not None:
            self.dynamic_scale_momentum = float(momentum)
        if eps is not None:
            self.dynamic_scale_eps = float(eps)
        if static_absmax is not None:
            value = max(float(static_absmax), self.dynamic_scale_eps)
            self.static_absmax.fill_(value)

    @staticmethod
    def _normalize_scale_mode(mode):
        key = str(mode or "learned").strip().lower()
        if key not in {"learned", "dynamic", "static"}:
            raise ValueError(f"Unsupported StablePoly4 scale_mode: {mode}")
        return key

    @staticmethod
    def _normalize_poly_degree(degree):
        value = int(degree)
        if value not in {2, 3, 4}:
            raise ValueError(f"Unsupported StablePoly4 poly_degree: {degree}")
        return value

    def _calc_poly_input(self, x_work):
        if self.scale_mode == "learned":
            log_in_scale = torch.clamp(self.log_in_scale, min=-6.0, max=2.0)
            in_scale = torch.exp(log_in_scale)
            return x_work * in_scale, in_scale

        if self.scale_mode == "dynamic":
            batch_absmax = x_work.detach().abs().amax().float()
            batch_absmax = torch.clamp(batch_absmax, min=self.dynamic_scale_eps)
            if self.training:
                momentum = max(0.0, min(0.9999, self.dynamic_scale_momentum))
                self.running_absmax.mul_(momentum).add_(
                    batch_absmax.to(device=self.running_absmax.device) * (1.0 - momentum)
                )
                absmax = batch_absmax
            else:
                absmax = self.running_absmax.to(device=x_work.device, dtype=torch.float32)
                absmax = torch.clamp(absmax, min=self.dynamic_scale_eps)
            in_scale = x_work.new_tensor(1.0) / absmax.to(device=x_work.device, dtype=x_work.dtype)
            return x_work * in_scale, in_scale

        absmax = self.static_absmax.to(device=x_work.device, dtype=torch.float32)
        absmax = torch.clamp(absmax, min=self.dynamic_scale_eps)
        in_scale = x_work.new_tensor(1.0) / absmax.to(device=x_work.device, dtype=x_work.dtype)
        return x_work * in_scale, in_scale

    def _calc_poly_input_param_only(self, x_work, x_poly):
        """Poly input for parameter gradients while detaching activation input."""
        if self.scale_mode == "learned":
            log_in_scale = torch.clamp(self.log_in_scale, min=-6.0, max=2.0)
            return x_work.detach() * torch.exp(log_in_scale)
        return x_poly.detach()

    @staticmethod
    def _eval_poly(x_poly, a_eff, b_eff, c, d, e):
        return ((((a_eff * x_poly + b_eff) * x_poly + c) * x_poly + d) * x_poly + e)

    def _eval_poly_branch(self, x_poly, a_eff, b_eff, c, d, e):
        return self._eval_poly(x_poly, a_eff, b_eff, c, d, e)

    def _eval_poly_derivative(self, x_poly, a_eff, b_eff, c, d):
        return self.output_scale * (
            (((4.0 * a_eff) * x_poly + 3.0 * b_eff) * x_poly + 2.0 * c)
            * x_poly
            + d
        )

    @staticmethod
    def _build_warmup_act(warmup_act):
        if warmup_act is None:
            return nn.Identity()
        if isinstance(warmup_act, nn.Module):
            return warmup_act
        if isinstance(warmup_act, str):
            key = warmup_act.strip().lower()
            if key in ("swish", "silu"):
                return nn.SiLU()
            if key in ("sigmoid", "sig"):
                return nn.Sigmoid()
            if key == "relu":
                return nn.ReLU()
            if key == "tanh":
                return nn.Tanh()
            raise ValueError(f"Unsupported warmup_act: {warmup_act}")
        if isinstance(warmup_act, type) and issubclass(warmup_act, nn.Module):
            return warmup_act()
        if callable(warmup_act):
            return warmup_act()
        raise ValueError(f"Unsupported warmup_act: {warmup_act}")

    @staticmethod
    def _normalize_warmup_act_name(warmup_act):
        if warmup_act is None:
            return "identity"
        if isinstance(warmup_act, str):
            return warmup_act.strip().lower()
        if isinstance(warmup_act, nn.Module):
            return warmup_act.__class__.__name__.lower()
        if isinstance(warmup_act, type) and issubclass(warmup_act, nn.Module):
            return warmup_act.__name__.lower()
        return str(warmup_act)

    def forward(self, x):
        orig_dtype = x.dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            x_work = x.float()
        else:
            x_work = x

        # 计算Swish激活（用于预热）
        swish_out = self.warmup_act(x_work)

        # 输入范围控制：learned 使用可学习缩放；dynamic/static 使用 absmax 缩放。
        x_poly, _ = self._calc_poly_input(x_work)

        # 计算多项式激活
        # 对高阶参数进行软约束，防止它们过大
        a_clamped = torch.clamp(self.a, min=-0.01, max=0.01)
        b_clamped = torch.clamp(self.b, min=-0.1, max=0.1)
        c_clamped = torch.clamp(self.c, min=-0.5, max=0.5)
        d_clamped = torch.clamp(self.d, min=-5.0, max=5.0)
        e_clamped = torch.clamp(self.e, min=-5.0, max=5.0)
        degree = int(getattr(self, "poly_degree", 4))
        a_eff = a_clamped if degree >= 4 else torch.zeros_like(a_clamped)
        b_eff = b_clamped if degree >= 3 else torch.zeros_like(b_clamped)

        # Horner 形式计算多项式，数值更稳
        poly_out = self._eval_poly_branch(x_poly, a_eff, b_eff, c_clamped, d_clamped, e_clamped)

        if self.collect_stats:
            self.last_x_poly_stats = self._calc_stats(x_poly)

        # 输入范围正则
        if self.enable_range_loss:
            range_excess = F.relu(x_poly.abs() - self.range_r)
            self.range_loss = (range_excess * range_excess).mean()
        else:
            self.range_loss = x_poly.new_tensor(0.0)

        # 多项式导数正则（包含 output_scale）
        need_fprime = self.enable_deriv_loss or self.collect_stats
        if need_fprime:
            fprime = self._eval_poly_derivative(x_poly, a_eff, b_eff, c_clamped, d_clamped)
        if self.collect_stats:
            self.last_fprime_stats = self._calc_stats(fprime)
        if self.enable_deriv_loss:
            deriv_excess = F.relu(fprime.abs() - self.deriv_L)
            self.deriv_loss = (deriv_excess * deriv_excess).mean()
        else:
            self.deriv_loss = x_poly.new_tensor(0.0)

        # 渐进式过渡：前warmup_epochs个epoch完全使用Swish，然后平滑过渡到多项式
        # 使用 step 级进度（epoch 小数）进行平滑过渡
        epoch = float(self.current_epoch.item())
        step = float(self.current_step.item())
        steps_per_epoch = float(self.steps_per_epoch.item())
        epoch_progress = epoch + (step / max(1.0, steps_per_epoch))
        poly_start_epoch = float(getattr(self, "poly_start_epoch", self.warmup_epochs))
        transition_epochs = float(getattr(self, "poly_transition_epochs", 10.0))
        if epoch_progress < poly_start_epoch:
            # 预热阶段：使用Swish，但保持poly分支有梯度
            alpha = 0.0
        elif transition_epochs > 0 and epoch_progress < poly_start_epoch + transition_epochs:
            # 过渡阶段（10个epoch）：从Swish平滑过渡到多项式
            progress = (epoch_progress - poly_start_epoch) / transition_epochs
            alpha = progress
        else:
            alpha = 1.0

        # 混合Swish和多项式输出
        # warmup 期间保持 swish 为 1.0 缩放，逐步过渡到 poly 的输出缩放
        poly_branch = poly_out * self.output_scale
        if self.pat_swish_backward and self.training and alpha > 0.0:
            swish_surrogate = poly_branch.detach() + F.silu(x_work) - F.silu(x_work).detach()
            x_poly_param = self._calc_poly_input_param_only(x_work, x_poly)
            poly_param = self._eval_poly_branch(
                x_poly_param,
                a_eff,
                b_eff,
                c_clamped,
                d_clamped,
                e_clamped,
            ) * self.output_scale
            poly_branch = swish_surrogate + poly_param - poly_param.detach()
        out = (1 - alpha) * swish_out + alpha * poly_branch
        
        # 始终检查并处理 NaN/Inf，防止数值不稳定传播
        out = torch.nan_to_num(out, nan=0.0, posinf=100.0, neginf=-100.0)

        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out

    @staticmethod
    def _calc_stats(tensor):
        t = tensor.detach()
        t = t.abs()
        finite_mask = torch.isfinite(t)
        if finite_mask.any():
            t = t[finite_mask]
            t = t.float()
            # 限制采样数量，避免 quantile 对超大张量报错/过慢
            max_samples = 10_000
            if t.numel() > max_samples:
                idx = torch.randint(0, t.numel(), (max_samples,), device=t.device)
                t = t.view(-1)[idx]
            qs = torch.tensor([0.5, 0.9, 0.99], device=t.device)
            qv = torch.quantile(t, qs)
            return {
                "p50": float(qv[0].item()),
                "p90": float(qv[1].item()),
                "p99": float(qv[2].item()),
                "max": float(t.max().item()),
            }
        return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "max": float("nan")}


class HermitePoly4(StablePoly4):
    """
    StablePoly4 variant using AESPA/HerPN-style Hermite bases.

    The public training interface intentionally matches StablePoly4 so existing
    progressive scheduling, CT hooks, logging, and optimizer grouping can reuse
    the same controls.
    """

    def __init__(self, *args, basis_norm=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.basis_norm_enabled = bool(basis_norm)
        self.basis_norm1 = nn.LazyBatchNorm2d(eps=BN_EPS, affine=False)
        self.basis_norm2 = nn.LazyBatchNorm2d(eps=BN_EPS, affine=False)
        self.basis_norm3 = nn.LazyBatchNorm2d(eps=BN_EPS, affine=False)
        self.basis_norm4 = nn.LazyBatchNorm2d(eps=BN_EPS, affine=False)

    @staticmethod
    def _hermite_bases(x):
        h1 = x
        h2 = (x * x - 1.0) * (2.0 ** -0.5)
        h3 = (x * x * x - 3.0 * x) * (6.0 ** -0.5)
        h4 = (x * x * x * x - 6.0 * x * x + 3.0) * (24.0 ** -0.5)
        return h1, h2, h3, h4

    def _maybe_norm_basis(self, idx, basis):
        if not self.basis_norm_enabled or basis.dim() != 4:
            return basis
        norm = (self.basis_norm1, self.basis_norm2, self.basis_norm3, self.basis_norm4)[idx]
        return norm(basis)

    def _eval_poly_branch(self, x_poly, a_eff, b_eff, c, d, e):
        h1, h2, h3, h4 = self._hermite_bases(x_poly)
        h1 = self._maybe_norm_basis(0, h1)
        h2 = self._maybe_norm_basis(1, h2)
        h3 = self._maybe_norm_basis(2, h3)
        h4 = self._maybe_norm_basis(3, h4)
        out = e + d * h1 + c * h2 + b_eff * h3 + a_eff * h4
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

    def _eval_poly_derivative(self, x_poly, a_eff, b_eff, c, d):
        dh2 = (2.0 ** 0.5) * x_poly
        dh3 = (3.0 * x_poly * x_poly - 3.0) * (6.0 ** -0.5)
        dh4 = (4.0 * x_poly * x_poly * x_poly - 12.0 * x_poly) * (24.0 ** -0.5)
        return self.output_scale * (d + c * dh2 + b_eff * dh3 + a_eff * dh4)


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
        self.bn_3x3 = nn.BatchNorm2d(mid_channels, eps=BN_EPS)

        # 使用谱归一化限制Lipschitz常数，防止Gate值爆炸
        # 移除BN：BN会破坏SpectralNorm的Lipschitz约束，导致验证集上数值爆炸
        # 启用bias：因为移除了BN，需要卷积层自己学习偏置
        self.conv_gate = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=5, stride=1, padding=2, bias=False
        )
        # 移除BN层
        self.bn_gate = nn.BatchNorm2d(mid_channels, eps=BN_EPS)

        # Gate uses sigmoid when activation is Swish/SiLU to keep delta in [0,1]
        self.act = nn.Sigmoid() if _is_swish_activation(activation) else activation()

        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        
        # 门控缩放参数 (LayerScale/ReZero思想)
        self.gate_scale = nn.Parameter(torch.tensor(GATE_SCALE_INIT))

        self.shortcut = nn.Identity()
        
        # Zero-Init for Gated Branch
        # conv_out 输入是 concat([u, gate_branch])，后半通道 = gated-half
        # 初始化时把 conv_out 对后半通道的权重置 0
        with torch.no_grad():
             self.conv_out.weight[:, mid_channels:, :, :] = 0
        
        # 用于检测溢出的标志
        self._overflow_warned = False
        
        # 正则化损失缓存
        self.gate_reg_loss = torch.tensor(0.0)

    def forward(self, x):
        # 输入检测
        if (not _is_fx_proxy(x)) and (not self._overflow_warned) and (not torch.isfinite(x).all()):
            print("\n⚠️ SelfGated输入检测: 输入包含非有限值!")
            print(f"   输入shape: {x.shape}, dtype: {x.dtype}")
            print(f"   NaN数量: {torch.isnan(x).sum().item()}")
            print(f"   Inf数量: {torch.isinf(x).sum().item()}")
            self._overflow_warned = True
        
        feat_intrinsic = self.conv_3x3(x)
        feat_intrinsic = self.bn_3x3(feat_intrinsic)
        
        # 关键保护：限制 BN 输出范围，防止 fp16 溢出
        feat_intrinsic = _safe_bn_output(feat_intrinsic, max_feature_val=1000.0)
        
        # feat_intrinsic检测
        if (not _is_fx_proxy(feat_intrinsic)) and (not self._overflow_warned) and (not torch.isfinite(feat_intrinsic).all()):
            print("\n⚠️ SelfGated检测: conv_3x3+bn后产生非有限值!")
            print(f"   feat_intrinsic shape: {feat_intrinsic.shape}, dtype: {feat_intrinsic.dtype}")
            finite_mask = torch.isfinite(feat_intrinsic)
            if finite_mask.any():
                print(f"   有限值范围: [{feat_intrinsic[finite_mask].min().item():.2f}, {feat_intrinsic[finite_mask].max().item():.2f}]")
            print(f"   NaN数量: {torch.isnan(feat_intrinsic).sum().item()}")
            print(f"   Inf数量: {torch.isinf(feat_intrinsic).sum().item()}")
            self._overflow_warned = True

        gate = self.conv_gate(feat_intrinsic)
        gate = self.bn_gate(gate)
        
        # gate检测(激活前)
        if (not _is_fx_proxy(gate)) and (not self._overflow_warned) and (not torch.isfinite(gate).all()):
            print("\n⚠️ SelfGated检测: conv_gate+bn后产生非有限值!")
            print(f"   gate shape: {gate.shape}, dtype: {gate.dtype}")
            finite_mask = torch.isfinite(gate)
            if finite_mask.any():
                print(f"   有限值范围: [{gate[finite_mask].min().item():.2f}, {gate[finite_mask].max().item():.2f}]")
            print(f"   NaN数量: {torch.isnan(gate).sum().item()}")
            print(f"   Inf数量: {torch.isinf(gate).sum().item()}")
            self._overflow_warned = True
        
        # 结构性预缩放：缩小输入幅度
        delta = gate * 0.125
        
        # delta = φ(gate * 0.125)  (扰动项)
        delta = self.act(delta)
        
        # 计算激活正则化损失 (L2) - 改为对delta约束
        self.gate_reg_loss = (delta ** 2).mean()
        
        # delta检测(激活后)
        if (not _is_fx_proxy(delta)) and (not self._overflow_warned) and (not torch.isfinite(delta).all()):
            print("\n⚠️ SelfGated检测: 激活函数后产生非有限值!")
            print(f"   激活函数类型: {type(self.act).__name__}")
            print(f"   delta shape: {delta.shape}, dtype: {delta.dtype}")
            self._overflow_warned = True

        # u ⊙ delta
        gated_res_1 = _safe_gated_mul(feat_intrinsic, delta)
        # v = gate, (1-delta) ⊙ v
        gated_res_2 = _safe_gated_mul(gate, (1 - delta))

        feat_generated = self.gate_scale * (gated_res_1 + gated_res_2)
        
        # feat_generated检测
        if (not _is_fx_proxy(feat_generated)) and (not self._overflow_warned) and (not torch.isfinite(feat_generated).all()):
            print("\n⚠️ SelfGated检测: 门控乘法后产生非有限值!")
            print(f"   feat_generated shape: {feat_generated.shape}, dtype: {feat_generated.dtype}")
            finite_mask = torch.isfinite(feat_generated)
            if finite_mask.any():
                print(f"   有限值范围: [{feat_generated[finite_mask].min().item():.2f}, {feat_generated[finite_mask].max().item():.2f}]")
            print(f"   NaN数量: {torch.isnan(feat_generated).sum().item()}")
            print(f"   Inf数量: {torch.isinf(feat_generated).sum().item()}")
            self._overflow_warned = True

        out = torch.cat([feat_intrinsic, feat_generated], dim=1)
        
        # concat检测
        if (not _is_fx_proxy(out)) and (not self._overflow_warned) and (not torch.isfinite(out).all()):
            print("\n⚠️ SelfGated检测: concat后产生非有限值!")
            print(f"   out shape: {out.shape}, dtype: {out.dtype}")
            self._overflow_warned = True

        out = _safe_conv_bn(self.conv_out, self.bn_out, out)
        
        # 最终输出检测
        if (not _is_fx_proxy(out)) and (not self._overflow_warned) and (not torch.isfinite(out).all()):
            print("\n⚠️ SelfGated检测: conv_out+bn后产生非有限值!")
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
    - 输出：cat([主分支, gate_branch])
    - gate_branch = gate_scale * (u ⊙ delta + gate ⊙ (1 - delta))

    Args:
        channels: 减半后的通道数
        stride: 步长
        activation: 激活函数类
    """

    def __init__(
        self,
        channels: int,
        stride: int,
        activation: Activation,
        eps_reg_a: float = 0.1,
        enable_eps_reg: bool = True,
        poly_warmup_act: str = "sigmoid"
    ):
        super().__init__()

        # 主分支：3x3深度卷积
        self.dw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=3, stride=stride, padding=1,
            groups=channels, bias=False
        )
        self.bn = nn.BatchNorm2d(channels, eps=BN_EPS)

        # 门控分支：5x5深度卷积 + 谱归一化
        # 移除BN：BN会破坏SpectralNorm的Lipschitz约束
        # 启用bias：需卷积层自己学习偏置
        self.gate_conv = nn.Conv2d(
            channels, channels,
            kernel_size=5, stride=1, padding=2,
            groups=channels, bias=False
        )
        # 移除BN
        self.bn_gate = nn.BatchNorm2d(channels, eps=BN_EPS)
        
        use_sigmoid_gate = _is_swish_activation(activation)
        self.activation = nn.Sigmoid() if use_sigmoid_gate else activation()
        if hasattr(self.activation, "set_warmup_act") and poly_warmup_act is not None:
            self.activation.set_warmup_act(poly_warmup_act)
        
        self.gate_norm = nn.BatchNorm2d(channels, eps=BN_EPS)
        
        # 用于检测溢出的标志
        self._overflow_warned = False
        
        # 正则化损失缓存
        self.gate_reg_loss = torch.tensor(0.0)
        self.eps_reg_loss = torch.tensor(0.0)
        self.eps_reg_a = float(eps_reg_a)
        if use_sigmoid_gate:
            # Disable u-v regularization when using Swish->Sigmoid gate
            self.enable_eps_reg = False
            self.eps_reg_a = 0.0
        else:
            self.enable_eps_reg = bool(enable_eps_reg)

    def forward(self, x):
        # 输入检测
        if (not _is_fx_proxy(x)) and (not self._overflow_warned) and (not torch.isfinite(x).all()):
            print("\n⚠️ GatedDepthwiseConv输入检测: 输入包含非有限值!")
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
        if (not _is_fx_proxy(feat_intrinsic)) and (not self._overflow_warned) and (not torch.isfinite(feat_intrinsic).all()):
            print("\n⚠️ GatedDepthwiseConv检测: dw_conv+bn后产生非有限值!")
            print(f"   feat_intrinsic shape: {feat_intrinsic.shape}, dtype: {feat_intrinsic.dtype}")
            finite_mask = torch.isfinite(feat_intrinsic)
            if finite_mask.any():
                print(f"   有限值范围: [{feat_intrinsic[finite_mask].min().item():.2f}, {feat_intrinsic[finite_mask].max().item():.2f}]")
            print(f"   NaN数量: {torch.isnan(feat_intrinsic).sum().item()}")
            print(f"   Inf数量: {torch.isinf(feat_intrinsic).sum().item()}")
            self._overflow_warned = True
        # 门控分支
        gate = self.gate_conv(feat_intrinsic)
        gate = self.bn_gate(gate) # Added

        # epsilon 正则: u - v, w = 1 + a * v^2
        if self.enable_eps_reg and self.eps_reg_a > 0:
            v_reg = feat_intrinsic
            u_reg = gate
            # Skip dtype checks during FX tracing (Proxy has no concrete dtype).
            if (not _is_fx_proxy(v_reg)) and v_reg.dtype in (torch.float16, torch.bfloat16):
                v_reg = v_reg.float()
                u_reg = u_reg.float()
            eps = u_reg - v_reg
            w = 1.0 + self.eps_reg_a * (v_reg * v_reg)
            self.eps_reg_loss = (w * (eps * eps)).mean()
        else:
            self.eps_reg_loss = feat_intrinsic.new_tensor(0.0)
        
        # gate检测(激活前)
        if (not _is_fx_proxy(gate)) and (not self._overflow_warned) and (not torch.isfinite(gate).all()):
            print("\n⚠️ GatedDepthwiseConv检测: gate_conv+bn后产生非有限值!")
            print(f"   gate shape: {gate.shape}, dtype: {gate.dtype}")
            finite_mask = torch.isfinite(gate)
            if finite_mask.any():
                print(f"   有限值范围: [{gate[finite_mask].min().item():.2f}, {gate[finite_mask].max().item():.2f}]")
            print(f"   NaN数量: {torch.isnan(gate).sum().item()}")
            print(f"   Inf数量: {torch.isinf(gate).sum().item()}")
            self._overflow_warned = True
        
        # 结构性预缩放
        delta = gate * 0.125
        
        # delta = activation(raw)
        delta = self.activation(delta)
        
        # 计算激活正则化损失
        self.gate_reg_loss = (delta ** 2).mean()
        
        # delta检测(激活后)
        if (not _is_fx_proxy(delta)) and (not self._overflow_warned) and (not torch.isfinite(delta).all()):
            print("\n⚠️ GatedDepthwiseConv检测: 激活函数后产生非有限值!")
            print(f"   激活函数类型: {type(self.activation).__name__}")
            print(f"   delta shape: {delta.shape}, dtype: {delta.dtype}")
            self._overflow_warned = True

        # 生成门控特征 (SelfGated结构)
        gated_res_1 = _safe_gated_mul(feat_intrinsic, delta)
        gated_res_2 = _safe_gated_mul(gate, (1.0 - delta))
        feat_gated =  self.gate_norm(gated_res_1 + gated_res_2)
        
        # feat_gated检测
        if (not _is_fx_proxy(feat_gated)) and (not self._overflow_warned) and (not torch.isfinite(feat_gated).all()):
            print("\n⚠️ GatedDepthwiseConv检测: 门控乘法后产生非有限值!")
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
        if (not _is_fx_proxy(out)) and (not self._overflow_warned) and (not torch.isfinite(out).all()):
            print("\n⚠️ GatedDepthwiseConv检测: concat后产生非有限值!")
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
            self.bn1 = nn.BatchNorm2d(expansion_out_channels, eps=BN_EPS)

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
            self.bn2 = nn.BatchNorm2d(expansion_out_channels, eps=BN_EPS)
            self.dw_out_channels = expansion_out_channels

        # SE模块（可选）
        if use_se:
            self.se = SEBlock(self.dw_out_channels)

        # 1x1压缩层
        self.project_conv = nn.Conv2d(
            self.dw_out_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        
        # 关键优化：Init-Zero for Gated Branch
        if use_gated_dw:
            # dw_out_channels 是 concat 后的通道数
            # intrinsic_half = dw_out_channels // 2
            # gated_half starts at intrinsic_half
            intrinsic_half = self.dw_out_channels // 2
            
            # 初始化时让 project_conv 暂时“不看”gated_half
            # 这样初始状态下，网络表现得像没有加 gate 分支一样
            with torch.no_grad():
                self.project_conv.weight[:, intrinsic_half:, :, :] = 0

        self.activation = activation()

        # Shortcut连接
        self.use_shortcut = (stride == 1 and in_channels == out_channels)
        
        # 用于检测溢出的标志
        self._overflow_warned = False

    def forward(self, x):
        identity = x
        
        # 输入检测
        if (not _is_fx_proxy(x)) and (not self._overflow_warned) and (not torch.isfinite(x).all()):
            print("\n⚠️ MBConvBlock输入检测: 输入包含非有限值!")
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
            if (not _is_fx_proxy(out)) and (not self._overflow_warned) and (not torch.isfinite(out).all()):
                print("\n⚠️ MBConvBlock检测: expansion+bn后产生非有限值!")
                print(f"   out shape: {out.shape}, dtype: {out.dtype}")
                finite_mask = torch.isfinite(out)
                if finite_mask.any():
                    print(f"   有限值范围: [{out[finite_mask].min().item():.2f}, {out[finite_mask].max().item():.2f}]")
                print(f"   NaN数量: {torch.isnan(out).sum().item()}")
                print(f"   Inf数量: {torch.isinf(out).sum().item()}")
                self._overflow_warned = True
            
            out = self.activation(out)
            
            # activation后检测
            if (not _is_fx_proxy(out)) and (not self._overflow_warned) and (not torch.isfinite(out).all()):
                print("\n⚠️ MBConvBlock检测: expansion激活后产生非有限值!")
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
