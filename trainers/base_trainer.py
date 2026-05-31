"""
基础训练器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import time
import os
import csv
import sys
from tqdm import tqdm
from datetime import datetime
import pathlib

class Trainer:
    """基础训练器类"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        result_dir,
        epochs=60,
        scheduler=None,
        use_amp=True,
        save_freq=10,
        save_checkpoints=True,
        grad_clip_max_norm=1.0,
        poly4_warmup_ratio=0.5,
        resume_path=None,
        resume_strict=True,
        gate_reg_lambda=1e-3,
        eps_reg_lambda=0.0,
        poly4_range_lambda=0.0,
        poly4_deriv_lambda=0.0,
        poly4_range_r=2.0,
        poly4_deriv_L=3.0,
        nan_debug=False,
        val_force_fp32=True,
        val_batch_stats_path=None,
        val_batch_stats_quantile=0.999,
        val_batch_stats_anomaly_only=False,
        val_batch_stats_abs_logit_thresh=None,
        val_batch_stats_margin_thresh=None,
        val_batch_stats_loss_p999_thresh=None,
        smartpaf_progressive=False,
        smartpaf_start_epoch=None,
        smartpaf_group_epochs=5,
        smartpaf_transition_epochs=2,
        smartpaf_alternate_training=False,
        smartpaf_at_cycle_epochs=1,
        smartpaf_freeze_bn_during_poly_phase=True,
        collapse_guard_enabled=False,
        collapse_guard_drop=10.0,
        collapse_guard_patience=1,
        collapse_guard_action='warn',
        collapse_guard_lr_factor=0.2
    ):
        """
        初始化训练器

        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            device: 设备
            result_dir: 结果保存目录
            epochs: 训练epoch数
            scheduler: 学习率调度器
            use_amp: 是否使用混合精度训练
            save_freq: 保存检查点的频率
            grad_clip_max_norm: 梯度裁剪的最大范数，用于防止梯度爆炸
            poly4_warmup_ratio: StablePoly4的warmup比例（默认0.5，即50%的epoch用于warmup）
            gate_reg_lambda: 门控正则化权重
            eps_reg_lambda: epsilon正则化权重（u-v约束）
            poly4_range_lambda: StablePoly4输入范围正则权重
            poly4_deriv_lambda: StablePoly4导数正则权重
            poly4_range_r: StablePoly4输入范围阈值
            poly4_deriv_L: StablePoly4导数阈值
            nan_debug: 是否启用NaN定位钩子（默认关闭）
            val_force_fp32: 验证阶段强制使用FP32（禁用autocast）
            smartpaf_progressive: 是否逐层延迟启用 StablePoly4 多项式分支
            smartpaf_start_epoch: 第一个 StablePoly4 开始过渡的 epoch；默认使用 poly4 warmup 结束点
            smartpaf_group_epochs: 每个 StablePoly4 模块之间的启用间隔
            smartpaf_transition_epochs: 单个 StablePoly4 从 warmup 激活过渡到多项式的持续 epoch
            smartpaf_alternate_training: 是否按 epoch 交替训练普通权重和多项式系数
            smartpaf_at_cycle_epochs: AT 每个阶段持续的 epoch 数
            smartpaf_freeze_bn_during_poly_phase: AT 的多项式阶段是否冻结 BN 统计
            collapse_guard_enabled: 是否检测验证精度断崖式下降
            collapse_guard_drop: 相对上一轮或历史最佳下降超过多少百分点触发 guard
            collapse_guard_patience: 连续触发多少次后执行 action
            collapse_guard_action: warn/stop/restore_best_reduce_lr
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.result_dir = result_dir
        self.epochs = epochs
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.save_freq = save_freq
        self.save_checkpoints = save_checkpoints
        self.grad_clip_max_norm = grad_clip_max_norm
        self.poly4_warmup_ratio = poly4_warmup_ratio
        self.resume_path = resume_path
        self.resume_strict = resume_strict
        self.gate_reg_lambda = gate_reg_lambda
        self.eps_reg_lambda = eps_reg_lambda
        self.poly4_range_lambda = poly4_range_lambda
        self.poly4_deriv_lambda = poly4_deriv_lambda
        self.poly4_range_r = poly4_range_r
        self.poly4_deriv_L = poly4_deriv_L
        self.nan_debug = nan_debug
        self.val_force_fp32 = val_force_fp32
        self.val_batch_stats_path = val_batch_stats_path
        self.val_batch_stats_quantile = val_batch_stats_quantile
        self.val_batch_stats_anomaly_only = val_batch_stats_anomaly_only
        self.val_batch_stats_abs_logit_thresh = val_batch_stats_abs_logit_thresh
        self.val_batch_stats_margin_thresh = val_batch_stats_margin_thresh
        self.val_batch_stats_loss_p999_thresh = val_batch_stats_loss_p999_thresh
        self.smartpaf_progressive = bool(smartpaf_progressive)
        self.smartpaf_start_epoch = smartpaf_start_epoch
        self.smartpaf_group_epochs = smartpaf_group_epochs
        self.smartpaf_transition_epochs = max(0.0, float(smartpaf_transition_epochs))
        self.smartpaf_alternate_training = bool(smartpaf_alternate_training)
        self.smartpaf_at_cycle_epochs = max(1, int(smartpaf_at_cycle_epochs))
        self.smartpaf_freeze_bn_during_poly_phase = bool(smartpaf_freeze_bn_during_poly_phase)
        self.collapse_guard_enabled = bool(collapse_guard_enabled)
        self.collapse_guard_drop = float(collapse_guard_drop)
        self.collapse_guard_patience = max(1, int(collapse_guard_patience))
        self.collapse_guard_action = str(collapse_guard_action).strip().lower()
        self.collapse_guard_lr_factor = float(collapse_guard_lr_factor)
        self._collapse_guard_hits = 0
        self._last_train_stats = {}
        self._last_val_stats = {}
        self._smartpaf_poly_modules = []
        self._smartpaf_poly_param_ids = set()
        self._smartpaf_last_phase = None
        self._nan_debug_running = False
        self._nan_hooks = []
        self._nan_triggered = False
        self._nan_debug_active = False

        # 创建结果目录
        os.makedirs(result_dir, exist_ok=True)

        # 自动调整StablePoly4的warmup_epochs
        self._adjust_poly4_warmup()
        self._configure_poly4_modules()
        self._configure_smartpaf_modules()

        # 初始化scaler
        self.scaler = GradScaler() if use_amp else None

        # 训练历史
        self.start_epoch = 1
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': [],
            'train_valid_batches': [],
            'train_skipped_batches': [],
            'val_valid_batches': [],
            'val_skipped_batches': [],
            'nonfinite_train_batches': [],
            'nonfinite_val_batches': [],
            'smartpaf_phase': [],
            'collapse_guard_triggered': []
        }

        # 最佳准确率
        self.best_acc = 0.0

        if self.resume_path:
            self._load_checkpoint(self.resume_path, strict=self.resume_strict)

    def _ensure_history_fields(self):
        """Ensure older checkpoints have all current history columns."""
        fields = [
            'epoch',
            'train_loss',
            'train_acc',
            'val_loss',
            'val_acc',
            'learning_rate',
            'epoch_time',
            'train_valid_batches',
            'train_skipped_batches',
            'val_valid_batches',
            'val_skipped_batches',
            'nonfinite_train_batches',
            'nonfinite_val_batches',
            'smartpaf_phase',
            'collapse_guard_triggered',
        ]
        existing_len = len(self.history.get('epoch', []))
        for key in fields:
            if key not in self.history:
                fill = '' if key == 'smartpaf_phase' else 0
                self.history[key] = [fill for _ in range(existing_len)]
            elif len(self.history[key]) < existing_len:
                fill = '' if key == 'smartpaf_phase' else 0
                self.history[key].extend([fill for _ in range(existing_len - len(self.history[key]))])

    def _find_nonfinite_tensor(self, obj):
        if torch.is_tensor(obj):
            if not torch.isfinite(obj).all().item():
                return obj
            return None
        if isinstance(obj, (list, tuple)):
            for item in obj:
                t = self._find_nonfinite_tensor(item)
                if t is not None:
                    return t
        if isinstance(obj, dict):
            for item in obj.values():
                t = self._find_nonfinite_tensor(item)
                if t is not None:
                    return t
        return None

    def _tensor_nonfinite_stats(self, t):
        finite_mask = torch.isfinite(t)
        bad_count = (~finite_mask).sum().item()
        numel = t.numel()
        if finite_mask.any().item():
            finite_vals = t[finite_mask]
            min_val = finite_vals.min().item()
            max_val = finite_vals.max().item()
        else:
            min_val = float("nan")
            max_val = float("nan")
        return {
            "shape": tuple(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "bad_count": bad_count,
            "numel": numel,
            "min": min_val,
            "max": max_val,
        }

    def _register_nan_hooks(self):
        self._nan_hooks = []
        self._nan_triggered = False

        def hook_fn(name):
            def _hook(module, inputs, output):
                if self._nan_triggered:
                    return
                bad_tensor = self._find_nonfinite_tensor(inputs)
                location = "input"
                if bad_tensor is None:
                    bad_tensor = self._find_nonfinite_tensor(output)
                    location = "output"
                if bad_tensor is None:
                    return
                self._nan_triggered = True
                stats = self._tensor_nonfinite_stats(bad_tensor)
                print(
                    f"Non-finite {location} detected in "
                    f"{name} ({module.__class__.__name__}): "
                    f"shape={stats['shape']} dtype={stats['dtype']} "
                    f"device={stats['device']} bad={stats['bad_count']}/{stats['numel']} "
                    f"finite_min={stats['min']:.6g} finite_max={stats['max']:.6g}"
                )
                raise RuntimeError("Non-finite detected during forward pass")
            return _hook

        for name, module in self.model.named_modules():
            if name == "":
                continue
            if any(True for _ in module.children()):
                continue
            self._nan_hooks.append(module.register_forward_hook(hook_fn(name)))
        self._nan_debug_active = True

    def _remove_nan_hooks(self):
        for handle in self._nan_hooks:
            handle.remove()
        self._nan_hooks = []
        self._nan_debug_active = False

    def _debug_nan_forward(self, images):
        if not self.nan_debug or self._nan_debug_running:
            return
        self._nan_debug_running = True
        print("Non-finite output detected, running forward with NaN hooks...")
        self._register_nan_hooks()
        try:
            with torch.no_grad():
                _ = self.model(images)
        finally:
            self._remove_nan_hooks()
            self._nan_debug_running = False

    def _is_tensor_finite(self, tensor):
        return torch.is_tensor(tensor) and torch.isfinite(tensor).all().item()

    def _current_smartpaf_phase(self, epoch):
        if not self.smartpaf_alternate_training or not self._smartpaf_poly_param_ids:
            return 'disabled'
        return 'poly' if self._is_smartpaf_poly_phase(epoch) else 'weights'

    def _zero_all_gradients(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.model.zero_grad(set_to_none=True)

    def _adjust_poly4_warmup(self):
        """
        自动调整模型中所有StablePoly4的warmup_epochs

        根据训练总epoch数和warmup_ratio，动态设置合适的warmup_epochs。
        例如：训练25 epochs，ratio=0.5 → warmup_epochs=12
              训练60 epochs，ratio=0.5 → warmup_epochs=30
        """
        # 计算目标warmup_epochs
        target_warmup_epochs = int(self.epochs * self.poly4_warmup_ratio)

        # 确保至少有几个epoch用于warmup（最少5个epoch）
        target_warmup_epochs = max(5, target_warmup_epochs)

        # 确保warmup不超过总epoch数的80%
        target_warmup_epochs = min(target_warmup_epochs, int(self.epochs * 0.8))
        self.poly4_warmup_epochs = target_warmup_epochs

        poly4_count = 0
        for module in self.model.modules():
            # 检查是否是StablePoly4（通过检查是否有set_warmup_epochs方法）
            if hasattr(module, 'set_warmup_epochs') and callable(module.set_warmup_epochs):
                module.set_warmup_epochs(target_warmup_epochs)
                poly4_count += 1

        if poly4_count > 0:
            print(f"✓ 自动调整 {poly4_count} 个StablePoly4模块:")
            print(f"  - 训练总epoch: {self.epochs}")
            print(f"  - Warmup比例: {self.poly4_warmup_ratio:.1%}")
            print(f"  - Warmup epochs: {target_warmup_epochs}")
            print(f"  - 多项式激活将在第 {target_warmup_epochs + 1} epoch开始生效")

    def _configure_poly4_modules(self):
        """
        配置StablePoly4的alpha调度与正则项开关/阈值
        """
        poly4_count = 0
        for module in self.model.modules():
            if hasattr(module, 'set_range_params') and callable(module.set_range_params):
                module.set_range_params(
                    range_r=self.poly4_range_r,
                    enable=self.poly4_range_lambda > 0,
                )
            if hasattr(module, 'set_deriv_params') and callable(module.set_deriv_params):
                module.set_deriv_params(
                    deriv_L=self.poly4_deriv_L,
                    enable=self.poly4_deriv_lambda > 0,
                )
            if hasattr(module, 'set_warmup_epochs') and callable(module.set_warmup_epochs):
                poly4_count += 1

        if poly4_count > 0:
            print("✓ StablePoly4正则/调度配置:")
            if self.poly4_range_lambda > 0:
                print(f"  - range_r: {self.poly4_range_r}, lambda_range: {self.poly4_range_lambda}")
            if self.poly4_deriv_lambda > 0:
                print(f"  - deriv_L: {self.poly4_deriv_L}, lambda_deriv: {self.poly4_deriv_lambda}")

    def _configure_smartpaf_modules(self):
        """配置 SmartPAF-lite 的逐层多项式启用与 AT 参数集合。"""
        self._smartpaf_poly_modules = []
        self._smartpaf_poly_param_ids = set()

        for name, module in self.model.named_modules():
            if hasattr(module, "set_poly_schedule") and callable(module.set_poly_schedule):
                self._smartpaf_poly_modules.append((name, module))
                for param in module.parameters(recurse=True):
                    self._smartpaf_poly_param_ids.add(id(param))

        if not self._smartpaf_poly_modules:
            return

        if self.smartpaf_progressive:
            start_epoch = self.smartpaf_start_epoch
            if start_epoch is None:
                start_epoch = getattr(self, "poly4_warmup_epochs", 0)
            start_epoch = float(start_epoch)

            if isinstance(self.smartpaf_group_epochs, str) and self.smartpaf_group_epochs.lower() == 'auto':
                available = max(1.0, self.epochs - start_epoch - max(0.0, self.smartpaf_transition_epochs))
                if len(self._smartpaf_poly_modules) > 1:
                    group_epochs = max(1.0, available / (len(self._smartpaf_poly_modules) - 1))
                else:
                    group_epochs = 1.0
            else:
                group_epochs = max(1.0, float(self.smartpaf_group_epochs))
            self.smartpaf_group_epochs = group_epochs

            print("✓ SmartPAF-lite逐层启用配置:")
            print(f"  - StablePoly4模块数: {len(self._smartpaf_poly_modules)}")
            print(f"  - 第一个模块开始epoch: {start_epoch:g}")
            print(f"  - 模块间隔epoch: {group_epochs:g}")
            print(f"  - 单模块过渡epoch: {self.smartpaf_transition_epochs:g}")
            for idx, (name, module) in enumerate(self._smartpaf_poly_modules):
                module_start = start_epoch + idx * group_epochs
                module.set_poly_schedule(
                    start_epoch=module_start,
                    transition_epochs=self.smartpaf_transition_epochs,
                )
                if idx < 8:
                    print(f"    {idx + 1}. {name}: start={module_start:g}")
            if len(self._smartpaf_poly_modules) > 8:
                print(f"    ... 其余 {len(self._smartpaf_poly_modules) - 8} 个模块按相同间隔继续")

        if self.smartpaf_alternate_training:
            print("✓ SmartPAF-lite AT配置:")
            print(f"  - 每阶段epoch: {self.smartpaf_at_cycle_epochs}")
            print("  - 阶段顺序: 普通权重 -> 多项式系数 -> ...")

    def _progress_bar(self, iterable, **kwargs):
        """
        Keep progress bars interactive-only. Captured non-TTY logs store every tqdm
        refresh as a new line, which can grow task logs quickly during long runs.
        """
        force = os.environ.get("FORCE_TQDM", "").lower() in {"1", "true", "yes", "on"}
        disable = (not force) and (not sys.stderr.isatty())
        kwargs.setdefault("leave", False)
        kwargs.setdefault("disable", disable)
        kwargs.setdefault("mininterval", 5.0)
        return tqdm(iterable, **kwargs)

    def _is_smartpaf_poly_phase(self, epoch):
        if not self.smartpaf_alternate_training:
            return False
        phase_idx = (max(1, int(epoch)) - 1) // self.smartpaf_at_cycle_epochs
        return phase_idx % 2 == 1

    def _set_batchnorm_eval(self):
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def _apply_smartpaf_training_mode(self, epoch):
        """按 AT 阶段切换参数 requires_grad。"""
        if not self.smartpaf_alternate_training or not self._smartpaf_poly_param_ids:
            return

        train_poly = self._is_smartpaf_poly_phase(epoch)
        phase = "poly" if train_poly else "weights"
        if phase != self._smartpaf_last_phase:
            print(f"SmartPAF-lite AT阶段: {phase}")
            self._smartpaf_last_phase = phase
            self._zero_all_gradients()

        for param in self.model.parameters():
            is_poly_param = id(param) in self._smartpaf_poly_param_ids
            param.requires_grad = is_poly_param if train_poly else not is_poly_param

        if train_poly and self.smartpaf_freeze_bn_during_poly_phase:
            self._set_batchnorm_eval()

    def _restore_all_trainable(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def _calc_eps_reg_weight(self, epoch, batch_idx, steps_per_epoch):
        if self.eps_reg_lambda <= 0:
            return 0.0
        warmup_end = getattr(self, "poly4_warmup_epochs", None)
        if warmup_end is None:
            warmup_end = int(self.epochs * self.poly4_warmup_ratio)
            warmup_end = max(0, min(warmup_end, self.epochs))
        if self.epochs <= warmup_end:
            return 0.0
        steps = max(1, int(steps_per_epoch))
        epoch_progress = (epoch - 1) + (batch_idx + 1) / steps
        if epoch_progress <= warmup_end:
            return 0.0
        if epoch_progress >= self.epochs:
            return 1.0
        return (epoch_progress - warmup_end) / (self.epochs - warmup_end)

    def _load_checkpoint(self, path, strict=True):
        if not os.path.exists(path):
            print(f"Warning: checkpoint not found: {path}")
            return
        checkpoint = torch.load(path, map_location=self.device)

        model_state = checkpoint.get('model_state_dict')
        if model_state is not None:
            self.model.load_state_dict(model_state, strict=strict)

        optim_state = checkpoint.get('optimizer_state_dict')
        if optim_state is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(optim_state)

        sched_state = checkpoint.get('scheduler_state_dict')
        if sched_state is not None and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(sched_state)
            except Exception as exc:
                print(f"Warning: failed to load scheduler state: {exc}")

        scaler_state = checkpoint.get('scaler_state_dict')
        if scaler_state is not None and self.scaler is not None:
            try:
                self.scaler.load_state_dict(scaler_state)
            except Exception as exc:
                print(f"Warning: failed to load GradScaler state: {exc}")

        self.best_acc = checkpoint.get('best_acc', self.best_acc)

        history = checkpoint.get('history')
        if isinstance(history, dict):
            self.history = history
            self._ensure_history_fields()

        epoch = checkpoint.get('epoch')
        if isinstance(epoch, int):
            self.start_epoch = max(1, epoch + 1)
            print(f"Resumed from epoch {epoch}. Next epoch: {self.start_epoch}")

    def _set_epoch_for_model(self, epoch):
        """
        递归地为模型中所有需要 epoch 信息的模块设置 epoch
        
        这个方法会遍历模型的所有子模块，找到所有具有 set_epoch 方法的模块
        （例如 StablePoly4 激活函数）并调用它们的 set_epoch 方法。
        
        Args:
            epoch: 当前训练的 epoch 编号
        """
        # 使用 model.modules() 获取所有子模块（包括模型自身）
        for module in self.model.modules():
            # 检查模块是否有 set_epoch 方法
            if hasattr(module, 'set_epoch') and callable(module.set_epoch):
                # 调用 set_epoch 方法
                module.set_epoch(epoch)

    def _set_epoch_progress_for_model(self, epoch, step_idx, steps_per_epoch):
        """
        递归地为模型中所有需要 epoch 进度信息的模块设置细粒度进度

        Args:
            epoch: 当前 epoch 编号（1-based）
            step_idx: 当前 batch 索引（0-based）
            steps_per_epoch: 每个 epoch 的 batch 数
        """
        for module in self.model.modules():
            if hasattr(module, 'set_epoch_progress') and callable(module.set_epoch_progress):
                module.set_epoch_progress(epoch, step_idx, steps_per_epoch)

    def _log_poly4_params(self, epoch=None):
        """进入验证时打印 StablePoly4 的关键参数"""
        header = f"StablePoly4参数 (Epoch {epoch})" if epoch is not None else "StablePoly4参数"
        printed = 0
        for name, module in self.model.named_modules():
            if not all(hasattr(module, attr) for attr in ('a', 'b', 'c', 'd', 'e', 'log_in_scale')):
                continue
            try:
                a = module.a.detach().float().cpu().item()
                b = module.b.detach().float().cpu().item()
                c = module.c.detach().float().cpu().item()
                d = module.d.detach().float().cpu().item()
                e = module.e.detach().float().cpu().item()
                log_in = module.log_in_scale.detach().float().cpu().item()
                log_in_clamped = max(-6.0, min(2.0, log_in))
                in_scale = float(torch.exp(torch.tensor(log_in_clamped)))
                out_scale = float(getattr(module, "output_scale", 1.0))
                warmup_epochs = int(getattr(module, "warmup_epochs", 0))
                cur_epoch = int(module.current_epoch.item()) if hasattr(module, "current_epoch") else None
            except Exception:
                continue
            if printed == 0:
                print(f"\n{header}:")
            printed += 1
            print(
                f"  - {name}: "
                f"a={a:.4g} b={b:.4g} c={c:.4g} d={d:.4g} e={e:.4g} "
                f"log_in_scale={log_in:.4g} in_scale≈{in_scale:.4g} "
                f"output_scale={out_scale:.4g} warmup_epochs={warmup_epochs}"
                + (f" current_epoch={cur_epoch}" if cur_epoch is not None else "")
            )
        if printed == 0:
            print("\nStablePoly4参数: 未检测到 StablePoly4 模块")

    def _set_poly4_collect_stats(self, enabled: bool):
        for module in self.model.modules():
            if hasattr(module, 'set_collect_stats') and callable(module.set_collect_stats):
                module.set_collect_stats(enabled)

    def _log_poly4_stats(self, epoch=None):
        header = f"StablePoly4诊断统计 (Epoch {epoch})" if epoch is not None else "StablePoly4诊断统计"
        printed = 0
        for name, module in self.model.named_modules():
            if not hasattr(module, "last_x_poly_stats"):
                continue
            x_stats = getattr(module, "last_x_poly_stats", None)
            f_stats = getattr(module, "last_fprime_stats", None)
            if x_stats is None and f_stats is None:
                continue
            if printed == 0:
                print(f"\n{header}:")
            printed += 1
            if x_stats is not None:
                print(
                    f"  - {name}: |x_poly| p50={x_stats['p50']:.4g} "
                    f"p90={x_stats['p90']:.4g} p99={x_stats['p99']:.4g} max={x_stats['max']:.4g}"
                )
            if f_stats is not None:
                print(
                    f"    |f'| p50={f_stats['p50']:.4g} "
                    f"p90={f_stats['p90']:.4g} p99={f_stats['p99']:.4g} max={f_stats['max']:.4g}"
                )
        if printed == 0:
            print("\nStablePoly4诊断统计: 未检测到 StablePoly4 模块")
    
    def train_one_epoch(self, epoch):
        """
        训练一个epoch

        Returns:
            avg_loss: 平均损失
            avg_acc: 平均准确率
        """
        self.model.train()
        self._apply_smartpaf_training_mode(epoch)

        total_loss = 0.0
        correct = 0
        total = 0
        valid_batches = 0
        skipped_batches = 0
        nonfinite_batches = 0

        model_name = pathlib.Path(self.result_dir).stem
        pbar = self._progress_bar(self.train_loader, desc=f'Epoch [{epoch}]({model_name})')
        first_batch_diagnostic = (epoch == 1)

        for batch_idx, (images, labels) in enumerate(pbar):
            self._set_epoch_progress_for_model(epoch, batch_idx, len(self.train_loader))
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if first_batch_diagnostic and batch_idx == 0:
                print(f"\n{'='*60}")
                print(f"第一个batch诊断 (Epoch {epoch}):")
                print(f"  Batch shape: {images.shape}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Labels范围: [{labels.min().item()}, {labels.max().item()}]")
                print(f"  Labels dtype: {labels.dtype}")
                print(f"  唯一标签数: {len(labels.unique())}")
                print(f"{'='*60}\n")

            self._zero_all_gradients()

            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
            with autocast(device_type=device_type, enabled=self.use_amp):
                outputs = self.model(images)

                if first_batch_diagnostic and batch_idx == 0:
                    print("模型输出诊断:")
                    print(f"  Output shape: {outputs.shape}")
                    print(f"  Output dtype: {outputs.dtype}")
                    if torch.isfinite(outputs).any().item():
                        finite_outputs = outputs[torch.isfinite(outputs)]
                        print(f"  Output有限值范围: [{finite_outputs.min().item():.2f}, {finite_outputs.max().item():.2f}]")
                    print(f"  Output包含NaN: {torch.isnan(outputs).any().item()}")
                    print(f"  Output包含Inf: {torch.isinf(outputs).any().item()}")

                outputs_fp32 = outputs.float()

                if first_batch_diagnostic and batch_idx == 0:
                    print("\nLoss计算前:")
                    if torch.isfinite(outputs_fp32).any().item():
                        finite_outputs = outputs_fp32[torch.isfinite(outputs_fp32)]
                        print(f"  outputs_fp32有限值范围: [{finite_outputs.min().item():.2f}, {finite_outputs.max().item():.2f}]")
                    print(f"  outputs_fp32 dtype: {outputs_fp32.dtype}")
                    print(f"  labels范围: [{labels.min().item()}, {labels.max().item()}]")
                    print(f"  期望类别数: 0 到 {outputs.shape[1] - 1}")
                    if labels.max().item() >= outputs.shape[1]:
                        print(f"  ❌ 错误: 标签 {labels.max().item()} 超出输出维度 {outputs.shape[1]}!")

                reg_loss = outputs_fp32.new_tensor(0.0)

                if self.gate_reg_lambda > 0:
                    gate_reg = outputs_fp32.new_tensor(0.0)
                    for module in self.model.modules():
                        if hasattr(module, 'gate_reg_loss'):
                            gate_reg = gate_reg + module.gate_reg_loss
                    reg_loss = reg_loss + gate_reg * self.gate_reg_lambda
                else:
                    gate_reg = outputs_fp32.new_tensor(0.0)

                if self.poly4_range_lambda > 0:
                    range_reg = outputs_fp32.new_tensor(0.0)
                    for module in self.model.modules():
                        if hasattr(module, 'range_loss'):
                            range_reg = range_reg + module.range_loss
                    reg_loss = reg_loss + range_reg * self.poly4_range_lambda
                else:
                    range_reg = outputs_fp32.new_tensor(0.0)

                if self.poly4_deriv_lambda > 0:
                    deriv_reg = outputs_fp32.new_tensor(0.0)
                    for module in self.model.modules():
                        if hasattr(module, 'deriv_loss'):
                            deriv_reg = deriv_reg + module.deriv_loss
                    reg_loss = reg_loss + deriv_reg * self.poly4_deriv_lambda
                else:
                    deriv_reg = outputs_fp32.new_tensor(0.0)

                if self.eps_reg_lambda > 0:
                    eps_reg = outputs_fp32.new_tensor(0.0)
                    for module in self.model.modules():
                        if hasattr(module, 'eps_reg_loss'):
                            eps_reg = eps_reg + module.eps_reg_loss
                    eps_weight = self._calc_eps_reg_weight(epoch, batch_idx, len(self.train_loader))
                    reg_loss = reg_loss + eps_reg * self.eps_reg_lambda * eps_weight
                else:
                    eps_reg = outputs_fp32.new_tensor(0.0)
                    eps_weight = 0.0

                loss = self.criterion(outputs_fp32, labels) + reg_loss

                if first_batch_diagnostic and batch_idx == 0:
                    print("\nLoss计算后:")
                    print(f"  Loss value: {loss.item():.6f}" if torch.isfinite(loss).all().item() else f"  Loss value: {loss.item()}")
                    if self.poly4_range_lambda > 0 or self.poly4_deriv_lambda > 0:
                        print(f"  Range reg: {range_reg.item():.6f} (λ={self.poly4_range_lambda})")
                        print(f"  Deriv reg: {deriv_reg.item():.6f} (λ={self.poly4_deriv_lambda})")
                    if self.eps_reg_lambda > 0:
                        print(f"  Eps reg: {eps_reg.item():.6f} (λ={self.eps_reg_lambda}, w={eps_weight:.3f})")
                    print(f"  Loss is finite: {torch.isfinite(loss).all().item()}")
                    print(f"{'='*60}\n")
                    first_batch_diagnostic = False

            if self.nan_debug and not self._nan_debug_running:
                if self._find_nonfinite_tensor(outputs) is not None:
                    print("Non-finite output detected in forward")
                    raise RuntimeError("Non-finite output detected")
                if not torch.isfinite(loss).all().item():
                    print("Non-finite loss detected before backward")
                    raise RuntimeError("Non-finite loss detected")

            if not torch.isfinite(outputs_fp32).all().item() or not torch.isfinite(loss).all().item():
                nonfinite_batches += 1
                skipped_batches += 1
                self._zero_all_gradients()
                self._log_poly4_stats(epoch=epoch)
                print(f"\n⚠ Warning: Non-finite output/loss before backward; skipping train batch {batch_idx}.")
                continue

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip_max_norm,
                    error_if_nonfinite=False,
                )
                if not torch.isfinite(grad_norm).all().item():
                    nonfinite_batches += 1
                    skipped_batches += 1
                    self._zero_all_gradients()
                    self.scaler.update()
                    print(f"\n⚠ Warning: Non-finite grad norm ({grad_norm}) in train batch {batch_idx}; skipping optimizer step.")
                    continue
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip_max_norm,
                    error_if_nonfinite=False,
                )
                if not torch.isfinite(grad_norm).all().item():
                    nonfinite_batches += 1
                    skipped_batches += 1
                    self._zero_all_gradients()
                    print(f"\n⚠ Warning: Non-finite grad norm ({grad_norm}) in train batch {batch_idx}; skipping optimizer step.")
                    continue
                self.optimizer.step()

            loss_value = loss.item()
            total_loss += loss_value
            valid_batches += 1
            _, predicted = outputs_fp32.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'reg': f'{reg_loss.item():.4f}' if isinstance(reg_loss, torch.Tensor) else '0.00',
                'acc': f'{100. * correct / total:.2f}%',
                'skip': skipped_batches,
            })

        self._last_train_stats = {
            'valid_batches': valid_batches,
            'skipped_batches': skipped_batches,
            'nonfinite_batches': nonfinite_batches,
        }

        if total == 0 or valid_batches == 0:
            print("\n⚠ Warning: No valid training samples!")
            return float('inf'), 0.0

        avg_loss = total_loss / valid_batches
        avg_acc = 100. * correct / total
        return avg_loss, avg_acc

    def validate(self, epoch=None):
        """
        验证模型
        
        Args:
            epoch: 当前epoch（用于诊断输出）        Returns:
            avg_loss: 平均损失
            avg_acc: 平均准确率
        """
        self.model.eval()

        # 进入验证时打印 StablePoly4 参数
        self._log_poly4_params(epoch=epoch)
        # 启用统计，若发现 NaN 将打印对应 batch 的 |x_poly| / |f'|
        self._set_poly4_collect_stats(True)
        
        total_loss = 0.0
        correct = 0
        total = 0
        valid_batches = 0
        skipped_batches = 0
        nonfinite_batches = 0

        stats_file = None
        stats_writer = None
        if self.val_batch_stats_path:
            if self.val_batch_stats_anomaly_only:
                if self.val_batch_stats_abs_logit_thresh is None:
                    self.val_batch_stats_abs_logit_thresh = 50.0
                if self.val_batch_stats_margin_thresh is None:
                    self.val_batch_stats_margin_thresh = 50.0
                if self.val_batch_stats_loss_p999_thresh is None:
                    self.val_batch_stats_loss_p999_thresh = 50.0
            try:
                write_header = not os.path.exists(self.val_batch_stats_path) or os.path.getsize(self.val_batch_stats_path) == 0
            except OSError:
                write_header = True
            stats_file = open(self.val_batch_stats_path, 'a', newline='')
            fieldnames = [
                'epoch',
                'batch',
                'num_samples',
                'max_abs_logit',
                'max_margin',
                'loss_p999'
            ]
            stats_writer = csv.DictWriter(stats_file, fieldnames=fieldnames)
            if write_header:
                stats_writer.writeheader()

        with torch.no_grad():
            pbar = self._progress_bar(self.val_loader, desc='Validating')
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.val_force_fp32 and images.dtype != torch.float32:
                    images = images.float()

                device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                use_autocast = self.use_amp and not self.val_force_fp32
                with autocast(device_type=device_type, enabled=use_autocast):
                    outputs = self.model(images)
                
                # 重要：将 outputs 转回 float32 再计算 loss
                # AMP 下 outputs 可能是 float16，大的 logits 值会导致 log_softmax 溢出
                if self.val_force_fp32:
                    outputs_fp32 = outputs
                else:
                    outputs_fp32 = outputs.float() if self.use_amp else outputs
                loss = self.criterion(outputs_fp32, labels)
                
                # Check for NaN/Inf in loss and outputs
                loss_value = loss.item()
                if not torch.isfinite(loss).all() or not torch.isfinite(outputs).all():
                    loss_value = float('nan')
                    nonfinite_batches += 1
                    skipped_batches += 1
                    # 打印当前 batch 的 StablePoly4 统计
                    self._log_poly4_stats(epoch=epoch)
                    # 详细诊断信息
                    out_min = outputs.min().item() if torch.isfinite(outputs.min()) else float('nan')
                    out_max = outputs.max().item() if torch.isfinite(outputs.max()) else float('nan')
                    nan_count = (~torch.isfinite(outputs)).sum().item()
                    print(f"\n⚠ Warning: Non-finite values detected in validation!{f' (Epoch {epoch})' if epoch else ''}")
                    print(f"  Loss: {loss.item() if torch.isfinite(loss).all().item() else loss_value}")
                    print(f"  Output shape: {outputs.shape}")
                    print(f"  Output stats: min={out_min:.2f}, max={out_max:.2f}, nan_count={nan_count}/{outputs.numel()}")
                    
                    # 打印前几个样本的logits分布
                    print(f"\n  前3个样本的logits分析:")
                    for i in range(min(3, outputs.shape[0])):
                        sample_logits = outputs[i]  # shape: (num_classes,)
                        finite_mask = torch.isfinite(sample_logits)
                        num_finite = finite_mask.sum().item()
                        num_nan = torch.isnan(sample_logits).sum().item()
                        num_inf = torch.isinf(sample_logits).sum().item()
                        
                        print(f"    样本 {i}: finite={num_finite}/{len(sample_logits)}, NaN={num_nan}, Inf={num_inf}")
                        
                        if num_finite > 0:
                            finite_logits = sample_logits[finite_mask]
                            print(f"      有限值范围: [{finite_logits.min().item():.2f}, {finite_logits.max().item():.2f}]")
                        
                        # 显示前10个logit值
                        logit_preview = sample_logits[:10].cpu().numpy()
                        print(f"      前10个logits: {logit_preview}")
                    
                    # Skip this batch
                    if stats_writer is not None:
                        stats_writer.writerow({
                            'epoch': epoch if epoch is not None else '',
                            'batch': batch_idx,
                            'num_samples': labels.size(0),
                            'max_abs_logit': float('nan'),
                            'max_margin': float('nan'),
                            'loss_p999': float('nan')
                        })
                    continue

                if stats_writer is not None:
                    max_abs_logit = outputs_fp32.abs().max().item()
                    max_logits = outputs_fp32.max(dim=1).values
                    true_logits = outputs_fp32.gather(1, labels.unsqueeze(1)).squeeze(1)
                    margins = max_logits - true_logits
                    max_margin = margins.max().item()
                    loss_per_sample = F.cross_entropy(outputs_fp32, labels, reduction='none')
                    loss_p999 = torch.quantile(loss_per_sample.float(), self.val_batch_stats_quantile).item()
                    is_anomalous = True
                    if self.val_batch_stats_anomaly_only:
                        is_anomalous = (
                            (self.val_batch_stats_abs_logit_thresh is not None and max_abs_logit >= self.val_batch_stats_abs_logit_thresh)
                            or (self.val_batch_stats_margin_thresh is not None and max_margin >= self.val_batch_stats_margin_thresh)
                            or (self.val_batch_stats_loss_p999_thresh is not None and loss_p999 >= self.val_batch_stats_loss_p999_thresh)
                        )
                    if is_anomalous:
                        stats_writer.writerow({
                            'epoch': epoch if epoch is not None else '',
                            'batch': batch_idx,
                            'num_samples': labels.size(0),
                            'max_abs_logit': max_abs_logit,
                            'max_margin': max_margin,
                            'loss_p999': loss_p999
                        })
                
                total_loss += loss_value
                valid_batches += 1
                _, predicted = outputs_fp32.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%',
                    'skip': skipped_batches,
                })
        
        self._last_val_stats = {
            'valid_batches': valid_batches,
            'skipped_batches': skipped_batches,
            'nonfinite_batches': nonfinite_batches,
        }

        # Safety check: ensure we have valid data
        if total == 0 or valid_batches == 0:
            print("\n⚠ Warning: No valid samples in validation!")
            if stats_file is not None:
                stats_file.close()
            self._set_poly4_collect_stats(False)
            return float('inf'), 0.0

        avg_loss = total_loss / valid_batches
        avg_acc = 100. * correct / total
        
        # Final NaN check
        if not torch.isfinite(torch.tensor(avg_loss)):
            print(f"\n⚠ Warning: avg_loss is NaN/Inf! Setting to inf.")
            avg_loss = float('inf')
        
        if stats_file is not None:
            stats_file.close()
        self._set_poly4_collect_stats(False)

        return avg_loss, avg_acc
    
    def _run_collapse_guard(self, epoch, val_acc):
        if not self.collapse_guard_enabled:
            return False

        prev_acc = self.history['val_acc'][-1] if self.history.get('val_acc') else None
        best_before = self.best_acc
        drops = []
        if prev_acc is not None:
            drops.append(prev_acc - val_acc)
        if best_before is not None:
            drops.append(best_before - val_acc)
        max_drop = max(drops) if drops else 0.0
        triggered = max_drop >= self.collapse_guard_drop
        if not triggered:
            self._collapse_guard_hits = 0
            return False

        self._collapse_guard_hits += 1
        print(f"\n⚠ Collapse guard triggered at epoch {epoch}: val_acc={val_acc:.2f}%, max_drop={max_drop:.2f} pp")
        print(f"  - action: {self.collapse_guard_action}")
        print(f"  - consecutive hits: {self._collapse_guard_hits}/{self.collapse_guard_patience}")
        print(f"  - SmartPAF phase: {self._current_smartpaf_phase(epoch)}")
        print(f"  - LR: {[group.get('lr') for group in self.optimizer.param_groups]}")
        print(f"  - train stats: {self._last_train_stats}")
        print(f"  - val stats: {self._last_val_stats}")
        self._log_poly4_stats(epoch=epoch)

        if self.save_checkpoints:
            self.save_checkpoint(epoch, is_best=False, filename=f'collapse_epoch_{epoch}.pth')

        if self._collapse_guard_hits < self.collapse_guard_patience:
            return True

        if self.collapse_guard_action == 'stop':
            raise RuntimeError(f"Collapse guard stopped training at epoch {epoch}")
        if self.collapse_guard_action == 'restore_best_reduce_lr':
            best_path = os.path.join(self.result_dir, 'best_model.pth')
            if os.path.exists(best_path):
                checkpoint = torch.load(best_path, map_location=self.device)
                model_state = checkpoint.get('model_state_dict')
                if model_state is not None:
                    self.model.load_state_dict(model_state, strict=self.resume_strict)
                    print(f"  ✓ Restored best model from {best_path}")
            for group in self.optimizer.param_groups:
                group['lr'] *= self.collapse_guard_lr_factor
            if self.scaler is not None:
                self.scaler = GradScaler()
            print(f"  ✓ Reduced LR by factor {self.collapse_guard_lr_factor}")

        return True

    def save_checkpoint(self, epoch, is_best=False, filename=None):
        """
        保存检查点
        
        Args:
            epoch: 当前epoch
            is_best: 是否为最佳模型
            filename: 保存文件名（可选）
        """
        if not self.save_checkpoints:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler is not None else None,
            'best_acc': self.best_acc,
            'history': self.history
        }
        
        if is_best:
            # 只保存best_model.pth
            best_path = os.path.join(self.result_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ 新的最佳准确率: {self.best_acc:.2f}% - 已保存到 best_model.pth")
        else:
            # 定期保存checkpoint
            if filename is None:
                filename = f'checkpoint_epoch_{epoch}.pth'
            save_path = os.path.join(self.result_dir, filename)
            torch.save(checkpoint, save_path)
            print(f"  ✓ 保存检查点: {filename}")
    
    def save_history(self):
        """保存训练历史到CSV"""
        csv_path = os.path.join(self.result_dir, 'train_history.csv')
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.history.keys())
            writer.writeheader()
            
            for i in range(len(self.history['epoch'])):
                row = {k: v[i] for k, v in self.history.items()}
                writer.writerow(row)
    
    def train(self):
        """
        完整训练流程
        
        Returns:
            best_acc: 最佳验证准确率
        """
        print(f"\n{'=' * 60}")
        print(f"开始训练 - 设备: {self.device}")
        print(f"{'=' * 60}")
        print(f"总epoch数: {self.epochs}")
        print(f"初始学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"批次大小: {self.train_loader.batch_size}")
        print(f"混合精度训练: {self.use_amp}")
        print(f"验证强制FP32: {self.val_force_fp32}")
        print(f"结果保存目录: {self.result_dir}")
        self._ensure_history_fields()

        if self.start_epoch > 1:
            print(f"Resuming training from epoch {self.start_epoch}")

        start_time = time.time()

        if self.nan_debug and not self._nan_debug_active:
            self._register_nan_hooks()

        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                # 为所有需要 epoch 信息的模块更新 epoch
                self._set_epoch_for_model(epoch)
                
                epoch_start = time.time()
                
                # 训练
                train_loss, train_acc = self.train_one_epoch(epoch)
                
                # 验证
                val_loss, val_acc = self.validate(epoch=epoch)
                
                # 更新学习率
                if self.scheduler is not None:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # 计算时间
                epoch_time = time.time() - epoch_start
                
                collapse_error = None
                try:
                    collapse_triggered = self._run_collapse_guard(epoch, val_acc)
                except RuntimeError as exc:
                    collapse_triggered = True
                    collapse_error = exc

                # 记录历史
                self.history['epoch'].append(epoch)
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['learning_rate'].append(current_lr)
                self.history['epoch_time'].append(epoch_time)
                self.history['train_valid_batches'].append(self._last_train_stats.get('valid_batches', 0))
                self.history['train_skipped_batches'].append(self._last_train_stats.get('skipped_batches', 0))
                self.history['val_valid_batches'].append(self._last_val_stats.get('valid_batches', 0))
                self.history['val_skipped_batches'].append(self._last_val_stats.get('skipped_batches', 0))
                self.history['nonfinite_train_batches'].append(self._last_train_stats.get('nonfinite_batches', 0))
                self.history['nonfinite_val_batches'].append(self._last_val_stats.get('nonfinite_batches', 0))
                self.history['smartpaf_phase'].append(self._current_smartpaf_phase(epoch))
                self.history['collapse_guard_triggered'].append(int(collapse_triggered))

                # 打印结果
                print(f"\nEpoch [{epoch}/{self.epochs}] - {epoch_time:.2f}s")
                if torch.isfinite(torch.tensor(train_loss)):
                    print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
                else:
                    print(f"  训练 - Loss: NaN/Inf, Acc: {train_acc:.2f}%")
                if torch.isfinite(torch.tensor(val_loss)):
                    print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
                else:
                    print(f"  验证 - Loss: NaN/Inf, Acc: {val_acc:.2f}%")
                print(f"  学习率: {current_lr:.6f}")
                
                # 如果第一个epoch就出现NaN，警告用户
                if epoch == 1 and not torch.isfinite(torch.tensor(val_loss)):
                    print(f"\n{'='*60}")
                    print("⚠ 警告: 第一个epoch验证loss为NaN！")
                    print("可能原因：")
                    print("  1. 模型初始化不当，输出值过大")
                    print("  2. 学习率过高导致梯度爆炸")
                    print("  3. 架构本身不稳定")
                    print("建议: 考虑降低学习率或检查模型架构")
                    print(f"{'='*60}")
                
                # 保存最佳模型
                is_new_best = val_acc > self.best_acc
                if is_new_best:
                    self.best_acc = val_acc
                    if self.save_checkpoints:
                        self.save_checkpoint(epoch, is_best=True)
                
                # 定期保存检查点（如果不是当前epoch的最佳模型）
                if self.save_checkpoints and self.save_freq and self.save_freq > 0:
                    if epoch % self.save_freq == 0 and not is_new_best:
                        self.save_checkpoint(epoch, is_best=False)
                
                # 保存历史
                self.save_history()
                if collapse_error is not None:
                    raise collapse_error
        finally:
            if self.smartpaf_alternate_training:
                self._restore_all_trainable()
            if self._nan_debug_active:
                self._remove_nan_hooks()
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print("训练完成!")
        print(f"{'=' * 60}")
        print(f"总训练时间: {total_time / 3600:.2f} 小时")
        print(f"最佳验证准确率: {self.best_acc:.2f}%")
        print(f"训练历史已保存到: {os.path.join(self.result_dir, 'train_history.csv')}")
        
        return self.best_acc
