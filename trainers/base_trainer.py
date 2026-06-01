"""
基础训练器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.optim.swa_utils import AveragedModel
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
        poly4_scale_mode='learned',
        poly4_dynamic_scale_momentum=0.99,
        poly4_dynamic_scale_eps=1e-6,
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
        smartpaf_at_start_epoch=None,
        smartpaf_at_initial_phase='weights',
        smartpaf_at_poly_scope='all',
        smartpaf_at_weight_scope='all',
        smartpaf_revalidate_rejected_phase=False,
        smartpaf_at_reject_nonimproving_poly=False,
        smartpaf_at_accept_min_delta=0.0,
        smartpaf_at_reject_lr_factor=1.0,
        smartpaf_at_reject_before_collapse_guard=False,
        smartpaf_at_skip_rejected_poly_group=False,
        smartpaf_at_stop_after_rejected_poly_groups=0,
        smartpaf_at_restore_phase_best=False,
        smartpaf_at_restore_phase_min_delta=0.0,
        smartpaf_at_phase_swa=False,
        smartpaf_at_dropout_on_overfit=False,
        smartpaf_at_dropout_gap=10.0,
        smartpaf_at_dropout_p=0.5,
        smartpaf_freeze_bn_during_poly_phase=True,
        smartpaf_ct_init=False,
        smartpaf_ct_batches=8,
        smartpaf_ct_max_samples=20000,
        smartpaf_ct_steps=300,
        smartpaf_ct_lr=0.01,
        smartpaf_ss_calibrate=False,
        smartpaf_ss_batches=8,
        smartpaf_ss_max_samples=20000,
        smartpaf_ss_percentile=1.0,
        smartpaf_ss_margin=1.0,
        smartpaf_ds_to_ss_after_training=False,
        smartpaf_ds_to_ss_use_best=True,
        bn_recalibrate_after_training=False,
        bn_recalibrate_batches=0,
        bn_recalibrate_use_best=True,
        swa_enabled=False,
        swa_start_epoch=None,
        swa_bn_update=True,
        swa_bn_batches=0,
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
            poly4_scale_mode: StablePoly4输入缩放模式 learned/dynamic/static
            poly4_dynamic_scale_momentum: dynamic scale running absmax 动量
            poly4_dynamic_scale_eps: dynamic/static scale 的最小 absmax
            nan_debug: 是否启用NaN定位钩子（默认关闭）
            val_force_fp32: 验证阶段强制使用FP32（禁用autocast）
            smartpaf_progressive: 是否逐层延迟启用 StablePoly4 多项式分支
            smartpaf_start_epoch: 第一个 StablePoly4 开始过渡的 epoch；默认使用 poly4 warmup 结束点
            smartpaf_group_epochs: 每个 StablePoly4 模块之间的启用间隔
            smartpaf_transition_epochs: 单个 StablePoly4 从 warmup 激活过渡到多项式的持续 epoch
            smartpaf_alternate_training: 是否按 epoch 交替训练普通权重和多项式系数
            smartpaf_at_cycle_epochs: AT 每个阶段持续的 epoch 数
            smartpaf_at_start_epoch: AT 起始 epoch；None 保持旧行为，auto 表示从第一个 PA 模块开始
            smartpaf_at_initial_phase: AT 起始阶段，weights 或 poly
            smartpaf_at_poly_scope: poly 阶段训练范围，all 或 active
            smartpaf_at_weight_scope: weights 阶段训练范围，all 或 active_related
            smartpaf_revalidate_rejected_phase: collapse guard 恢复 best 后是否重新验证并记录恢复模型
            smartpaf_at_reject_nonimproving_poly: 是否拒绝未提升 best 的 AT poly 阶段
            smartpaf_at_accept_min_delta: poly 阶段至少超过 best_acc 多少百分点才接受
            smartpaf_at_reject_lr_factor: 非 collapse poly reject 后的学习率倍率
            smartpaf_at_reject_before_collapse_guard: 是否在 collapse guard 前拒绝坏 poly 候选
            smartpaf_at_skip_rejected_poly_group: poly 组内某 epoch 被拒后是否跳过该组剩余 poly epoch
            smartpaf_at_stop_after_rejected_poly_groups: 连续拒绝多少个 poly 组后停止后续 poly AT；0 表示关闭
            smartpaf_at_restore_phase_best: AT phase group 结束时是否恢复组内 best/起点模型
            smartpaf_at_restore_phase_min_delta: phase group 需要超过起点多少百分点才接受
            smartpaf_at_phase_swa: phase group 内是否额外比较 SWA 平均模型候选
            smartpaf_at_dropout_on_overfit: 是否在训练/验证精度差过大时启用分类头 dropout
            smartpaf_at_dropout_gap: 触发 dropout 的训练/验证精度差，单位百分点
            smartpaf_at_dropout_p: 分类头输入 dropout 概率
            smartpaf_freeze_bn_during_poly_phase: AT 的多项式阶段是否冻结 BN 统计
            smartpaf_ct_init: 是否在训练前用采样激活拟合 StablePoly4 系数
            smartpaf_ct_batches: CT 采样 train batch 数
            smartpaf_ct_max_samples: 每个 StablePoly4 最多使用多少个标量激活样本
            smartpaf_ct_steps: 每个 StablePoly4 CT 优化步数
            smartpaf_ct_lr: CT Adam 学习率
            smartpaf_ss_calibrate: 是否在训练前校准 static scale
            smartpaf_ss_batches: SS 校准采样 train batch 数
            smartpaf_ss_max_samples: 每个 StablePoly4 最多使用多少个标量样本估计 scale
            smartpaf_ss_percentile: SS 使用的 abs 激活分位数；1.0 表示最大值
            smartpaf_ss_margin: SS absmax 安全余量倍率
            smartpaf_ds_to_ss_after_training: 训练后是否将 dynamic scale 固定成 static scale 并验证
            smartpaf_ds_to_ss_use_best: DS->SS 转换前是否加载 best_model.pth
            bn_recalibrate_after_training: 是否在训练结束后重算 BatchNorm running stats
            bn_recalibrate_batches: BN recalibration 使用的 train batch 数；0 表示全量
            bn_recalibrate_use_best: recalibration 前是否加载 best_model.pth
            swa_enabled: 是否启用 Stochastic Weight Averaging
            swa_start_epoch: SWA 开始平均的 epoch；None 表示最后 25% epoch
            swa_bn_update: SWA 评估前是否重算 BatchNorm running stats
            swa_bn_batches: SWA BN 更新使用的 train batch 数；0 表示全量
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
        self.poly4_scale_mode = str(poly4_scale_mode).strip().lower()
        self.poly4_dynamic_scale_momentum = float(poly4_dynamic_scale_momentum)
        self.poly4_dynamic_scale_eps = float(poly4_dynamic_scale_eps)
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
        self.smartpaf_at_start_epoch = smartpaf_at_start_epoch
        self.smartpaf_at_initial_phase = str(smartpaf_at_initial_phase).strip().lower()
        if self.smartpaf_at_initial_phase not in {'weights', 'poly'}:
            raise ValueError(
                f"Unsupported smartpaf_at_initial_phase: {smartpaf_at_initial_phase}"
            )
        self.smartpaf_at_poly_scope = str(smartpaf_at_poly_scope).strip().lower()
        if self.smartpaf_at_poly_scope not in {'all', 'active'}:
            raise ValueError(f"Unsupported smartpaf_at_poly_scope: {smartpaf_at_poly_scope}")
        self.smartpaf_at_weight_scope = str(smartpaf_at_weight_scope).strip().lower()
        if self.smartpaf_at_weight_scope not in {'all', 'active_related'}:
            raise ValueError(f"Unsupported smartpaf_at_weight_scope: {smartpaf_at_weight_scope}")
        self.smartpaf_revalidate_rejected_phase = bool(smartpaf_revalidate_rejected_phase)
        self.smartpaf_at_reject_nonimproving_poly = bool(smartpaf_at_reject_nonimproving_poly)
        self.smartpaf_at_accept_min_delta = float(smartpaf_at_accept_min_delta)
        self.smartpaf_at_reject_lr_factor = float(smartpaf_at_reject_lr_factor)
        self.smartpaf_at_reject_before_collapse_guard = bool(smartpaf_at_reject_before_collapse_guard)
        self.smartpaf_at_skip_rejected_poly_group = bool(smartpaf_at_skip_rejected_poly_group)
        self.smartpaf_at_stop_after_rejected_poly_groups = max(0, int(smartpaf_at_stop_after_rejected_poly_groups))
        self.smartpaf_at_restore_phase_best = bool(smartpaf_at_restore_phase_best)
        self.smartpaf_at_restore_phase_min_delta = float(smartpaf_at_restore_phase_min_delta)
        self.smartpaf_at_phase_swa = bool(smartpaf_at_phase_swa)
        self.smartpaf_at_dropout_on_overfit = bool(smartpaf_at_dropout_on_overfit)
        self.smartpaf_at_dropout_gap = float(smartpaf_at_dropout_gap)
        self.smartpaf_at_dropout_p = float(smartpaf_at_dropout_p)
        self.smartpaf_freeze_bn_during_poly_phase = bool(smartpaf_freeze_bn_during_poly_phase)
        self.smartpaf_ct_init = bool(smartpaf_ct_init)
        self.smartpaf_ct_batches = max(1, int(smartpaf_ct_batches))
        self.smartpaf_ct_max_samples = max(256, int(smartpaf_ct_max_samples))
        self.smartpaf_ct_steps = max(1, int(smartpaf_ct_steps))
        self.smartpaf_ct_lr = float(smartpaf_ct_lr)
        self.smartpaf_ss_calibrate = bool(smartpaf_ss_calibrate)
        self.smartpaf_ss_batches = max(1, int(smartpaf_ss_batches))
        self.smartpaf_ss_max_samples = max(256, int(smartpaf_ss_max_samples))
        self.smartpaf_ss_percentile = max(0.0, min(1.0, float(smartpaf_ss_percentile)))
        self.smartpaf_ss_margin = max(1e-6, float(smartpaf_ss_margin))
        self.smartpaf_ds_to_ss_after_training = bool(smartpaf_ds_to_ss_after_training)
        self.smartpaf_ds_to_ss_use_best = bool(smartpaf_ds_to_ss_use_best)
        self.bn_recalibrate_after_training = bool(bn_recalibrate_after_training)
        self.bn_recalibrate_batches = max(0, int(bn_recalibrate_batches))
        self.bn_recalibrate_use_best = bool(bn_recalibrate_use_best)
        self.swa_enabled = bool(swa_enabled)
        if swa_start_epoch is None:
            self.swa_start_epoch = max(1, int(self.epochs * 0.75))
        else:
            self.swa_start_epoch = max(1, int(swa_start_epoch))
        self.swa_bn_update = bool(swa_bn_update)
        self.swa_bn_batches = max(0, int(swa_bn_batches))
        self.collapse_guard_enabled = bool(collapse_guard_enabled)
        self.collapse_guard_drop = float(collapse_guard_drop)
        self.collapse_guard_patience = max(1, int(collapse_guard_patience))
        self.collapse_guard_action = str(collapse_guard_action).strip().lower()
        self.collapse_guard_lr_factor = float(collapse_guard_lr_factor)
        self._collapse_guard_hits = 0
        self._last_collapse_guard_restored = False
        self._last_train_stats = {}
        self._last_val_stats = {}
        self._smartpaf_poly_modules = []
        self._smartpaf_poly_param_ids = set()
        self._smartpaf_last_phase = None
        self._smartpaf_first_poly_epoch = None
        self._smartpaf_at_start_epoch = None
        self._smartpaf_poly_module_param_ids = []
        self._smartpaf_related_weight_module_param_ids = []
        self._smartpaf_skipped_poly_phase_idx = None
        self._smartpaf_last_rejected_poly_phase_idx = None
        self._smartpaf_consecutive_rejected_poly_groups = 0
        self._smartpaf_poly_stopped_after_rejections = False
        self._smartpaf_restore_phase_idx = None
        self._smartpaf_restore_phase_label = None
        self._smartpaf_restore_phase_start_acc = None
        self._smartpaf_restore_phase_start_state = None
        self._smartpaf_restore_phase_best_acc = None
        self._smartpaf_restore_phase_best_epoch = None
        self._smartpaf_restore_phase_best_state = None
        self._smartpaf_restore_phase_last_restored_acc = None
        self._smartpaf_restore_phase_swa_model = None
        self._smartpaf_restore_phase_swa_updates = 0
        self._smartpaf_restore_phase_swa_acc = None
        self._swa_model = None
        self._swa_updates = 0
        self._nan_debug_running = False
        self._nan_hooks = []
        self._nan_triggered = False
        self._nan_debug_active = False
        self._smartpaf_overfit_dropout_active = False
        self._smartpaf_overfit_dropout_hooks = []
        self._smartpaf_overfit_dropout_targets = []

        # 创建结果目录
        os.makedirs(result_dir, exist_ok=True)

        # 自动调整StablePoly4的warmup_epochs
        self._adjust_poly4_warmup()
        self._configure_poly4_modules()
        self._configure_smartpaf_modules()
        self._configure_smartpaf_overfit_dropout()

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
            'smartpaf_overfit_dropout_active': [],
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
            'smartpaf_overfit_dropout_active',
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
        if self._smartpaf_at_start_epoch is not None and float(epoch) < self._smartpaf_at_start_epoch:
            return 'all'
        return 'poly' if self._is_smartpaf_poly_phase(epoch) else 'weights'

    def _last_linear_in_module(self, module):
        if isinstance(module, nn.Linear):
            return module
        linear = None
        for child in module.modules():
            if isinstance(child, nn.Linear):
                linear = child
        return linear

    def _find_smartpaf_dropout_targets(self):
        targets = []
        for attr_name in ('fc', 'classifier', 'head'):
            if not hasattr(self.model, attr_name):
                continue
            target = self._last_linear_in_module(getattr(self.model, attr_name))
            if target is not None:
                targets.append((attr_name, target))
                return targets

        last_name = None
        last_linear = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                last_name = name
                last_linear = module
        if last_linear is not None:
            targets.append((last_name, last_linear))
        return targets

    def _configure_smartpaf_overfit_dropout(self):
        if not self.smartpaf_at_dropout_on_overfit:
            return

        if not (0.0 < self.smartpaf_at_dropout_p < 1.0):
            raise ValueError(
                f"smartpaf_at_dropout_p must be between 0 and 1, got {self.smartpaf_at_dropout_p}"
            )

        targets = self._find_smartpaf_dropout_targets()
        if not targets:
            print("⚠ SmartPAF overfit dropout enabled but no Linear classifier head was found")
            return

        def make_hook(name):
            def _hook(module, inputs):
                if (
                    not self._smartpaf_overfit_dropout_active
                    or not self.model.training
                    or not inputs
                    or not torch.is_tensor(inputs[0])
                ):
                    return None
                dropped = F.dropout(inputs[0], p=self.smartpaf_at_dropout_p, training=True)
                return (dropped, *inputs[1:])
            return _hook

        for name, module in targets:
            self._smartpaf_overfit_dropout_hooks.append(module.register_forward_pre_hook(make_hook(name)))
            self._smartpaf_overfit_dropout_targets.append(name)

        target_text = ", ".join(self._smartpaf_overfit_dropout_targets)
        print(
            "✓ SmartPAF overfit dropout: "
            f"targets={target_text}, gap>={self.smartpaf_at_dropout_gap:g}pp, p={self.smartpaf_at_dropout_p:g}"
        )

    def _remove_smartpaf_overfit_dropout_hooks(self):
        for handle in self._smartpaf_overfit_dropout_hooks:
            handle.remove()
        self._smartpaf_overfit_dropout_hooks = []

    def _update_smartpaf_overfit_dropout(self, epoch, train_acc, val_acc):
        if not self.smartpaf_at_dropout_on_overfit or not self._smartpaf_overfit_dropout_targets:
            return
        if not self.smartpaf_alternate_training:
            return

        gap = float(train_acc) - float(val_acc)
        should_enable = gap >= self.smartpaf_at_dropout_gap
        if should_enable != self._smartpaf_overfit_dropout_active:
            state = "enabled" if should_enable else "disabled"
            print(
                f"  - SmartPAF overfit dropout {state}: "
                f"train-val gap={gap:.2f}pp, threshold={self.smartpaf_at_dropout_gap:.2f}pp"
            )
        self._smartpaf_overfit_dropout_active = should_enable

    def _resolve_smartpaf_at_start_epoch(self):
        if self.smartpaf_at_start_epoch is None:
            return None

        if isinstance(self.smartpaf_at_start_epoch, str):
            key = self.smartpaf_at_start_epoch.strip().lower()
            if key in {'', 'none', 'false'}:
                return None
            if key in {'auto', 'poly_start', 'pa_start'}:
                if self._smartpaf_first_poly_epoch is not None:
                    return float(self._smartpaf_first_poly_epoch)
                return float(getattr(self, "poly4_warmup_epochs", 0))
            return float(key)

        return float(self.smartpaf_at_start_epoch)

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
            if hasattr(module, 'set_scale_mode') and callable(module.set_scale_mode):
                module.set_scale_mode(
                    mode=self.poly4_scale_mode,
                    momentum=self.poly4_dynamic_scale_momentum,
                    eps=self.poly4_dynamic_scale_eps,
                )
            if hasattr(module, 'set_warmup_epochs') and callable(module.set_warmup_epochs):
                poly4_count += 1

        if poly4_count > 0:
            print("✓ StablePoly4正则/调度配置:")
            print(f"  - scale_mode: {self.poly4_scale_mode}")
            if self.poly4_range_lambda > 0:
                print(f"  - range_r: {self.poly4_range_r}, lambda_range: {self.poly4_range_lambda}")
            if self.poly4_deriv_lambda > 0:
                print(f"  - deriv_L: {self.poly4_deriv_L}, lambda_deriv: {self.poly4_deriv_lambda}")

    def _configure_smartpaf_modules(self):
        """配置 SmartPAF-lite 的逐层多项式启用与 AT 参数集合。"""
        self._smartpaf_poly_modules = []
        self._smartpaf_poly_param_ids = set()
        self._smartpaf_poly_module_param_ids = []
        self._smartpaf_related_weight_module_param_ids = []
        named_params = list(self.model.named_parameters())

        for name, module in self.model.named_modules():
            if hasattr(module, "set_poly_schedule") and callable(module.set_poly_schedule):
                self._smartpaf_poly_modules.append((name, module))
                module_param_ids = set()
                for param in module.parameters(recurse=True):
                    param_id = id(param)
                    self._smartpaf_poly_param_ids.add(param_id)
                    module_param_ids.add(param_id)
                self._smartpaf_poly_module_param_ids.append((module, module_param_ids))
                parent_prefix = name.rsplit('.', 1)[0] + '.' if '.' in name else ''
                related_param_ids = {
                    id(param)
                    for param_name, param in named_params
                    if parent_prefix and param_name.startswith(parent_prefix) and id(param) not in module_param_ids
                }
                self._smartpaf_related_weight_module_param_ids.append((module, related_param_ids))

        if not self._smartpaf_poly_modules:
            return

        if self.smartpaf_progressive:
            start_epoch = self.smartpaf_start_epoch
            if start_epoch is None:
                start_epoch = getattr(self, "poly4_warmup_epochs", 0)
            start_epoch = float(start_epoch)
            self._smartpaf_first_poly_epoch = start_epoch

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
        else:
            self._smartpaf_first_poly_epoch = float(getattr(self, "poly4_warmup_epochs", 0))

        if self.smartpaf_alternate_training:
            self._smartpaf_at_start_epoch = self._resolve_smartpaf_at_start_epoch()
            print("✓ SmartPAF-lite AT配置:")
            print(f"  - 每阶段epoch: {self.smartpaf_at_cycle_epochs}")
            if self._smartpaf_at_start_epoch is not None:
                print(f"  - 起始epoch: {self._smartpaf_at_start_epoch:g}")
            print(f"  - 多项式训练范围: {self.smartpaf_at_poly_scope}")
            print(f"  - 权重训练范围: {self.smartpaf_at_weight_scope}")
            if self.smartpaf_at_initial_phase == 'poly':
                print("  - 阶段顺序: 多项式系数 -> 普通权重 -> ...")
            else:
                print("  - 阶段顺序: 普通权重 -> 多项式系数 -> ...")

    def _ct_poly_eval(self, module, x):
        scale_mode = str(getattr(module, "scale_mode", "learned"))
        if scale_mode == "static" and hasattr(module, "static_absmax"):
            absmax = module.static_absmax.to(device=x.device, dtype=torch.float32)
            absmax = torch.clamp(absmax, min=getattr(module, "dynamic_scale_eps", 1e-6))
            x_poly = x / absmax.to(dtype=x.dtype)
        elif scale_mode == "dynamic":
            absmax = x.detach().abs().amax().float()
            absmax = torch.clamp(absmax, min=getattr(module, "dynamic_scale_eps", 1e-6))
            x_poly = x / absmax.to(device=x.device, dtype=x.dtype)
        else:
            log_in_scale = torch.clamp(module.log_in_scale, min=-6.0, max=2.0)
            x_poly = x * torch.exp(log_in_scale)
        a = torch.clamp(module.a, min=-0.01, max=0.01)
        b = torch.clamp(module.b, min=-0.1, max=0.1)
        c = torch.clamp(module.c, min=-0.5, max=0.5)
        d = torch.clamp(module.d, min=-5.0, max=5.0)
        e = torch.clamp(module.e, min=-5.0, max=5.0)
        poly = ((((a * x_poly + b) * x_poly + c) * x_poly + d) * x_poly + e)
        return poly * float(getattr(module, "output_scale", 1.0))

    def _run_smartpaf_ct_init(self):
        """Fit StablePoly4 coefficients to their warmup activation on sampled inputs."""
        if not self.smartpaf_ct_init or not self._smartpaf_poly_modules:
            return

        print("✓ SmartPAF-lite CT初始化:")
        print(
            f"  - batches={self.smartpaf_ct_batches}, max_samples={self.smartpaf_ct_max_samples}, "
            f"steps={self.smartpaf_ct_steps}, lr={self.smartpaf_ct_lr}"
        )

        was_training = self.model.training
        samples = {module: [] for _, module in self._smartpaf_poly_modules}
        handles = []

        def make_hook(module):
            def hook(_module, inputs):
                if not inputs:
                    return
                remaining = self.smartpaf_ct_max_samples - sum(t.numel() for t in samples[module])
                if remaining <= 0:
                    return
                values = inputs[0].detach().float().flatten()
                if values.numel() > remaining:
                    idx = torch.linspace(0, values.numel() - 1, remaining, device=values.device).long()
                    values = values.index_select(0, idx)
                samples[module].append(values.cpu())
            return hook

        for _, module in self._smartpaf_poly_modules:
            handles.append(module.register_forward_pre_hook(make_hook(module)))

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(self.train_loader):
                if batch_idx >= self.smartpaf_ct_batches:
                    break
                images = images.to(self.device, non_blocking=True)
                self.model(images)

        for handle in handles:
            handle.remove()

        for name, module in self._smartpaf_poly_modules:
            if not samples[module]:
                print(f"  - {name}: skipped, no activation samples")
                continue

            x = torch.cat(samples[module], dim=0)
            if x.numel() > self.smartpaf_ct_max_samples:
                x = x[:self.smartpaf_ct_max_samples]
            x = x.to(self.device)
            with torch.no_grad():
                target = module.warmup_act(x).detach()

            params = [module.a, module.b, module.c, module.d, module.e, module.log_in_scale]
            old_requires_grad = [param.requires_grad for param in params]
            for param in params:
                param.requires_grad = True
            optimizer = optim.Adam(params, lr=self.smartpaf_ct_lr)

            last_loss = None
            for _ in range(self.smartpaf_ct_steps):
                optimizer.zero_grad(set_to_none=True)
                pred = self._ct_poly_eval(module, x)
                loss = F.mse_loss(pred, target)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())

            for param, requires_grad in zip(params, old_requires_grad):
                param.requires_grad = requires_grad
            print(f"  - {name}: samples={x.numel()}, mse={last_loss:.6g}")

        self.model.train(was_training)

    def _run_smartpaf_ss_calibration(self):
        """Calibrate StablePoly4 static_absmax from sampled pre-activation inputs."""
        if not self.smartpaf_ss_calibrate or not self._smartpaf_poly_modules:
            return

        static_modules = [
            (name, module)
            for name, module in self._smartpaf_poly_modules
            if str(getattr(module, "scale_mode", "learned")) == "static"
            and hasattr(module, "set_scale_mode")
        ]
        if not static_modules:
            print("SmartPAF-lite SS校准: skipped, no StablePoly4 modules in static scale mode")
            return

        print("✓ SmartPAF-lite SS校准:")
        print(
            f"  - batches={self.smartpaf_ss_batches}, max_samples={self.smartpaf_ss_max_samples}, "
            f"percentile={self.smartpaf_ss_percentile:g}, margin={self.smartpaf_ss_margin:g}"
        )

        was_training = self.model.training
        absmax = {module: 0.0 for _, module in static_modules}
        samples = {module: [] for _, module in static_modules}
        handles = []

        def make_hook(module):
            def hook(_module, inputs):
                if not inputs:
                    return
                values = inputs[0].detach().float().abs().flatten()
                if values.numel() == 0:
                    return
                batch_max = float(values.max().cpu())
                absmax[module] = max(absmax[module], batch_max)
                if self.smartpaf_ss_percentile < 1.0:
                    current = sum(t.numel() for t in samples[module])
                    remaining = self.smartpaf_ss_max_samples - current
                    if remaining <= 0:
                        return
                    if values.numel() > remaining:
                        idx = torch.linspace(0, values.numel() - 1, remaining, device=values.device).long()
                        values = values.index_select(0, idx)
                    samples[module].append(values.cpu())
            return hook

        for _, module in static_modules:
            handles.append(module.register_forward_pre_hook(make_hook(module)))

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(self.train_loader):
                if batch_idx >= self.smartpaf_ss_batches:
                    break
                images = images.to(self.device, non_blocking=True)
                self.model(images)

        for handle in handles:
            handle.remove()

        for name, module in static_modules:
            if self.smartpaf_ss_percentile < 1.0 and samples[module]:
                values = torch.cat(samples[module], dim=0)
                if values.numel() > self.smartpaf_ss_max_samples:
                    values = values[:self.smartpaf_ss_max_samples]
                q = torch.quantile(values.float(), self.smartpaf_ss_percentile).item()
                calibrated = q
            else:
                calibrated = absmax[module]
            calibrated = max(calibrated * self.smartpaf_ss_margin, getattr(module, "dynamic_scale_eps", 1e-6))
            module.set_scale_mode(static_absmax=calibrated)
            print(f"  - {name}: static_absmax={calibrated:.6g}, in_scale≈{1.0 / calibrated:.6g}")

        self.model.train(was_training)

    def _run_smartpaf_ds_to_ss_evaluation(self):
        """Convert dynamic StablePoly4 scales to deployable static scales and validate."""
        if not self.smartpaf_ds_to_ss_after_training or not self._smartpaf_poly_modules:
            return None

        dynamic_modules = [
            (name, module)
            for name, module in self._smartpaf_poly_modules
            if str(getattr(module, "scale_mode", "learned")) == "dynamic"
            and hasattr(module, "running_absmax")
            and hasattr(module, "set_scale_mode")
        ]
        if not dynamic_modules:
            print("SmartPAF-lite DS->SS: skipped, no StablePoly4 modules in dynamic scale mode")
            return None

        print("✓ SmartPAF-lite DS->SS conversion:")
        print(f"  - use_best={self.smartpaf_ds_to_ss_use_best}")

        if self.smartpaf_ds_to_ss_use_best:
            best_path = os.path.join(self.result_dir, 'best_model.pth')
            if os.path.exists(best_path):
                checkpoint = torch.load(best_path, map_location=self.device)
                model_state = checkpoint.get('model_state_dict')
                if model_state is not None:
                    self.model.load_state_dict(model_state, strict=self.resume_strict)
                    print(f"  - loaded best model from {best_path}")
            else:
                print(f"  - best model not found at {best_path}; using current model")

        converted = []
        for name, module in dynamic_modules:
            absmax = float(module.running_absmax.detach().float().cpu().item())
            absmax = max(absmax, getattr(module, "dynamic_scale_eps", 1e-6))
            module.set_scale_mode(mode='static', static_absmax=absmax)
            converted.append((name, absmax))
            print(f"  - {name}: static_absmax={absmax:.6g}, in_scale≈{1.0 / absmax:.6g}")

        start_time = time.time()
        val_loss, val_acc = self.validate(epoch='ds_to_ss')
        eval_time = time.time() - start_time
        current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0.0
        self._last_train_stats = {
            'valid_batches': 0,
            'skipped_batches': 0,
            'nonfinite_batches': 0,
        }
        self._append_history_row(
            epoch=self._next_history_epoch(),
            train_loss=0.0,
            train_acc=0.0,
            val_loss=val_loss,
            val_acc=val_acc,
            learning_rate=current_lr,
            epoch_time=eval_time,
            smartpaf_phase='ds_to_ss',
            collapse_guard_triggered=0,
        )

        if self.save_checkpoints:
            is_new_best = val_acc > self.best_acc
            if is_new_best:
                self.best_acc = val_acc
                self.save_checkpoint(self.history['epoch'][-1], is_best=True)
            self.save_checkpoint(self.history['epoch'][-1], is_best=False, filename='ds_to_ss_model.pth')

        self.save_history()
        print(f"  - DS->SS validation: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        return {'val_loss': val_loss, 'val_acc': val_acc, 'converted': converted}

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
        if self._smartpaf_poly_stopped_after_rejections:
            return False
        phase_idx = self._smartpaf_phase_index(epoch)
        if phase_idx is None:
            return False
        poly_on_even_phase = self.smartpaf_at_initial_phase == 'poly'
        is_poly = (phase_idx % 2 == 0) if poly_on_even_phase else (phase_idx % 2 == 1)
        if (
            is_poly
            and self.smartpaf_at_skip_rejected_poly_group
            and self._smartpaf_skipped_poly_phase_idx == phase_idx
        ):
            return False
        return is_poly

    def _smartpaf_phase_index(self, epoch):
        if not self.smartpaf_alternate_training:
            return None
        if self._smartpaf_at_start_epoch is not None:
            if float(epoch) < self._smartpaf_at_start_epoch:
                return None
            epoch_offset = max(0.0, float(epoch) - self._smartpaf_at_start_epoch)
        else:
            epoch_offset = max(0.0, float(max(1, int(epoch)) - 1))
        return int(epoch_offset // self.smartpaf_at_cycle_epochs)

    def _mark_rejected_poly_group(self, epoch):
        phase_idx = self._smartpaf_phase_index(epoch)
        if phase_idx is None:
            return

        if self.smartpaf_at_skip_rejected_poly_group:
            self._smartpaf_skipped_poly_phase_idx = phase_idx
        if self.smartpaf_at_skip_rejected_poly_group and self.smartpaf_at_cycle_epochs > 1:
            print(f"  - Skipping remaining poly epochs in AT phase group {phase_idx}")

        if self._smartpaf_last_rejected_poly_phase_idx != phase_idx:
            self._smartpaf_last_rejected_poly_phase_idx = phase_idx
            self._smartpaf_consecutive_rejected_poly_groups += 1
        if (
            self.smartpaf_at_stop_after_rejected_poly_groups > 0
            and self._smartpaf_consecutive_rejected_poly_groups
            >= self.smartpaf_at_stop_after_rejected_poly_groups
        ):
            self._smartpaf_poly_stopped_after_rejections = True
            print(
                "  - Stopping future AT poly phases after "
                f"{self._smartpaf_consecutive_rejected_poly_groups} rejected group(s)"
            )

    def _mark_accepted_poly_group(self, epoch):
        phase_idx = self._smartpaf_phase_index(epoch)
        if phase_idx is None:
            return
        if self._smartpaf_last_rejected_poly_phase_idx != phase_idx:
            self._smartpaf_consecutive_rejected_poly_groups = 0

    def _clone_model_state_cpu(self):
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }

    def _clone_state_from_model_cpu(self, model):
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

    def _restore_model_state(self, state, reason):
        if state is None:
            return False
        self.model.load_state_dict(state, strict=self.resume_strict)
        print(f"  ✓ Restored model state ({reason})")
        return True

    def _last_recorded_val_acc(self):
        values = self.history.get('val_acc') or []
        if not values:
            return self.best_acc
        try:
            return float(values[-1])
        except (TypeError, ValueError):
            return self.best_acc

    def _begin_smartpaf_phase_restore_group(self, epoch):
        phase_idx = self._smartpaf_phase_index(epoch)
        if phase_idx is None:
            return
        start_acc = self._smartpaf_restore_phase_last_restored_acc
        if start_acc is None:
            start_acc = self._last_recorded_val_acc()
        self._smartpaf_restore_phase_last_restored_acc = None
        self._smartpaf_restore_phase_idx = phase_idx
        self._smartpaf_restore_phase_label = self._current_smartpaf_phase(epoch)
        self._smartpaf_restore_phase_start_acc = start_acc
        self._smartpaf_restore_phase_start_state = self._clone_model_state_cpu()
        self._smartpaf_restore_phase_best_acc = self._smartpaf_restore_phase_start_acc
        self._smartpaf_restore_phase_best_epoch = None
        self._smartpaf_restore_phase_best_state = self._smartpaf_restore_phase_start_state
        self._smartpaf_restore_phase_swa_model = None
        self._smartpaf_restore_phase_swa_updates = 0
        self._smartpaf_restore_phase_swa_acc = None
        print(
            "SmartPAF-lite AT phase restore group started: "
            f"idx={phase_idx}, phase={self._smartpaf_restore_phase_label}, "
            f"start_acc={self._smartpaf_restore_phase_start_acc:.2f}%"
        )

    def _finalize_smartpaf_phase_restore_group(self, final=False):
        if self._smartpaf_restore_phase_idx is None:
            return None

        start_acc = self._smartpaf_restore_phase_start_acc
        best_acc = self._smartpaf_restore_phase_best_acc
        best_epoch = self._smartpaf_restore_phase_best_epoch
        best_source = 'best'
        if self._smartpaf_restore_phase_swa_model is not None:
            swa_model = self._smartpaf_restore_phase_swa_model.module.to(self.device)
            swa_loss, swa_acc = self._validate_with_model(
                swa_model,
                epoch=f'phase_swa_{self._smartpaf_restore_phase_idx}',
            )
            self._smartpaf_restore_phase_swa_acc = swa_acc
            print(
                "  - AT phase SWA candidate: "
                f"idx={self._smartpaf_restore_phase_idx}, "
                f"updates={self._smartpaf_restore_phase_swa_updates}, "
                f"loss={swa_loss:.4f}, acc={swa_acc:.2f}%"
            )
            if swa_acc > best_acc:
                best_acc = float(swa_acc)
                best_epoch = 'swa'
                best_source = 'swa'
                self._smartpaf_restore_phase_best_state = self._clone_state_from_model_cpu(swa_model)
        accepted = best_acc > start_acc + self.smartpaf_at_restore_phase_min_delta
        if accepted:
            restored_state = self._smartpaf_restore_phase_best_state
            restored_acc = best_acc
            reason = (
                f"AT phase {self._smartpaf_restore_phase_idx} best"
                f"{f' epoch {best_epoch}' if best_epoch is not None else ''}"
            )
        else:
            restored_state = self._smartpaf_restore_phase_start_state
            restored_acc = start_acc
            reason = f"AT phase {self._smartpaf_restore_phase_idx} start"

        self._restore_model_state(restored_state, reason)
        print(
            "  - AT phase group finalized: "
            f"idx={self._smartpaf_restore_phase_idx}, phase={self._smartpaf_restore_phase_label}, "
            f"start={start_acc:.2f}%, best={best_acc:.2f}%, "
            f"accepted={int(accepted)}"
        )

        result = {
            'phase_idx': self._smartpaf_restore_phase_idx,
            'phase': self._smartpaf_restore_phase_label,
            'start_acc': start_acc,
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'best_source': best_source,
            'swa_acc': self._smartpaf_restore_phase_swa_acc,
            'swa_updates': self._smartpaf_restore_phase_swa_updates,
            'restored_acc': restored_acc,
            'accepted': accepted,
            'final': final,
        }
        self._smartpaf_restore_phase_last_restored_acc = restored_acc
        self._smartpaf_restore_phase_idx = None
        self._smartpaf_restore_phase_label = None
        self._smartpaf_restore_phase_start_acc = None
        self._smartpaf_restore_phase_start_state = None
        self._smartpaf_restore_phase_best_acc = None
        self._smartpaf_restore_phase_best_epoch = None
        self._smartpaf_restore_phase_best_state = None
        self._smartpaf_restore_phase_swa_model = None
        self._smartpaf_restore_phase_swa_updates = 0
        self._smartpaf_restore_phase_swa_acc = None
        return result

    def _prepare_smartpaf_phase_restore_group(self, epoch):
        if not self.smartpaf_at_restore_phase_best or not self.smartpaf_alternate_training:
            return None
        phase_idx = self._smartpaf_phase_index(epoch)
        if phase_idx is None:
            return None
        finalized = None
        if self._smartpaf_restore_phase_idx is None:
            self._begin_smartpaf_phase_restore_group(epoch)
        elif phase_idx != self._smartpaf_restore_phase_idx:
            finalized = self._finalize_smartpaf_phase_restore_group()
            self._begin_smartpaf_phase_restore_group(epoch)
        return finalized

    def _update_smartpaf_phase_restore_group(self, epoch, val_acc):
        if (
            not self.smartpaf_at_restore_phase_best
            or self._smartpaf_restore_phase_idx is None
        ):
            return
        phase_idx = self._smartpaf_phase_index(epoch)
        if phase_idx != self._smartpaf_restore_phase_idx:
            return
        if val_acc > self._smartpaf_restore_phase_best_acc:
            self._smartpaf_restore_phase_best_acc = float(val_acc)
            self._smartpaf_restore_phase_best_epoch = epoch
            self._smartpaf_restore_phase_best_state = self._clone_model_state_cpu()
            print(
                "  - AT phase group new best: "
                f"idx={phase_idx}, epoch={epoch}, val_acc={val_acc:.2f}%"
            )

        if self.smartpaf_at_phase_swa:
            if self._smartpaf_restore_phase_swa_model is None:
                self._smartpaf_restore_phase_swa_model = AveragedModel(self.model).to(self.device)
                self._smartpaf_restore_phase_swa_model.update_parameters(self.model)
                self._smartpaf_restore_phase_swa_updates = 1
            else:
                self._smartpaf_restore_phase_swa_model.update_parameters(self.model)
                self._smartpaf_restore_phase_swa_updates += 1
            print(
                "  - AT phase SWA update: "
                f"idx={phase_idx}, updates={self._smartpaf_restore_phase_swa_updates}"
            )

    def _active_smartpaf_poly_param_ids(self, epoch):
        if self.smartpaf_at_poly_scope == 'all':
            return self._smartpaf_poly_param_ids

        active_ids = set()
        epoch_value = float(epoch)
        for module, param_ids in self._smartpaf_poly_module_param_ids:
            start_epoch = float(getattr(module, "poly_start_epoch", 0.0))
            if epoch_value >= start_epoch:
                active_ids.update(param_ids)
        return active_ids if active_ids else self._smartpaf_poly_param_ids

    def _active_smartpaf_related_weight_param_ids(self, epoch):
        if self.smartpaf_at_weight_scope == 'all':
            return None

        active_ids = set()
        epoch_value = float(epoch)
        for module, param_ids in self._smartpaf_related_weight_module_param_ids:
            start_epoch = float(getattr(module, "poly_start_epoch", 0.0))
            if epoch_value >= start_epoch:
                active_ids.update(param_ids)
        return active_ids if active_ids else None

    def _set_batchnorm_eval(self):
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def _apply_smartpaf_training_mode(self, epoch):
        """按 AT 阶段切换参数 requires_grad。"""
        if not self.smartpaf_alternate_training or not self._smartpaf_poly_param_ids:
            return

        if self._smartpaf_at_start_epoch is not None and float(epoch) < self._smartpaf_at_start_epoch:
            if self._smartpaf_last_phase != "all":
                print("SmartPAF-lite AT阶段: all")
                self._smartpaf_last_phase = "all"
                self._restore_all_trainable()
                self._zero_all_gradients()
            return

        train_poly = self._is_smartpaf_poly_phase(epoch)
        phase = "poly" if train_poly else "weights"
        if phase != self._smartpaf_last_phase:
            print(f"SmartPAF-lite AT阶段: {phase}")
            self._smartpaf_last_phase = phase
            self._zero_all_gradients()

        trainable_poly_ids = self._active_smartpaf_poly_param_ids(epoch) if train_poly else set()
        trainable_weight_ids = None if train_poly else self._active_smartpaf_related_weight_param_ids(epoch)
        for param in self.model.parameters():
            is_poly_param = id(param) in self._smartpaf_poly_param_ids
            if train_poly:
                param.requires_grad = id(param) in trainable_poly_ids
            elif trainable_weight_ids is not None:
                param.requires_grad = (not is_poly_param) and id(param) in trainable_weight_ids
            else:
                param.requires_grad = not is_poly_param

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

    def _set_epoch_progress_for_model(self, epoch, step_idx, steps_per_epoch, model=None):
        """
        递归地为模型中所有需要 epoch 进度信息的模块设置细粒度进度

        Args:
            epoch: 当前 epoch 编号（1-based）
            step_idx: 当前 batch 索引（0-based）
            steps_per_epoch: 每个 epoch 的 batch 数
        """
        target_model = model if model is not None else self.model
        for module in target_model.modules():
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
                scale_mode = str(getattr(module, "scale_mode", "learned"))
                if scale_mode == "learned":
                    in_scale = float(torch.exp(torch.tensor(log_in_clamped)))
                else:
                    if scale_mode == "static" and hasattr(module, "static_absmax"):
                        absmax = module.static_absmax.detach().float().cpu().item()
                    elif hasattr(module, "running_absmax"):
                        absmax = module.running_absmax.detach().float().cpu().item()
                    else:
                        absmax = 1.0
                    in_scale = 1.0 / max(absmax, getattr(module, "dynamic_scale_eps", 1e-6))
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
                f"scale_mode={scale_mode} log_in_scale={log_in:.4g} in_scale≈{in_scale:.4g} "
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

    def _append_history_row(
        self,
        epoch,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        learning_rate,
        epoch_time,
        smartpaf_phase,
        collapse_guard_triggered=0,
    ):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rate'].append(learning_rate)
        self.history['epoch_time'].append(epoch_time)
        self.history['train_valid_batches'].append(self._last_train_stats.get('valid_batches', 0))
        self.history['train_skipped_batches'].append(self._last_train_stats.get('skipped_batches', 0))
        self.history['val_valid_batches'].append(self._last_val_stats.get('valid_batches', 0))
        self.history['val_skipped_batches'].append(self._last_val_stats.get('skipped_batches', 0))
        self.history['nonfinite_train_batches'].append(self._last_train_stats.get('nonfinite_batches', 0))
        self.history['nonfinite_val_batches'].append(self._last_val_stats.get('nonfinite_batches', 0))
        self.history['smartpaf_phase'].append(smartpaf_phase)
        self.history['smartpaf_overfit_dropout_active'].append(int(self._smartpaf_overfit_dropout_active))
        self.history['collapse_guard_triggered'].append(int(collapse_guard_triggered))

    def _next_history_epoch(self):
        numeric_epochs = []
        for epoch in self.history.get('epoch', []):
            try:
                numeric_epochs.append(int(epoch))
            except (TypeError, ValueError):
                continue
        return (max(numeric_epochs) + 1) if numeric_epochs else 1

    def _has_batchnorm_modules(self, model=None):
        target_model = model if model is not None else self.model
        return any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in target_model.modules())

    def _recalibrate_model_batchnorm(self, model, batches, epoch_for_progress, label):
        if not self._has_batchnorm_modules(model):
            print(f"{label}: skipped, no BatchNorm modules")
            return 0, 0.0

        was_training = model.training
        bn_modules = [
            module for module in model.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        original_momenta = {module: module.momentum for module in bn_modules}

        for module in bn_modules:
            module.reset_running_stats()
            module.momentum = None

        model.train()
        start_time = time.time()
        valid_batches = 0
        limit = max(0, int(batches))
        steps_per_epoch = min(len(self.train_loader), limit) if limit else len(self.train_loader)

        with torch.no_grad():
            pbar = self._progress_bar(self.train_loader, desc=label)
            for batch_idx, (images, _) in enumerate(pbar):
                if limit and batch_idx >= limit:
                    break
                images = images.to(self.device, non_blocking=True)
                if self.val_force_fp32 and images.dtype != torch.float32:
                    images = images.float()
                self._set_epoch_progress_for_model(epoch_for_progress, batch_idx, steps_per_epoch, model=model)
                model(images)
                valid_batches += 1
                pbar.set_postfix({'batches': valid_batches})

        for module, momentum in original_momenta.items():
            module.momentum = momentum
        model.train(was_training)

        return valid_batches, time.time() - start_time

    def _validate_with_model(self, model, epoch):
        original_model = self.model
        self.model = model
        try:
            return self.validate(epoch=epoch)
        finally:
            self.model = original_model

    def _run_bn_recalibration(self):
        if not self.bn_recalibrate_after_training:
            return None
        if not self._has_batchnorm_modules(self.model):
            print("BN recalibration: skipped, no BatchNorm modules")
            return None

        print("✓ BN recalibration:")
        print(f"  - use_best={self.bn_recalibrate_use_best}, batches={self.bn_recalibrate_batches or 'all'}")

        if self.bn_recalibrate_use_best:
            best_path = os.path.join(self.result_dir, 'best_model.pth')
            if os.path.exists(best_path):
                checkpoint = torch.load(best_path, map_location=self.device)
                model_state = checkpoint.get('model_state_dict')
                if model_state is not None:
                    self.model.load_state_dict(model_state, strict=self.resume_strict)
                    print(f"  - loaded best model from {best_path}")
            else:
                print(f"  - best model not found at {best_path}; using current model")

        valid_batches, recal_time = self._recalibrate_model_batchnorm(
            self.model,
            batches=self.bn_recalibrate_batches,
            epoch_for_progress=self.epochs,
            label='BN recalibration',
        )
        self._last_train_stats = {
            'valid_batches': valid_batches,
            'skipped_batches': 0,
            'nonfinite_batches': 0,
        }
        print(f"  - recalibrated BatchNorm stats with {valid_batches} batches in {recal_time:.2f}s")

        val_loss, val_acc = self.validate(epoch='bn_recal')
        current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0.0
        self._append_history_row(
            epoch=self._next_history_epoch(),
            train_loss=0.0,
            train_acc=0.0,
            val_loss=val_loss,
            val_acc=val_acc,
            learning_rate=current_lr,
            epoch_time=recal_time,
            smartpaf_phase='bn_recal',
            collapse_guard_triggered=0,
        )

        if self.save_checkpoints:
            is_new_best = val_acc > self.best_acc
            if is_new_best:
                self.best_acc = val_acc
                self.save_checkpoint(self.history['epoch'][-1], is_best=True)
            self.save_checkpoint(self.history['epoch'][-1], is_best=False, filename='bn_recalibrated_model.pth')

        self.save_history()
        print(f"  - BN recalibrated validation: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        return {'val_loss': val_loss, 'val_acc': val_acc, 'batches': valid_batches}

    def _maybe_update_swa_model(self, epoch):
        if not self.swa_enabled or epoch < self.swa_start_epoch:
            return
        if self._swa_model is None:
            self._swa_model = AveragedModel(self.model).to(self.device)
            print(f"✓ SWA started at epoch {epoch}")
        self._swa_model.update_parameters(self.model)
        self._swa_updates += 1
        print(f"  - SWA update count: {self._swa_updates}")

    def _run_swa_evaluation(self):
        if not self.swa_enabled:
            return None
        if self._swa_model is None or self._swa_updates == 0:
            print("SWA: skipped, no averaged checkpoints")
            return None

        swa_model = self._swa_model.module.to(self.device)
        print("✓ SWA evaluation:")
        print(f"  - updates={self._swa_updates}, start_epoch={self.swa_start_epoch}")

        valid_batches = 0
        recal_time = 0.0
        if self.swa_bn_update:
            print(f"  - BN update batches={self.swa_bn_batches or 'all'}")
            valid_batches, recal_time = self._recalibrate_model_batchnorm(
                swa_model,
                batches=self.swa_bn_batches,
                epoch_for_progress=self.epochs,
                label='SWA BN update',
            )
        self._last_train_stats = {
            'valid_batches': valid_batches,
            'skipped_batches': 0,
            'nonfinite_batches': 0,
        }

        val_loss, val_acc = self._validate_with_model(swa_model, epoch='swa')
        current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0.0
        self._append_history_row(
            epoch=self._next_history_epoch(),
            train_loss=0.0,
            train_acc=0.0,
            val_loss=val_loss,
            val_acc=val_acc,
            learning_rate=current_lr,
            epoch_time=recal_time,
            smartpaf_phase='swa',
            collapse_guard_triggered=0,
        )

        original_model = self.model
        self.model = swa_model
        try:
            if self.save_checkpoints:
                is_new_best = val_acc > self.best_acc
                if is_new_best:
                    self.best_acc = val_acc
                    self.save_checkpoint(self.history['epoch'][-1], is_best=True)
                self.save_checkpoint(self.history['epoch'][-1], is_best=False, filename='swa_model.pth')
        finally:
            self.model = original_model

        self.save_history()
        print(f"  - SWA validation: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        return {'val_loss': val_loss, 'val_acc': val_acc, 'updates': self._swa_updates}
    
    def _restore_best_and_scale_lr(self, lr_factor, reason):
        restored = False
        best_path = os.path.join(self.result_dir, 'best_model.pth')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            model_state = checkpoint.get('model_state_dict')
            if model_state is not None:
                self.model.load_state_dict(model_state, strict=self.resume_strict)
                restored = True
                print(f"  ✓ Restored best model from {best_path}")
        else:
            print(f"  - best model not found at {best_path}; cannot restore for {reason}")

        for group in self.optimizer.param_groups:
            group['lr'] *= lr_factor
        if self.scaler is not None:
            self.scaler = GradScaler()
        print(f"  ✓ Scaled LR by factor {lr_factor} ({reason})")
        return restored

    def _should_reject_nonimproving_poly(self, smartpaf_phase, val_acc):
        return (
            self.smartpaf_at_reject_nonimproving_poly
            and smartpaf_phase == 'poly'
            and val_acc <= self.best_acc + self.smartpaf_at_accept_min_delta
        )

    def _reject_nonimproving_poly(self, val_acc):
        print(
            f"  - Rejecting non-improving poly phase: "
            f"val_acc={val_acc:.2f}%, best={self.best_acc:.2f}%"
        )
        self._last_collapse_guard_restored = self._restore_best_and_scale_lr(
            lr_factor=self.smartpaf_at_reject_lr_factor,
            reason='non-improving AT poly phase',
        )
        return self._last_collapse_guard_restored

    def _run_collapse_guard(self, epoch, val_acc):
        self._last_collapse_guard_restored = False
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
            self._last_collapse_guard_restored = self._restore_best_and_scale_lr(
                lr_factor=self.collapse_guard_lr_factor,
                reason='collapse guard',
            )

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
        if self.swa_enabled:
            print(f"SWA: enabled (start_epoch={self.swa_start_epoch}, bn_update={self.swa_bn_update})")
        self._ensure_history_fields()

        if self.start_epoch > 1:
            print(f"Resuming training from epoch {self.start_epoch}")

        start_time = time.time()

        if self.nan_debug and not self._nan_debug_active:
            self._register_nan_hooks()

        try:
            self._run_smartpaf_ss_calibration()
            self._run_smartpaf_ct_init()
            for epoch in range(self.start_epoch, self.epochs + 1):
                self._prepare_smartpaf_phase_restore_group(epoch)

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
                smartpaf_phase = self._current_smartpaf_phase(epoch)
                if (
                    self.smartpaf_at_reject_before_collapse_guard
                    and self._should_reject_nonimproving_poly(smartpaf_phase, val_acc)
                ):
                    self._last_collapse_guard_restored = False
                    collapse_triggered = False
                    rejected_phase = self._reject_nonimproving_poly(val_acc)
                else:
                    try:
                        collapse_triggered = self._run_collapse_guard(epoch, val_acc)
                    except RuntimeError as exc:
                        collapse_triggered = True
                        collapse_error = exc

                    rejected_phase = (
                        collapse_triggered
                        and collapse_error is None
                        and self._last_collapse_guard_restored
                    )
                    if (
                        not rejected_phase
                        and collapse_error is None
                        and self._should_reject_nonimproving_poly(smartpaf_phase, val_acc)
                    ):
                        rejected_phase = self._reject_nonimproving_poly(val_acc)

                if rejected_phase and smartpaf_phase == 'poly':
                    self._mark_rejected_poly_group(epoch)
                elif smartpaf_phase == 'poly':
                    self._mark_accepted_poly_group(epoch)

                if rejected_phase and self.smartpaf_revalidate_rejected_phase:
                    print("  - Revalidating restored model for rejected phase")
                    val_loss, val_acc = self.validate(epoch=f'{epoch}_rejected')
                    current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0.0
                    epoch_time = time.time() - epoch_start
                    smartpaf_phase = f"{smartpaf_phase}_rejected"

                self._update_smartpaf_phase_restore_group(epoch, val_acc)

                # 记录历史
                self._append_history_row(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    learning_rate=current_lr,
                    epoch_time=epoch_time,
                    smartpaf_phase=smartpaf_phase,
                    collapse_guard_triggered=collapse_triggered,
                )

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

                if collapse_error is None:
                    self._update_smartpaf_overfit_dropout(epoch, train_acc, val_acc)

                if collapse_error is None:
                    self._maybe_update_swa_model(epoch)
                
                # 保存历史
                self.save_history()
                if collapse_error is not None:
                    raise collapse_error

            finalized_phase = self._finalize_smartpaf_phase_restore_group(final=True)
            if finalized_phase is not None:
                restore_start = time.time()
                val_loss, val_acc = self.validate(epoch='phase_restore')
                restore_time = time.time() - restore_start
                current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0.0
                self._last_train_stats = {
                    'valid_batches': 0,
                    'skipped_batches': 0,
                    'nonfinite_batches': 0,
                }
                self._append_history_row(
                    epoch=self._next_history_epoch(),
                    train_loss=0.0,
                    train_acc=0.0,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    learning_rate=current_lr,
                    epoch_time=restore_time,
                    smartpaf_phase='phase_restore',
                    collapse_guard_triggered=0,
                )
                if self.save_checkpoints:
                    is_new_best = val_acc > self.best_acc
                    if is_new_best:
                        self.best_acc = val_acc
                        self.save_checkpoint(self.history['epoch'][-1], is_best=True)
                    self.save_checkpoint(self.history['epoch'][-1], is_best=False, filename='phase_restored_model.pth')
                self.save_history()
        finally:
            if self.smartpaf_alternate_training:
                self._restore_all_trainable()
            self._remove_smartpaf_overfit_dropout_hooks()
            if self._nan_debug_active:
                self._remove_nan_hooks()

        self._run_swa_evaluation()
        self._run_bn_recalibration()
        self._run_smartpaf_ds_to_ss_evaluation()
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print("训练完成!")
        print(f"{'=' * 60}")
        print(f"总训练时间: {total_time / 3600:.2f} 小时")
        print(f"最佳验证准确率: {self.best_acc:.2f}%")
        print(f"训练历史已保存到: {os.path.join(self.result_dir, 'train_history.csv')}")
        
        return self.best_acc
