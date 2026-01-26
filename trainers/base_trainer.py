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
        nan_debug=False,
        val_batch_stats_path=None,
        val_batch_stats_quantile=0.999,
        val_batch_stats_anomaly_only=False,
        val_batch_stats_abs_logit_thresh=None,
        val_batch_stats_margin_thresh=None,
        val_batch_stats_loss_p999_thresh=None
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
            nan_debug: 是否启用NaN定位钩子（默认关闭）
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
        self.nan_debug = nan_debug
        self.val_batch_stats_path = val_batch_stats_path
        self.val_batch_stats_quantile = val_batch_stats_quantile
        self.val_batch_stats_anomaly_only = val_batch_stats_anomaly_only
        self.val_batch_stats_abs_logit_thresh = val_batch_stats_abs_logit_thresh
        self.val_batch_stats_margin_thresh = val_batch_stats_margin_thresh
        self.val_batch_stats_loss_p999_thresh = val_batch_stats_loss_p999_thresh
        self._nan_debug_running = False
        self._nan_hooks = []
        self._nan_triggered = False
        self._nan_debug_active = False

        # 创建结果目录
        os.makedirs(result_dir, exist_ok=True)

        # 自动调整StablePoly4的warmup_epochs
        self._adjust_poly4_warmup()

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
            'epoch_time': []
        }

        # 最佳准确率
        self.best_acc = 0.0

        if self.resume_path:
            self._load_checkpoint(self.resume_path, strict=self.resume_strict)

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

        self.best_acc = checkpoint.get('best_acc', self.best_acc)

        history = checkpoint.get('history')
        if isinstance(history, dict):
            self.history = history

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
    
    def train_one_epoch(self, epoch):
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch
        
        Returns:
            avg_loss: 平均损失
            avg_acc: 平均准确率
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0

        model_name = pathlib.Path(self.result_dir).stem
        
        pbar = tqdm(self.train_loader, desc=f'Epoch [{epoch}]({model_name})', leave=False)
        
        # 在第一个epoch添加详细诊断
        first_batch_diagnostic = (epoch == 1)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # 第一个batch的诊断信息
            if first_batch_diagnostic and batch_idx == 0:
                print(f"\n{'='*60}")
                print(f"第一个batch诊断 (Epoch {epoch}):")
                print(f"  Batch shape: {images.shape}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Labels范围: [{labels.min().item()}, {labels.max().item()}]")
                print(f"  Labels dtype: {labels.dtype}")
                print(f"  唯一标签数: {len(labels.unique())}")
                print(f"{'='*60}\n")
            
            self.optimizer.zero_grad()
            
            # 使用混合精度训练
            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
            with autocast(device_type=device_type, enabled=self.use_amp):
                outputs = self.model(images)
                
                # 第一个batch的输出诊断
                if first_batch_diagnostic and batch_idx == 0:
                    print(f"模型输出诊断:")
                    print(f"  Output shape: {outputs.shape}")
                    print(f"  Output dtype: {outputs.dtype}")
                    print(f"  Output范围: [{outputs.min().item():.2f}, {outputs.max().item():.2f}]")
                    print(f"  Output包含NaN: {torch.isnan(outputs).any().item()}")
                    print(f"  Output包含Inf: {torch.isinf(outputs).any().item()}")
                
                # 重要：将 outputs 转为 float32 再计算 loss
                # 在 1000 类分类任务中，float16 的 log_softmax 容易溢出
                # 因为 exp(logit) 在 logit > 11 时就会变成 inf
                outputs_fp32 = outputs.float()
                
                # Loss计算诊断
                if first_batch_diagnostic and batch_idx == 0:
                    print(f"\nLoss计算前:")
                    print(f"  outputs_fp32范围: [{outputs_fp32.min().item():.2f}, {outputs_fp32.max().item():.2f}]")
                    print(f"  outputs_fp32 dtype: {outputs_fp32.dtype}")
                    print(f"  labels范围: [{labels.min().item()}, {labels.max().item()}]")
                    print(f"  期望类别数: 0 到 {outputs.shape[1] - 1}")
                    
                    # 检查标签是否越界
                    if labels.max().item() >= outputs.shape[1]:
                        print(f"  ❌ 错误: 标签 {labels.max().item()} 超出输出维度 {outputs.shape[1]}!")
                
                # 计算正则化损失
                reg_loss = 0.0
                if self.gate_reg_lambda > 0:
                    for module in self.model.modules():
                        if hasattr(module, 'gate_reg_loss'):
                            reg_loss += module.gate_reg_loss
                    reg_loss = reg_loss * self.gate_reg_lambda
                
                loss = self.criterion(outputs_fp32, labels) + reg_loss
                
                if first_batch_diagnostic and batch_idx == 0:
                    print(f"\nLoss计算后:")
                    print(f"  Loss value: {loss.item():.6f}")
                    print(f"  Loss is finite: {torch.isfinite(loss).all().item()}")
                    print(f"{'='*60}\n")
                    first_batch_diagnostic = False  # 只诊断一次

            if self.nan_debug and not self._nan_debug_running:
                if self._find_nonfinite_tensor(outputs) is not None:
                    print("Non-finite output detected in forward")
                    raise RuntimeError("Non-finite output detected")
                if not torch.isfinite(loss).all().item():
                    print("Non-finite loss detected before backward")
                    raise RuntimeError("Non-finite loss detected")
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
                # 梯度裁剪（在 unscale 之前进行）
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                self.optimizer.step()
            
            # 统计
            loss_value = loss.item()
            
            # Check for NaN in training
            if not torch.isfinite(torch.tensor(loss_value)):
                print(f"\n⚠ Warning: Non-finite loss in training batch! Skipping...")
                continue
            
            total_loss += loss_value
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'reg': f'{reg_loss.item():.4f}' if isinstance(reg_loss, torch.Tensor) else '0.00',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        # Safety check
        if total == 0:
            print("\n⚠ Warning: No valid training samples!")
            return float('inf'), 0.0
        
        avg_loss = total_loss / len(self.train_loader)
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
        
        total_loss = 0.0
        correct = 0
        total = 0
        
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
            pbar = tqdm(self.val_loader, desc='Validating', leave=False)
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                with autocast(device_type=device_type, enabled=self.use_amp):
                    outputs = self.model(images)
                
                # 重要：将 outputs 转回 float32 再计算 loss
                # AMP 下 outputs 可能是 float16，大的 logits 值会导致 log_softmax 溢出
                outputs_fp32 = outputs.float() if self.use_amp else outputs
                loss = self.criterion(outputs_fp32, labels)
                
                # Check for NaN/Inf in loss and outputs
                loss_value = loss.item()
                if not torch.isfinite(loss).all() or not torch.isfinite(outputs).all():
                    # 详细诊断信息
                    out_min = outputs.min().item() if torch.isfinite(outputs.min()) else float('nan')
                    out_max = outputs.max().item() if torch.isfinite(outputs.max()) else float('nan')
                    nan_count = (~torch.isfinite(outputs)).sum().item()
                    print(f"\n⚠ Warning: Non-finite values detected in validation!{f' (Epoch {epoch})' if epoch else ''}")
                    print(f"  Loss: {loss_value}")
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
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
        
        # Safety check: ensure we have valid data
        if total == 0:
            print("\n⚠ Warning: No valid samples in validation!")
            if stats_file is not None:
                stats_file.close()
            return float('inf'), 0.0
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100. * correct / total
        
        # Final NaN check
        if not torch.isfinite(torch.tensor(avg_loss)):
            print(f"\n⚠ Warning: avg_loss is NaN/Inf! Setting to inf.")
            avg_loss = float('inf')
        
        if stats_file is not None:
            stats_file.close()

        return avg_loss, avg_acc
    
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
        print(f"结果保存目录: {self.result_dir}")

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
                
                # 记录历史
                self.history['epoch'].append(epoch)
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['learning_rate'].append(current_lr)
                self.history['epoch_time'].append(epoch_time)
                
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
        finally:
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
