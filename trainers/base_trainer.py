"""
基础训练器
"""

import torch
import torch.nn as nn
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
        nan_debug=False
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
        self.nan_debug = nan_debug
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
        
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # 使用混合精度训练
            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
            with autocast(device_type=device_type):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

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
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self):
        """
        验证模型
        
        Returns:
            avg_loss: 平均损失
            avg_acc: 平均准确率
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating', leave=False)
            
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                with autocast(device_type=device_type):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100. * correct / total
        
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
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_acc': self.best_acc,
            'history': self.history
        }
        
        save_path = os.path.join(self.result_dir, filename)
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = os.path.join(self.result_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ 新的最佳准确率: {self.best_acc:.2f}% - 已保存到 {best_path}")
    
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

        start_time = time.time()

        if self.nan_debug and not self._nan_debug_active:
            self._register_nan_hooks()

        try:
            for epoch in range(1, self.epochs + 1):
                # 为所有需要 epoch 信息的模块更新 epoch
                self._set_epoch_for_model(epoch)
                
                epoch_start = time.time()
                
                # 训练
                train_loss, train_acc = self.train_one_epoch(epoch)
                
                # 验证
                val_loss, val_acc = self.validate()
                
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
                print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
                print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
                print(f"  学习率: {current_lr:.6f}")
                
                # 保存最佳模型
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    if self.save_checkpoints:
                        self.save_checkpoint(epoch, is_best=True)
                
                # 定期保存检查点
                if self.save_checkpoints and self.save_freq and self.save_freq > 0:
                    if epoch % self.save_freq == 0:
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
