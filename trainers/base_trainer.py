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
        grad_clip_max_norm=1.0
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
        self.grad_clip_max_norm = grad_clip_max_norm
        
        # 创建结果目录
        os.makedirs(result_dir, exist_ok=True)
        
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
                self.save_checkpoint(epoch, is_best=True)
            
            # 定期保存检查点
            if epoch % self.save_freq == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # 保存历史
            self.save_history()
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print("训练完成!")
        print(f"{'=' * 60}")
        print(f"总训练时间: {total_time / 3600:.2f} 小时")
        print(f"最佳验证准确率: {self.best_acc:.2f}%")
        print(f"训练历史已保存到: {os.path.join(self.result_dir, 'train_history.csv')}")
        
        return self.best_acc