"""
多GPU管理器
支持并行训练多个模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from .base_trainer import Trainer
from models import get_model
from data import create_dataloaders
from typing import List, Dict, Optional
import threading
import queue


class MultiGPUManager:
    """多GPU训练管理器"""
    
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        result_dir: str = './results',
        gpus: List[int] = [0, 1, 2, 3],
        num_classes: int = 100,
        default_epochs: int = 60,
        default_batch_size: int = 128,
        default_lr: float = 0.001,
        default_num_workers: int = 16,
        use_cache: bool = True
    ):
        """
        初始化多GPU管理器
        
        Args:
            train_dir: 训练集目录
            val_dir: 验证集目录
            result_dir: 结果保存目录
            gpus: 可用的GPU设备列表
            num_classes: 类别数量
            default_epochs: 默认训练epoch数
            default_batch_size: 默认批次大小
            default_lr: 默认学习率
            default_num_workers: 默认数据加载worker数
            use_cache: 是否使用内存缓存
        """
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.result_dir = result_dir
        self.gpus = gpus
        self.num_classes = num_classes
        self.default_epochs = default_epochs
        self.default_batch_size = default_batch_size
        self.default_lr = default_lr
        self.default_num_workers = default_num_workers
        self.use_cache = use_cache
        
        # 创建结果目录
        import os
        os.makedirs(result_dir, exist_ok=True)
        
        # 检查GPU可用性
        self.available_gpus = self._check_available_gpus()
        print(f"可用GPU: {self.available_gpus}")
    
    def _check_available_gpus(self) -> List[int]:
        """
        检查可用的GPU
        
        Returns:
            可用的GPU列表
        """
        if not torch.cuda.is_available():
            print("⚠ CUDA不可用，将使用CPU")
            return []
        
        available = []
        for gpu_id in self.gpus:
            try:
                torch.cuda.set_device(gpu_id)
                props = torch.cuda.get_device_properties(gpu_id)
                available.append(gpu_id)
                print(f"✓ GPU {gpu_id}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
            except Exception as e:
                print(f"⚠ GPU {gpu_id} 不可用: {e}")
        
        return available
    
    def is_model_trained(self, model_name: str) -> bool:
        """
        检查模型是否已经训练过
        
        Args:
            model_name: 模型名称
        
        Returns:
            是否已训练
        """
        import os
        model_result_dir = f"{self.result_dir}/{model_name}"
        best_model_path = f"{model_result_dir}/best_model.pth"
        
        if os.path.exists(best_model_path):
            print(f"✓ 模型 {model_name} 已训练，结果在 {model_result_dir}")
            return True
        return False
    
    def train_model(
        self,
        model_config: Dict,
        gpu_id: int,
        force: bool = False
    ) -> Optional[float]:
        """
        在指定GPU上训练单个模型
        
        Args:
            model_config: 模型配置字典
            gpu_id: GPU设备ID
            force: 是否强制重新训练
        
        Returns:
            最佳验证准确率，如果跳过则返回None
        """
        model_name = model_config['name']
        
        # 检查是否已训练
        if not force and self.is_model_trained(model_name):
            return None
        
        print(f"\n{'=' * 60}")
        print(f"开始训练模型: {model_name} (GPU: {gpu_id})")
        print(f"{'=' * 60}")
        
        # 设置设备
        device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
        
        # 创建数据加载器
        batch_size = model_config.get('batch_size', self.default_batch_size)
        num_workers = model_config.get('num_workers', self.default_num_workers)
        
        train_loader, val_loader, _, _ = create_dataloaders(
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda',
            use_cache=self.use_cache
        )
        
        # 创建模型
        model_params = model_config.get('params', {})
        model_params['num_classes'] = model_params.get('num_classes', self.num_classes)
        
        model = get_model(model_name, **model_params)
        model = model.to(device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,}")
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 优化器
        lr = model_config.get('learning_rate', self.default_lr)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
        
        # 学习率调度器
        epochs = model_config.get('epochs', self.default_epochs)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr * 0.01
        )
        
        # 结果保存目录
        model_result_dir = f"{self.result_dir}/{model_name}"
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            result_dir=model_result_dir,
            epochs=epochs,
            scheduler=scheduler,
            use_amp=True,
            save_freq=10
        )
        
        # 开始训练
        best_acc = trainer.train()
        
        return best_acc
    
    def train_models(
        self,
        model_configs: List[Dict],
        force: bool = False,
        parallel: bool = True
    ) -> Dict[str, float]:
        """
        训练多个模型
        
        Args:
            model_configs: 模型配置列表
            force: 是否强制重新训练所有模型
            parallel: 是否并行训练
        
        Returns:
            模型名称到最佳准确率的映射
        """
        results = {}
        
        if not self.available_gpus:
            print("⚠ 没有可用的GPU，将使用CPU训练")
            device = torch.device('cpu')
            
            for model_config in model_configs:
                best_acc = self.train_model(model_config, None, force)
                if best_acc is not None:
                    results[model_config['name']] = best_acc
        elif parallel and len(self.available_gpus) > 1:
            # 并行训练
            print(f"\n并行训练模式，使用 {len(self.available_gpus)} 个GPU")
            
            # 任务队列
            task_queue = queue.Queue()
            result_queue = queue.Queue()
            
            # 添加任务到队列
            for model_config in model_configs:
                task_queue.put(model_config)
            
            # 工作线程函数
            def worker(gpu_id):
                while not task_queue.empty():
                    try:
                        model_config = task_queue.get(timeout=1)
                        best_acc = self.train_model(model_config, gpu_id, force)
                        result_queue.put((model_config['name'], best_acc))
                        task_queue.task_done()
                    except queue.Empty:
                        break
            
            # 创建工作线程
            threads = []
            for i, gpu_id in enumerate(self.available_gpus):
                if i < len(model_configs):
                    thread = threading.Thread(target=worker, args=(gpu_id,))
                    thread.start()
                    threads.append(thread)
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            # 收集结果
            while not result_queue.empty():
                model_name, best_acc = result_queue.get()
                if best_acc is not None:
                    results[model_name] = best_acc
        
        else:
            # 串行训练
            print(f"\n串行训练模式，使用 GPU {self.available_gpus[0]}")
            
            for model_config in model_configs:
                best_acc = self.train_model(
                    model_config,
                    self.available_gpus[0] if self.available_gpus else None,
                    force
                )
                if best_acc is not None:
                    results[model_config['name']] = best_acc
        
        # 打印总结
        print(f"\n{'=' * 60}")
        print("训练总结")
        print(f"{'=' * 60}")
        
        if results:
            for model_name, acc in results.items():
                print(f"  {model_name}: {acc:.2f}%")
        
        # 统计跳过的模型
        skipped = [m['name'] for m in model_configs 
                  if m['name'] not in results and not force]
        if skipped:
            print(f"\n跳过的模型（已训练）: {', '.join(skipped)}")
        
        return results