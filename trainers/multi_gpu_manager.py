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
from utils import set_random_seed
from typing import List, Dict, Optional
import threading
import queue
import os


def create_smart_optimizer(model, lr=0.001):
    """
    智能优化器：为不同类型的参数使用不同的权重衰减策略
    
    Args:
        model: 模型
        lr: 学习率
    
    Returns:
        optimizer: 配置好的优化器
    """
    poly_params = []
    beta_params = []
    normal_params = []
    
    for name, param in model.named_parameters():
        # LearnableSwish和LearnableRelu的beta - 不约束防止归零
        # 必须先匹配beta，因为'.act.beta'包含'.act.b'
        if name.endswith('.beta'):
            beta_params.append(param)
        # StablePoly4的多项式系数 (如: .act.a, .act.b, .act.c, .act.d, .act.e) - 强约束防止爆炸
        # 使用更精确的匹配，确保只匹配单个字母作为参数名
        elif any(name.endswith(f'.act.{p}') for p in ['a', 'b', 'c', 'd', 'e']):
            poly_params.append(param)
        # 普通权重（卷积层、线性层等）- 标准约束
        else:
            normal_params.append(param)
    
    # 创建参数组，使用不同的权重衰减
    optimizer = optim.AdamW(
        [
            {"params": normal_params, "weight_decay": 1e-4},
            {"params": poly_params, "weight_decay": 0.1},  # Poly 强约束
            {"params": beta_params, "weight_decay": 0.0},   # Beta 不约束
        ],
        lr=lr,
    )
    
    return optimizer


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
        use_memory_fs: bool = True,
        dataset: str = "imagenet100",
        download: bool = False,
        input_size: Optional[int] = None,
        seed: Optional[int] = 42
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
            use_memory_fs: 是否使用内存文件系统（推荐，避免并发内存问题）
                           注意：内存文件系统本身就可以被所有进程共享访问，
                           每个进程会创建独立的DataLoader实例
            dataset: 数据集类型 (imagenet100/imagenet1k/cifar10/cifar100)
            download: 是否允许下载数据集（仅CIFAR有效）
            input_size: 输入图像大小（可选，覆盖默认值）
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
        self.use_memory_fs = use_memory_fs
        self.dataset = dataset
        self.download = download
        self.input_size = input_size
        self.seed = seed

        set_random_seed(self.seed)
        
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

    def _find_latest_checkpoint(self, model_result_dir: str) -> Optional[str]:
        if not os.path.isdir(model_result_dir):
            return None
        candidates = []
        for name in os.listdir(model_result_dir):
            if name.startswith('checkpoint_epoch_') and name.endswith('.pth'):
                try:
                    epoch_str = name[len('checkpoint_epoch_'):-4]
                    epoch = int(epoch_str)
                except ValueError:
                    continue
                candidates.append((epoch, os.path.join(model_result_dir, name)))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    def _resolve_resume_path(self, model_config: Dict, model_result_dir: str) -> Optional[str]:
        resume_path = model_config.get('resume_path')
        resume_enabled = bool(model_config.get('resume', False) or resume_path)
        if not resume_enabled:
            return None

        resume_mode = model_config.get('resume_mode', 'auto')
        if not resume_path:
            best_path = os.path.join(model_result_dir, 'best_model.pth')
            latest_path = self._find_latest_checkpoint(model_result_dir)
            if resume_mode == 'best':
                resume_path = best_path
            elif resume_mode == 'last':
                resume_path = latest_path
            else:
                resume_path = best_path if os.path.exists(best_path) else latest_path

        if resume_path and not os.path.exists(resume_path):
            print(f"Warning: resume checkpoint not found: {resume_path}")
            return None

        if resume_path:
            print(f"Resume enabled. Using checkpoint: {resume_path}")
        return resume_path
    
    def is_model_trained(self, model_name: str, model_result_dir: Optional[str] = None) -> bool:
        """
        检查模型是否已经训练过
        
        Args:
            model_name: 模型名称
        
        Returns:
            是否已训练
        """
        import os
        if model_result_dir is None:
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
        force: bool = False,
        return_details: bool = False
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
        model_class = model_config.get('class', model_name)
        model_result_dir = model_config.get('result_dir', f"{self.result_dir}/{model_name}")
        
        # 检查是否已训练
        if not force and self.is_model_trained(model_name, model_result_dir=model_result_dir):
            return None
        
        print(f"\n{'=' * 60}")
        print(f"开始训练模型: {model_name} (GPU: {gpu_id})")
        print(f"{'=' * 60}")
        
        # 检测是否使用 StablePoly4 激活函数
        uses_stablepoly = (
            'stablepoly' in model_name.lower()
            or 'stablepoly' in str(model_class).lower()
        )
        if uses_stablepoly:
            print("⚠ 检测到 StablePoly4 激活函数，将使用更严格的梯度裁剪")
        
        # 设置设备
        device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
        
        # 创建数据加载器
        batch_size = model_config.get('batch_size', self.default_batch_size)
        num_workers = model_config.get('num_workers', self.default_num_workers)
        
        # 每个GPU进程创建独立的DataLoader实例
        # 如果启用use_memory_fs，所有进程都会从同一个内存文件系统路径读取数据
        # 这样实现了数据源的共享，而无需共享DataLoader实例
        train_loader, val_loader, _, _ = create_dataloaders(
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda',
            use_memory_fs=self.use_memory_fs,
            dataset=self.dataset,
            download=self.download,
            input_size=self.input_size,
            seed=self.seed
        )
        
        # 创建模型
        model_params = model_config.get('params', {})
        model_params['num_classes'] = model_params.get('num_classes', self.num_classes)
        
        model = get_model(model_class, **model_params)
        model = model.to(device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,}")
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 优化器 - 使用智能优化器，为不同参数类型使用不同权重衰减
        lr = model_config.get('learning_rate', self.default_lr)
        optimizer = create_smart_optimizer(model, lr=lr)
        
        # 学习率调度器 - 使用实际训练轮数
        epochs = model_config.get('epochs', self.default_epochs)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr * 0.01
        )
        
        # 结果保存目录
        # model_result_dir 已在上方解析（支持按模型覆盖）
        
        # 创建训练器
        # 对于 StablePoly4 模型，使用更严格的梯度裁剪
        grad_clip_max_norm = model_config.get(
            'grad_clip_max_norm',
            0.5 if uses_stablepoly else 1.0
        )
        
        # 从配置中获取是否保存检查点（默认True）
        save_checkpoints = model_config.get('save_checkpoints', True)
        save_freq = model_config.get('save_freq', 10)
        use_amp = model_config.get('use_amp', True)
        resume_path = self._resolve_resume_path(model_config, model_result_dir)
        if (model_config.get('resume', False) or model_config.get('resume_path')) and not resume_path:
            print("Resume enabled but no checkpoint found. Starting from scratch.")
        trainer_kwargs = dict(model_config.get('trainer_kwargs', {}) or {})
        for key in (
            'model',
            'train_loader',
            'val_loader',
            'criterion',
            'optimizer',
            'device',
            'result_dir',
            'epochs',
            'scheduler',
            'use_amp',
            'save_freq',
            'save_checkpoints',
            'grad_clip_max_norm',
            'resume_path',
        ):
            trainer_kwargs.pop(key, None)
        
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
            use_amp=use_amp,
            save_freq=save_freq,
            save_checkpoints=save_checkpoints,
            grad_clip_max_norm=grad_clip_max_norm,
            resume_path=resume_path,
            **trainer_kwargs,
        )
        
        # 开始训练
        try:
            best_acc = trainer.train()
            if return_details:
                return {
                    'best_acc': best_acc,
                    'train_time': sum(trainer.history['epoch_time']) if trainer.history['epoch_time'] else 0,
                    'final_train_loss': trainer.history['train_loss'][-1] if trainer.history['train_loss'] else 0,
                    'final_val_loss': trainer.history['val_loss'][-1] if trainer.history['val_loss'] else 0,
                    'epochs': epochs,
                }
            return best_acc
        except Exception as e:
            print(f"❌ 训练失败: {model_name}")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()
            raise  # 重新抛出异常供上层处理
    
    def train_models(
        self,
        model_configs: List[Dict],
        force: bool = False,
        parallel: bool = True,
        return_details: bool = False
    ) -> Dict[str, Dict]:
        """
        训练多个模型
        
        Args:
            model_configs: 模型配置列表
            force: 是否强制重新训练所有模型
            parallel: 是否并行训练
        
        Returns:
            包含成功、失败、跳过模型信息的字典
        """
        results = {'success': {}, 'failed': {}, 'skipped': {}}
        if return_details:
            results['details'] = {}
        
        if not self.available_gpus:
            print("⚠ 没有可用的GPU，将使用CPU训练")
            device = torch.device('cpu')
            
            for model_config in model_configs:
                model_name = model_config['name']
                try:
                    detail = self.train_model(
                        model_config,
                        None,
                        force,
                        return_details=return_details
                    )
                    if detail is not None:
                        if return_details:
                            results['success'][model_name] = detail['best_acc']
                            results['details'][model_name] = detail
                        else:
                            results['success'][model_name] = detail
                    else:
                        results['skipped'][model_name] = '已训练'
                except Exception as e:
                    results['failed'][model_name] = str(e)
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
                        model_name = model_config['name']
                        try:
                            detail = self.train_model(
                                model_config,
                                gpu_id,
                                force,
                                return_details=return_details
                            )
                            result_queue.put(('success', model_name, detail))
                        except Exception as e:
                            result_queue.put(('failed', model_name, str(e)))
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
                status, model_name, value = result_queue.get()
                if status == 'success':
                    if value is not None:
                        if return_details:
                            results['success'][model_name] = value['best_acc']
                            results['details'][model_name] = value
                        else:
                            results['success'][model_name] = value
                    else:
                        results['skipped'][model_name] = '已训练'
                elif status == 'failed':
                    results['failed'][model_name] = value
        
        else:
            # 串行训练
            print(f"\n串行训练模式，使用 GPU {self.available_gpus[0]}")
            
            for model_config in model_configs:
                model_name = model_config['name']
                try:
                    detail = self.train_model(
                        model_config,
                        self.available_gpus[0] if self.available_gpus else None,
                        force,
                        return_details=return_details
                    )
                    if detail is not None:
                        if return_details:
                            results['success'][model_name] = detail['best_acc']
                            results['details'][model_name] = detail
                        else:
                            results['success'][model_name] = detail
                    else:
                        results['skipped'][model_name] = '已训练'
                except Exception as e:
                    results['failed'][model_name] = str(e)
        
        # 打印详细总结
        print(f"\n{'=' * 60}")
        print("训练总结")
        print(f"{'=' * 60}")
        
        total = len(model_configs)
        success_count = len(results['success'])
        failed_count = len(results['failed'])
        skipped_count = len(results['skipped'])
        
        print(f"\n总计: {total} 个模型")
        print(f"  ✓ 成功: {success_count}")
        print(f"  ✗ 失败: {failed_count}")
        print(f"  ○ 跳过: {skipped_count}")
        
        if results['success']:
            print(f"\n成功训练的模型:")
            for model_name, acc in results['success'].items():
                print(f"  ✓ {model_name}: {acc:.2f}%")
        
        if results['failed']:
            print(f"\n训练失败的模型:")
            for model_name, error in results['failed'].items():
                print(f"  ✗ {model_name}")
                print(f"     错误: {error[:100]}...")  # 截断长错误信息
        
        if results['skipped']:
            print(f"\n跳过的模型: {', '.join(results['skipped'].keys())}")
        
        return results
