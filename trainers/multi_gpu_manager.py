"""
多GPU管理器
支持并行训练多个模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from .base_trainer import Trainer
from models import get_model
from data import create_dataloaders
from utils import set_random_seed
from typing import List, Dict, Optional
import multiprocessing as mp
import os


def _model_has_module_class(model, class_name):
    return any(module.__class__.__name__ == class_name for module in model.modules())


def _collect_module_param_ids(model, class_name):
    param_ids = set()
    for module in model.modules():
        if module.__class__.__name__ == class_name:
            for param in module.parameters(recurse=True):
                param_ids.add(id(param))
    return param_ids


def _collect_batchnorm_param_ids(model):
    param_ids = set()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for param in module.parameters(recurse=False):
                param_ids.add(id(param))
    return param_ids


def _is_no_decay_parameter(name):
    parts = name.split('.')
    return (
        name.endswith('.bias')
        or name.endswith('.beta')
        or name.endswith('.gate_scale')
        or any(part.startswith('bn') or 'batchnorm' in part.lower() for part in parts)
    )


def create_smart_optimizer(
    model,
    lr=0.001,
    optimizer_type='adamw',
    weight_decay=1e-4,
    poly_weight_decay=0.0,
    beta_weight_decay=0.0,
    poly_lr_mult=1.0,
    poly_scale_lr_mult=None,
    normal_lr_mult=1.0,
):
    """智能优化器：为不同类型的参数使用不同的学习率/权重衰减策略。"""
    poly_params = []
    poly_scale_params = []
    no_decay_params = []
    normal_params = []
    poly_param_ids = _collect_module_param_ids(model, "StablePoly4")
    bn_param_ids = _collect_batchnorm_param_ids(model)
    if poly_scale_lr_mult is None:
        poly_scale_lr_mult = poly_lr_mult

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in poly_param_ids:
            if name.endswith(".log_in_scale"):
                poly_scale_params.append(param)
            else:
                poly_params.append(param)
        elif id(param) in bn_param_ids or _is_no_decay_parameter(name):
            no_decay_params.append(param)
        else:
            normal_params.append(param)

    param_groups = []
    if normal_params:
        param_groups.append({
            "params": normal_params,
            "lr": lr * normal_lr_mult,
            "weight_decay": weight_decay,
            "name": "normal_decay",
            "is_poly": False,
        })
    if no_decay_params:
        param_groups.append({
            "params": no_decay_params,
            "lr": lr * normal_lr_mult,
            "weight_decay": beta_weight_decay,
            "name": "normal_no_decay",
            "is_poly": False,
        })
    if poly_params:
        param_groups.append({
            "params": poly_params,
            "lr": lr * poly_lr_mult,
            "weight_decay": poly_weight_decay,
            "name": "poly",
            "is_poly": True,
        })
    if poly_scale_params:
        param_groups.append({
            "params": poly_scale_params,
            "lr": lr * poly_scale_lr_mult,
            "weight_decay": poly_weight_decay,
            "name": "poly_scale",
            "is_poly": True,
        })

    optimizer_key = str(optimizer_type).lower()
    if optimizer_key == 'adamw':
        optimizer = optim.AdamW(param_groups, lr=lr)
    elif optimizer_key == 'adam':
        optimizer = optim.Adam(param_groups, lr=lr)
    elif optimizer_key == 'sgd':
        optimizer = optim.SGD(param_groups, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

    print("优化器参数组:")
    for group in optimizer.param_groups:
        n_params = sum(p.numel() for p in group['params'])
        print(
            f"  - {group.get('name', 'unnamed')}: params={n_params:,}, "
            f"lr={group['lr']:.6g}, wd={group.get('weight_decay', 0):.6g}, "
            f"is_poly={group.get('is_poly', False)}"
        )

    return optimizer


def create_lr_scheduler(optimizer, model_config, default_lr, epochs):
    scheduler_name = str(model_config.get('scheduler', 'cosine')).lower()
    min_lr_ratio = float(model_config.get('min_lr_ratio', 0.01))
    warmup_epochs = int(model_config.get('warmup_epochs', 0) or 0)
    warmup_start_factor = float(model_config.get('warmup_start_factor', 0.05))

    if scheduler_name in ('none', 'off', 'disabled'):
        return None
    if scheduler_name != 'cosine':
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    eta_min = default_lr * min_lr_ratio
    cosine_epochs = max(1, epochs - warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=eta_min)
    if warmup_epochs <= 0:
        return cosine

    warmup = LinearLR(
        optimizer,
        start_factor=max(1e-8, min(1.0, warmup_start_factor)),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def _process_worker(manager_kwargs, task_queue, result_queue, gpu_id, force, return_details):
    """Process-isolated worker to avoid CUDA/DataLoader hangs from Python threads."""
    manager = MultiGPUManager(**manager_kwargs)
    manager.available_gpus = [gpu_id] if gpu_id is not None else []
    while True:
        model_config = task_queue.get()
        if model_config is None:
            break
        model_name = model_config['name']
        try:
            detail = manager.train_model(
                model_config,
                gpu_id,
                force,
                return_details=return_details,
            )
            result_queue.put(('success', model_name, detail))
        except Exception as exc:
            result_queue.put(('failed', model_name, str(exc)))


class MultiGPUManager:
    """多GPU训练管理器"""
    
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        result_dir: str = './results',
        gpus: List[int] = [1, 2, 3],
        excluded_gpus: Optional[List[int]] = None,
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
            gpus: 可用的GPU设备列表，默认使用GPU 1/2/3并避开GPU 0
            excluded_gpus: 运行时强制排除的GPU列表，默认排除physical GPU 0
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
        self.excluded_gpus = [0] if excluded_gpus is None else list(excluded_gpus)
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
        self._init_kwargs = {
            'train_dir': train_dir,
            'val_dir': val_dir,
            'result_dir': result_dir,
            'gpus': gpus,
            'excluded_gpus': self.excluded_gpus,
            'num_classes': num_classes,
            'default_epochs': default_epochs,
            'default_batch_size': default_batch_size,
            'default_lr': default_lr,
            'default_num_workers': default_num_workers,
            'use_memory_fs': use_memory_fs,
            'dataset': dataset,
            'download': download,
            'input_size': input_size,
            'seed': seed,
        }

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
        excluded = set(self.excluded_gpus or [])
        if excluded:
            print(f"默认排除GPU: {sorted(excluded)} (physical GPU0/第一张V100存在memory/ECC风险)")
        for gpu_id in self.gpus:
            if gpu_id in excluded:
                print(f"⚠ 跳过 GPU {gpu_id}: 已按环境约束排除")
                continue
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
        uses_stablepoly_by_name = (
            'stablepoly' in model_name.lower()
            or 'stablepoly' in str(model_class).lower()
        )
        
        # 设置设备
        device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cpu')
        
        # 创建数据加载器
        batch_size = model_config.get('batch_size', self.default_batch_size)
        num_workers = model_config.get('num_workers', self.default_num_workers)
        prefetch_factor = model_config.get('prefetch_factor', 4)
        
        # 每个GPU进程创建独立的DataLoader实例
        # 如果启用use_memory_fs，所有进程都会从同一个内存文件系统路径读取数据
        # 这样实现了数据源的共享，而无需共享DataLoader实例
        train_loader, val_loader, _, _ = create_dataloaders(
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda',
            prefetch_factor=prefetch_factor,
            use_memory_fs=self.use_memory_fs,
            dataset=self.dataset,
            download=self.download,
            input_size=self.input_size,
            seed=self.seed
        )
        
        # 创建模型
        model_params = dict(model_config.get('params', {}) or {})
        model_params['num_classes'] = model_params.get('num_classes', self.num_classes)
        
        model = get_model(model_class, **model_params)
        model = model.to(device)

        uses_stablepoly = uses_stablepoly_by_name or _model_has_module_class(model, "StablePoly4")
        if uses_stablepoly:
            print("⚠ 检测到 StablePoly4 激活函数，将启用 SmartPAF 稳定训练默认项")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,}")
        
        # 损失函数
        label_smoothing = float(model_config.get('label_smoothing', 0.0) or 0.0)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # 优化器 - 使用智能优化器，为不同参数类型使用不同权重衰减
        lr = model_config.get('learning_rate', self.default_lr)
        optimizer = create_smart_optimizer(
            model,
            lr=lr,
            optimizer_type=model_config.get('optimizer_type', 'adamw'),
            weight_decay=model_config.get('weight_decay', 1e-4),
            poly_weight_decay=model_config.get('poly_weight_decay', 0.0 if uses_stablepoly else 1e-4),
            beta_weight_decay=model_config.get('beta_weight_decay', 0.0),
            poly_lr_mult=model_config.get('poly_lr_mult', 1.0),
            poly_scale_lr_mult=model_config.get('poly_scale_lr_mult', None),
            normal_lr_mult=model_config.get('normal_lr_mult', 1.0),
        )

        # 学习率调度器 - 支持 warmup + cosine
        epochs = model_config.get('epochs', self.default_epochs)
        scheduler = create_lr_scheduler(optimizer, model_config, lr, epochs)
        if scheduler is not None:
            print(
                f"学习率调度器: {model_config.get('scheduler', 'cosine')} "
                f"(warmup_epochs={model_config.get('warmup_epochs', 0)}, "
                f"min_lr_ratio={model_config.get('min_lr_ratio', 0.01)})"
            )
        
        # 结果保存目录
        # model_result_dir 已在上方解析（支持按模型覆盖）
        
        # 创建训练器
        grad_clip_max_norm = model_config.get('grad_clip_max_norm', 1.0)
        
        # 从配置中获取是否保存检查点（默认True）
        save_checkpoints = model_config.get('save_checkpoints', True)
        save_freq = model_config.get('save_freq', 10)
        use_amp = model_config.get('use_amp', True)
        val_force_fp32 = model_config.get('val_force_fp32', True)
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
            'val_force_fp32',
            'scheduler',
            'warmup_epochs',
            'warmup_start_factor',
            'min_lr_ratio',
            'optimizer_type',
            'weight_decay',
            'poly_weight_decay',
            'beta_weight_decay',
            'poly_lr_mult',
            'poly_scale_lr_mult',
            'normal_lr_mult',
            'label_smoothing',
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
            val_force_fp32=val_force_fp32,
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
            # 进程级并行训练：避免多线程共享 CUDA context / DataLoader 导致卡住
            print(f"\n并行训练模式，使用 {len(self.available_gpus)} 个GPU（process workers）")

            try:
                ctx = mp.get_context('spawn')
            except RuntimeError:
                ctx = mp.get_context()
            task_queue = ctx.Queue()
            result_queue = ctx.Queue()

            for model_config in model_configs:
                task_queue.put(model_config)

            worker_count = min(len(self.available_gpus), len(model_configs))
            for _ in range(worker_count):
                task_queue.put(None)

            processes = []
            for i, gpu_id in enumerate(self.available_gpus[:worker_count]):
                process = ctx.Process(
                    target=_process_worker,
                    args=(self._init_kwargs, task_queue, result_queue, gpu_id, force, return_details),
                    name=f"trainer-gpu-{gpu_id}",
                )
                process.start()
                processes.append(process)

            received = 0
            while received < len(model_configs):
                status, model_name, value = result_queue.get()
                received += 1
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

            for process in processes:
                process.join(timeout=30)
                if process.is_alive():
                    print(f"⚠ worker {process.name} 未正常退出，正在终止")
                    process.terminate()
                    process.join(timeout=10)

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
