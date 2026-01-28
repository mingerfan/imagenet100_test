"""
通用数据加载器
支持 ImageNet-100 / ImageNet-1k / CIFAR-10 / CIFAR-100
可选使用内存文件系统加速（ImageFolder类数据集）
"""

import os
import random
import threading
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .memory_fs import create_memory_fs_manager

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DATASET_ALIASES = {
    "imagenet100": "imagenet100",
    "imagenet-100": "imagenet100",
    "imagenet_100": "imagenet100",
    "imagenet1k": "imagenet1k",
    "imagenet-1k": "imagenet1k",
    "imagenet_1k": "imagenet1k",
    "imagenet": "imagenet1k",
    "cifar10": "cifar10",
    "cifar-10": "cifar10",
    "cifar_10": "cifar10",
    "cifar100": "cifar100",
    "cifar-100": "cifar100",
    "cifar_100": "cifar100",
}
DATASET_INFO = {
    "imagenet100": {
        "label": "ImageNet-100",
        "type": "imagefolder",
        "input_size": 224,
        "num_classes": 100,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "shm_name": "imagenet100",
    },
    "imagenet1k": {
        "label": "ImageNet-1k",
        "type": "imagefolder",
        "input_size": 224,
        "num_classes": 1000,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "shm_name": "imagenet1k",
    },
    "cifar10": {
        "label": "CIFAR-10",
        "type": "cifar",
        "input_size": 32,
        "num_classes": 10,
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010],
        "shm_name": None,
    },
    "cifar100": {
        "label": "CIFAR-100",
        "type": "cifar",
        "input_size": 32,
        "num_classes": 100,
        "mean": [0.5071, 0.4865, 0.4409],
        "std": [0.2673, 0.2564, 0.2761],
        "shm_name": None,
    },
}


class ImageFolderDataset(datasets.ImageFolder):
    """
    标准ImageFolder数据集
    可选通过内存文件系统加速数据加载
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

        # 转换targets为tensor以提高性能
        self.targets = torch.tensor([target for _, target in self.samples])

        print(f"✓ 数据集加载完成: {len(self.samples)} 张图片, {len(self.classes)} 个类别")

    def __getitem__(self, index):
        """
        获取图片并应用transform
        """
        path, target = self.samples[index]
        img = self.loader(path)

        # 应用transform
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index]

    def __len__(self):
        return len(self.samples)


class ImageNet100Dataset(ImageFolderDataset):
    """兼容旧名称"""


def normalize_dataset_name(dataset: str) -> str:
    if dataset is None:
        return "imagenet100"
    normalized = DATASET_ALIASES.get(dataset.lower())
    if normalized is None:
        raise ValueError(f"不支持的数据集类型: {dataset}")
    return normalized


def get_dataset_info(dataset: str) -> dict:
    dataset_name = normalize_dataset_name(dataset)
    return DATASET_INFO[dataset_name].copy()


def get_imagenet_train_transform(input_size: int):
    """
    获取ImageNet训练集数据增强变换
    
    Note: 使用更温和的数据增强策略，避免训练/验证集分布差异过大：
    - RandomResizedCrop scale=(0.08, 1.0): 标准ImageNet增强范围
    - RandomHorizontalFlip: 保持
    - 移除RandomRotation: 对ImageNet效果不好，容易造成过拟合
    - 减弱ColorJitter: 避免颜色扰动过大
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0)),  # 标准ImageNet范围
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 减弱强度
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_imagenet_val_transform(input_size: int):
    """
    获取ImageNet验证集数据变换
    """
    resize_size = max(256, input_size)
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_cifar_train_transform(input_size: int, mean, std):
    """
    获取CIFAR训练集数据增强变换
    """
    if input_size == 32:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_cifar_val_transform(input_size: int, mean, std):
    """
    获取CIFAR验证集数据变换
    """
    if input_size == 32:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _seed_worker(worker_id):
    """
    DataLoader worker initialization function (must be picklable)
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    if np is not None:
        np.random.seed(worker_seed)


def create_dataloaders(
    train_dir,
    val_dir,
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    use_memory_fs=False,
    dataset="imagenet100",
    download=False,
    input_size=None,
    seed=None,
    multiprocessing_context=None,
    persistent_workers=None,
    prefetch_factor=None,
    timeout=0.0,
):
    """
    创建训练和验证数据加载器
    
    Args:
        train_dir: 训练集目录
        val_dir: 验证集目录
        batch_size: 批次大小
        num_workers: 数据加载的worker数量
        pin_memory: 是否使用内存固定（GPU训练时设为True）
        use_memory_fs: 是否使用内存文件系统（推荐，避免并发问题）
        dataset: 数据集类型 (imagenet100/imagenet1k/cifar10/cifar100)
        download: 是否允许下载数据集（仅 CIFAR 有效）
        input_size: 输入图像大小（可选，覆盖默认值）
        seed: 随机种子（可选，启用时会为各worker固定种子）
        multiprocessing_context: DataLoader 多进程上下文（如 'spawn'）
        persistent_workers: 是否启用持久化worker（None 表示自动选择）
        prefetch_factor: worker 预取批次数（None 表示使用默认）
        timeout: DataLoader 超时（秒），0 表示不超时
    
    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    dataset_name = normalize_dataset_name(dataset)
    info = DATASET_INFO[dataset_name]
    input_size = input_size or info["input_size"]

    print("=" * 60)
    print(f"{info['label']} 数据加载器配置")
    print("=" * 60)
    print(f"  数据集类型: {dataset_name}")
    print(f"  输入大小: {input_size}x{input_size}")

    if info["type"] == "imagefolder" and (not train_dir or not val_dir):
        raise ValueError("ImageFolder 数据集需要提供 train_dir 和 val_dir")

    # 优先使用内存文件系统（仅ImageFolder类数据集）
    if use_memory_fs and info["type"] == "imagefolder":
        print("\n[0] 尝试使用内存文件系统...")
        manager = create_memory_fs_manager(
            train_dir,
            val_dir,
            use_memory_fs=True,
            shm_name=info["shm_name"]
        )

        if manager is not None:
            effective_path = manager.get_effective_path()

            # 使用内存FS路径或原始路径
            train_dir = str(effective_path / "train")
            val_dir = str(effective_path / "val")
            print("\n[0] ✓ 使用内存文件系统模式")
    elif use_memory_fs:
        print("\n[0] CIFAR 数据集已在内存中加载，忽略内存文件系统配置")

    # 数据预处理
    print("\n[1] 配置数据预处理...")

    if info["type"] == "imagefolder":
        train_transform = get_imagenet_train_transform(input_size)
        val_transform = get_imagenet_val_transform(input_size)
        print("  ✓ 训练集: RandomResizedCrop + RandomHorizontalFlip + RandomRotation + ColorJitter")
        print("  ✓ 验证集: Resize + CenterCrop")
    else:
        train_transform = get_cifar_train_transform(input_size, info["mean"], info["std"])
        val_transform = get_cifar_val_transform(input_size, info["mean"], info["std"])
        print("  ✓ 训练集: RandomCrop/ResizedCrop + RandomHorizontalFlip")
        print("  ✓ 验证集: ToTensor + Normalize")

    # 创建数据集
    print("\n[2] 加载数据集...")

    if info["type"] == "imagefolder":
        if download:
            print("  ⚠ ImageNet 不支持自动下载，忽略 --download")
        if use_memory_fs:
            print("  ✓ 使用内存文件系统模式（避免并发内存问题）")
        else:
            print("  使用标准磁盘读取模式")

        train_dataset = ImageFolderDataset(train_dir, transform=train_transform)
        val_dataset = ImageFolderDataset(val_dir, transform=val_transform)
    else:
        if download:
            print("  ✓ CIFAR 数据集允许下载（如本地不存在）")
        else:
            print("  使用本地 CIFAR 数据集")

        train_root = train_dir or val_dir
        val_root = val_dir or train_root
        if train_root is None:
            raise ValueError("CIFAR 数据集需要提供 train_dir 作为数据根目录")

        if dataset_name == "cifar10":
            train_dataset = datasets.CIFAR10(
                root=train_root,
                train=True,
                transform=train_transform,
                download=download
            )
            val_dataset = datasets.CIFAR10(
                root=val_root,
                train=False,
                transform=val_transform,
                download=download
            )
        else:
            train_dataset = datasets.CIFAR100(
                root=train_root,
                train=True,
                transform=train_transform,
                download=download
            )
            val_dataset = datasets.CIFAR100(
                root=val_root,
                train=False,
                transform=val_transform,
                download=download
            )
    
    # 打印数据集信息
    print(f"\n  训练集大小: {len(train_dataset):,} 张图片")
    print(f"  验证集大小: {len(val_dataset):,} 张图片")
    print(f"  类别数量: {len(train_dataset.classes)} 个类别")
    
    # 创建DataLoader
    print("\n[3] 创建DataLoader...")
    print(f"  批次大小: {batch_size}")
    print(f"  Worker数量: {num_workers}")
    print(f"  内存固定: {pin_memory}")
    
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = _seed_worker

    is_worker_thread = threading.current_thread() is not threading.main_thread()
    allow_workers_in_thread = os.environ.get("ALLOW_DATALOADER_WORKERS_IN_THREAD", "").lower() in ("1", "true", "yes")
    if is_worker_thread and num_workers > 0 and not allow_workers_in_thread:
        print("  ⚠ DataLoader创建于工作线程，强制 num_workers=0 以避免卡住")
        num_workers = 0
    if num_workers > 0:
        if multiprocessing_context is None and is_worker_thread and os.name != 'nt':
            multiprocessing_context = 'spawn'
            print("  ⚠ DataLoader创建于工作线程，使用 multiprocessing_context='spawn' 避免fork死锁")
        if persistent_workers is None:
            # 工作线程场景下持久化worker更容易在异常/中断时卡住，默认关闭
            persistent_workers = not is_worker_thread
        if prefetch_factor is None:
            prefetch_factor = 2
    else:
        persistent_workers = False

    print(f"  multiprocessing_context: {multiprocessing_context or 'default'}")
    print(f"  persistent_workers: {persistent_workers}")
    print(f"  prefetch_factor: {prefetch_factor if num_workers > 0 else 'N/A'}")
    print(f"  timeout: {timeout}")

    train_loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
        timeout=timeout,
    )
    val_loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
        timeout=timeout,
    )

    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = persistent_workers
        val_loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            train_loader_kwargs["prefetch_factor"] = prefetch_factor
            val_loader_kwargs["prefetch_factor"] = prefetch_factor
        if multiprocessing_context is not None:
            train_loader_kwargs["multiprocessing_context"] = multiprocessing_context
            val_loader_kwargs["multiprocessing_context"] = multiprocessing_context

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    
    print(f"\n  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    
    return train_loader, val_loader, train_dataset, val_dataset
