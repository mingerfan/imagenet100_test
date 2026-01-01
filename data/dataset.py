"""
ImageNet-100 数据加载器
支持内存文件系统加速模式（已移除内存缓存选项以避免并发问题）
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .memory_fs import create_memory_fs_manager


class ImageNet100Dataset(datasets.ImageFolder):
    """
    标准ImageFolder数据集
    使用内存文件系统加速数据加载，避免内存缓存导致的并发问题
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


def get_train_transform():
    """
    获取训练集数据增强变换
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transform():
    """
    获取验证集数据变换
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])


def create_dataloaders(
    train_dir,
    val_dir,
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    use_memory_fs=False
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
    
    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    print("=" * 60)
    print("ImageNet-100 数据加载器配置")
    print("=" * 60)
    
    # 优先使用内存文件系统
    if use_memory_fs:
        print("\n[0] 尝试使用内存文件系统...")
        manager = create_memory_fs_manager(train_dir, val_dir, use_memory_fs=True)
        
        if manager is not None:
            effective_path = manager.get_effective_path()
            
            # 使用内存FS路径或原始路径
            train_dir = str(effective_path / "train")
            val_dir = str(effective_path / "val")
            print("\n[0] ✓ 使用内存文件系统模式")
    
    # 数据预处理
    print("\n[1] 配置数据预处理...")
    
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    print("  ✓ 训练集: RandomResizedCrop + RandomHorizontalFlip + RandomRotation + ColorJitter")
    print("  ✓ 验证集: Resize + CenterCrop")
    
    # 创建数据集
    print("\n[2] 加载数据集...")
    
    if use_memory_fs:
        print("  ✓ 使用内存文件系统模式（避免并发内存问题）")
    else:
        print("  使用标准磁盘读取模式")
    
    train_dataset = ImageNet100Dataset(train_dir, transform=train_transform)
    val_dataset = ImageNet100Dataset(val_dir, transform=val_transform)
    
    # 打印数据集信息
    print(f"\n  训练集大小: {len(train_dataset):,} 张图片")
    print(f"  验证集大小: {len(val_dataset):,} 张图片")
    print(f"  类别数量: {len(train_dataset.classes)} 个类别")
    
    # 创建DataLoader
    print("\n[3] 创建DataLoader...")
    print(f"  批次大小: {batch_size}")
    print(f"  Worker数量: {num_workers}")
    print(f"  内存固定: {pin_memory}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )
    
    print(f"\n  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    
    return train_loader, val_loader, train_dataset, val_dataset
