"""
ImageNet-100 数据加载器
使用完全内存缓存优化数据加载性能
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import time
import os


class ImageNet100Dataset(datasets.ImageFolder):
    """
    带内存缓存的ImageFolder
    将所有图片预加载到内存中，大幅提升数据读取速度
    """
    def __init__(self, root, transform=None, use_cache=True):
        super().__init__(root, transform=transform)
        
        if use_cache:
            print(f"\n正在将数据集缓存到内存: {root}")
            print(f"总图片数: {len(self.samples)}")
            
            self.cache = {}
            self.targets = []
            
            # 统计信息
            total_images = len(self.samples)
            cache_start = time.time()
            
            # 预加载所有图片到内存
            for idx, (path, target) in enumerate(self.samples):
                # 加载图片
                img = Image.open(path).convert('RGB')
                self.cache[idx] = img
                self.targets.append(target)
                
                # 显示进度
                if (idx + 1) % 1000 == 0 or (idx + 1) == total_images:
                    progress = (idx + 1) / total_images * 100
                    elapsed = time.time() - cache_start
                    print(f"  进度: {idx + 1}/{total_images} ({progress:.1f}%) - 已用时间: {elapsed:.1f}s")
            
            self.targets = torch.tensor(self.targets)
            
            cache_time = time.time() - cache_start
            print(f"✓ 数据集缓存完成! 耗时: {cache_time:.2f} 秒")
            print(f"✓ 估计内存占用: ~{total_images * 0.5:.1f} MB (假设每张图片0.5MB)")
        else:
            self.cache = None
            self.targets = torch.tensor([target for _, target in self.samples])
    
    def __getitem__(self, index):
        """
        获取缓存的图片并应用transform
        """
        # 从缓存获取原始图片
        if self.cache is not None:
            img = self.cache[index]
        else:
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
    use_cache=True
):
    """
    创建训练和验证数据加载器
    
    Args:
        train_dir: 训练集目录
        val_dir: 验证集目录
        batch_size: 批次大小
        num_workers: 数据加载的worker数量
        pin_memory: 是否使用内存固定（GPU训练时设为True）
        use_cache: 是否使用内存缓存
    
    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    print("=" * 60)
    print("ImageNet-100 数据加载器配置")
    print("=" * 60)
    
    # 数据预处理
    print("\n[1] 配置数据预处理...")
    
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    print("  ✓ 训练集: RandomResizedCrop + RandomHorizontalFlip + RandomRotation + ColorJitter")
    print("  ✓ 验证集: Resize + CenterCrop")
    
    # 创建数据集
    print("\n[2] 加载数据集...")
    
    if use_cache:
        print("  使用完全内存缓存模式...")
    else:
        print("  使用标准磁盘读取模式...")
    
    train_dataset = ImageNet100Dataset(train_dir, transform=train_transform, use_cache=use_cache)
    val_dataset = ImageNet100Dataset(val_dir, transform=val_transform, use_cache=use_cache)
    
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