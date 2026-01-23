#!/usr/bin/env python3
"""
快速诊断脚本 - 10分钟内检查3个最关键问题
"""

import os
import sys
import torch
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 不导入create_dataloaders以避免memory_fs问题（Windows不支持fcntl）


def quick_check():
    """执行GPT建议的3个最关键检查"""
    
    print("="*80)
    print("快速诊断 - 检查3个最致命的问题")
    print("="*80)
    
    # 配置
    dataset = "imagenet100"
    train_dir = "/root/autodl-tmp/imagenet/train"
    val_dir = "/root/autodl-tmp/imagenet/val"
    batch_size = 32
    num_workers = 4
    
    # Windows路径
    if not os.path.exists(train_dir):
        train_dir = r"D:\dataset\ImageNet_100\train"
        val_dir = r"D:\dataset\ImageNet_100\val"
    
    print(f"\n加载数据集...")
    print(f"  训练集: {train_dir}")
    print(f"  验证集: {val_dir}")
    
    if not os.path.exists(train_dir):
        print(f"\n❌ 训练集目录不存在: {train_dir}")
        print("请修改quick_diagnosis.py中的路径配置")
        return
    
    if not os.path.exists(val_dir):
        print(f"\n❌ 验证集目录不存在: {val_dir}")
        print("请修改quick_diagnosis.py中的路径配置")
        return
    
    # ImageNet标准transforms
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # ============================================================
    # 检查1: 验证前16张图和标签（看是否明显错标）
    # ============================================================
    print("\n" + "="*80)
    print("检查 1: 验证集前16个样本的标签")
    print("="*80)
    
    print("\n前16个样本:")
    for i in range(min(16, len(val_dataset))):
        img, label = val_dataset[i]
        path, _ = val_dataset.samples[i]
        
        # 获取类名
        class_name = val_dataset.classes[label]
        
        # 从路径中提取文件夹名（应该是类名）
        folder_name = Path(path).parent.name
        
        match = "✓" if folder_name == class_name else "❌"
        
        print(f"  {i:2d}. 标签={label:3d} 类名={class_name:20s} 文件夹={folder_name:20s} {match}")
        if i < 3:
            print(f"      路径: {path}")
    
    # ============================================================
    # 检查2: train/val的class_to_idx是否一致（最致命）
    # ============================================================
    print("\n" + "="*80)
    print("检查 2: 类别索引映射一致性 ⚠️ 最关键")
    print("="*80)
    
    train_class_to_idx = train_dataset.class_to_idx
    val_class_to_idx = val_dataset.class_to_idx
    
    print(f"\n训练集类别数: {len(train_class_to_idx)}")
    print(f"验证集类别数: {len(val_class_to_idx)}")
    
    print("\n训练集 class_to_idx (前5个):")
    for i, (class_name, idx) in enumerate(sorted(train_class_to_idx.items())[:5]):
        print(f"  {class_name}: {idx}")
    
    print("\n验证集 class_to_idx (前5个):")
    for i, (class_name, idx) in enumerate(sorted(val_class_to_idx.items())[:5]):
        print(f"  {class_name}: {idx}")
    
    # 完整对比
    if train_class_to_idx == val_class_to_idx:
        print("\n✓✓✓ 训练集和验证集的class_to_idx完全一致!")
        print("    这个不是问题所在")
    else:
        print("\n❌❌❌ 严重错误: 训练集和验证集的class_to_idx不一致!")
        print("    这很可能是导致验证准确率低的根本原因!")
        
        # 详细对比
        train_classes = set(train_class_to_idx.keys())
        val_classes = set(val_class_to_idx.keys())
        
        only_in_train = train_classes - val_classes
        only_in_val = val_classes - train_classes
        
        if only_in_train:
            print(f"\n  只在训练集的类别 ({len(only_in_train)}): {list(only_in_train)[:3]}")
        if only_in_val:
            print(f"  只在验证集的类别 ({len(only_in_val)}): {list(only_in_val)[:3]}")
        
        # 检查索引不匹配
        common = train_classes & val_classes
        mismatched = []
        for cls in common:
            if train_class_to_idx[cls] != val_class_to_idx[cls]:
                mismatched.append((cls, train_class_to_idx[cls], val_class_to_idx[cls]))
        
        if mismatched:
            print(f"\n  索引不匹配的类别 ({len(mismatched)}):")
            for cls, train_idx, val_idx in mismatched[:5]:
                print(f"    {cls}: train={train_idx}, val={val_idx}")
    
    # ============================================================
    # 检查3: 对训练集用验证transforms做评估（确认93%是否真实）
    # ============================================================
    print("\n" + "="*80)
    print("检查 3: 训练集用验证transforms的准确率")
    print("="*80)
    print("\n⚠️ 此检查需要加载训练好的模型")
    print("如果你有模型检查点，可以运行:")
    print("  python diagnose_with_model.py --model_path <path_to_checkpoint>")
    
    # ============================================================
    # 额外检查: 数据归一化参数
    # ============================================================
    print("\n" + "="*80)
    print("额外检查: 数据归一化参数")
    print("="*80)
    
    print("\n训练集 transforms:")
    print(train_dataset.transform)
    
    print("\n验证集 transforms:")
    print(val_dataset.transform)
    
    # 检查实际数据范围
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    train_images, _ = train_batch
    val_images, _ = val_batch
    
    print(f"\n训练batch:")
    print(f"  Min: {train_images.min().item():.4f}, Max: {train_images.max().item():.4f}")
    print(f"  Mean: {train_images.mean().item():.4f}, Std: {train_images.std().item():.4f}")
    
    print(f"\n验证batch:")
    print(f"  Min: {val_images.min().item():.4f}, Max: {val_images.max().item():.4f}")
    print(f"  Mean: {val_images.mean().item():.4f}, Std: {val_images.std().item():.4f}")
    
    print("\n期望值（ImageNet标准）:")
    print("  Min: ~-2.x, Max: ~2.x")
    print("  Mean: ~0.0, Std: ~1.0")
    
    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "="*80)
    print("诊断完成")
    print("="*80)
    
    print("\n下一步:")
    print("1. 如果class_to_idx不一致 → 这是最致命的问题，必须修复")
    print("2. 如果样本标签明显错误 → 检查数据集是否正确准备")
    print("3. 如果归一化参数异常 → 检查transforms配置")
    print("4. 运行带模型的诊断来验证训练准确率是否真实")


if __name__ == '__main__':
    quick_check()
