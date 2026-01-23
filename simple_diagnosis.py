#!/usr/bin/env python3
"""
最简化的诊断脚本 - 只检查最关键的class_to_idx问题
无任何外部依赖
"""

import os
from pathlib import Path
from torchvision import datasets, transforms


def main():
    print("="*80)
    print("最简化诊断 - 检查class_to_idx映射（最致命问题）")
    print("="*80)
    
    # 配置路径（请根据实际情况修改）
    train_dir = "/home/xuming/Documents/dataset/ImageNet_100/train"
    val_dir = "/home/xuming/Documents/dataset/ImageNet_100/val"
    
    # Windows路径
    if not os.path.exists(train_dir):
        train_dir = r"D:\dataset\ImageNet_100\train"
        val_dir = r"D:\dataset\ImageNet_100\val"
    
    print(f"\n数据集路径:")
    print(f"  训练集: {train_dir}")
    print(f"  验证集: {val_dir}")
    
    # 检查路径
    if not os.path.exists(train_dir):
        print(f"\n❌ 错误: 训练集目录不存在!")
        print(f"请修改脚本中的train_dir路径")
        print(f"\n提示: 找到你的ImageNet100数据集，然后修改:")
        print(f"  train_dir = r'你的路径\\train'")
        print(f"  val_dir = r'你的路径\\val'")
        return
    
    if not os.path.exists(val_dir):
        print(f"\n❌ 错误: 验证集目录不存在!")
        return
    
    # 简单的transform（只需要加载数据集结构）
    simple_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    print(f"\n加载数据集...")
    train_dataset = datasets.ImageFolder(train_dir, transform=simple_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=simple_transform)
    
    print(f"  ✓ 训练集: {len(train_dataset)} 张图片, {len(train_dataset.classes)} 个类别")
    print(f"  ✓ 验证集: {len(val_dataset)} 张图片, {len(val_dataset.classes)} 个类别")
    
    # ============================================================
    # 核心检查: class_to_idx是否一致
    # ============================================================
    print("\n" + "="*80)
    print("检查: 类别索引映射 (class_to_idx)")
    print("="*80)
    
    train_class_to_idx = train_dataset.class_to_idx
    val_class_to_idx = val_dataset.class_to_idx
    
    print(f"\n训练集类别数: {len(train_class_to_idx)}")
    print(f"验证集类别数: {len(val_class_to_idx)}")
    
    # 打印前5个映射
    print("\n训练集 class_to_idx (前5个):")
    for class_name, idx in sorted(train_class_to_idx.items())[:5]:
        print(f"  {class_name:20s} -> {idx}")
    
    print("\n验证集 class_to_idx (前5个):")
    for class_name, idx in sorted(val_class_to_idx.items())[:5]:
        print(f"  {class_name:20s} -> {idx}")
    
    # 完整对比
    print("\n" + "-"*80)
    if train_class_to_idx == val_class_to_idx:
        print("✓✓✓ 结果: 训练集和验证集的class_to_idx完全一致!")
        print("\n  → 类别映射不是问题所在")
        print("  → 问题可能是:")
        print("     1. 数据增强过强（已修复）")
        print("     2. 严重过拟合")
        print("     3. 模型容量或训练策略问题")
    else:
        print("❌❌❌ 严重错误: 训练集和验证集的class_to_idx不一致!")
        print("\n  → 这很可能就是导致验证准确率只有54%的根本原因!")
        print("  → 模型在训练时学到的类别0对应的特征")
        print("     在验证时可能被映射到了完全不同的类别")
        
        # 详细分析差异
        print("\n详细差异分析:")
        
        train_classes = set(train_class_to_idx.keys())
        val_classes = set(val_class_to_idx.keys())
        
        only_in_train = train_classes - val_classes
        only_in_val = val_classes - train_classes
        common_classes = train_classes & val_classes
        
        if only_in_train:
            print(f"\n  只在训练集的类别 ({len(only_in_train)} 个):")
            for cls in list(only_in_train)[:3]:
                print(f"    - {cls}")
            if len(only_in_train) > 3:
                print(f"    ... 还有 {len(only_in_train) - 3} 个")
        
        if only_in_val:
            print(f"\n  只在验证集的类别 ({len(only_in_val)} 个):")
            for cls in list(only_in_val)[:3]:
                print(f"    - {cls}")
            if len(only_in_val) > 3:
                print(f"    ... 还有 {len(only_in_val) - 3} 个")
        
        # 检查索引不匹配的类别
        mismatched = []
        for cls in common_classes:
            if train_class_to_idx[cls] != val_class_to_idx[cls]:
                mismatched.append((cls, train_class_to_idx[cls], val_class_to_idx[cls]))
        
        if mismatched:
            print(f"\n  共同类别但索引不同 ({len(mismatched)} 个):")
            for cls, train_idx, val_idx in mismatched[:5]:
                print(f"    {cls:20s}: train={train_idx:3d}, val={val_idx:3d}")
            if len(mismatched) > 5:
                print(f"    ... 还有 {len(mismatched) - 5} 个不匹配")
        
        print("\n修复建议:")
        print("  1. 确保train和val目录结构完全一致")
        print("  2. 类别文件夹名称必须完全相同")
        print("  3. 可能需要重新组织数据集")
    
    print("\n" + "="*80)
    
    # ============================================================
    # 额外检查: 验证集样本标签
    # ============================================================
    print("\n检查: 验证集前10个样本")
    print("="*80)
    
    print("\n前10个样本的标签和路径:")
    for i in range(min(10, len(val_dataset))):
        path, label = val_dataset.samples[i]
        class_name = val_dataset.classes[label]
        folder_name = Path(path).parent.name
        
        match = "✓" if folder_name == class_name else "❌"
        
        print(f"  {i:2d}. 标签={label:3d} 类名={class_name:20s} 文件夹={folder_name:20s} {match}")
    
    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "="*80)
    print("诊断完成")
    print("="*80)
    
    print("\n下一步:")
    if train_class_to_idx != val_class_to_idx:
        print("  ❌ class_to_idx不一致 - 这是最严重的问题!")
        print("  → 必须修复数据集结构后重新训练")
    else:
        print("  ✓ class_to_idx一致")
        print("  → 问题可能是数据增强或过拟合")
        print("  → 数据增强已修复，建议重新训练（--epochs 25）")


if __name__ == '__main__':
    main()
