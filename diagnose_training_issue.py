#!/usr/bin/env python3
"""
训练问题诊断脚本
快速检查训练准确率93%但验证准确率54%的根本原因
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import create_dataloaders, get_dataset_info, normalize_dataset_name


def check_1_eval_mode_and_transforms(train_loader, val_loader):
    """检查1: 验证transforms是否正确"""
    print("\n" + "="*80)
    print("检查 1: 验证模式和Transforms")
    print("="*80)
    
    # 检查transforms
    print("\n训练集 transforms:")
    print(train_loader.dataset.transform)
    
    print("\n验证集 transforms:")
    print(val_loader.dataset.transform)
    
    # 获取一个batch看看实际数据
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    train_images, train_labels = train_batch
    val_images, val_labels = val_batch
    
    print(f"\n训练batch统计:")
    print(f"  Shape: {train_images.shape}")
    print(f"  Min: {train_images.min().item():.4f}, Max: {train_images.max().item():.4f}")
    print(f"  Mean: {train_images.mean().item():.4f}, Std: {train_images.std().item():.4f}")
    
    print(f"\n验证batch统计:")
    print(f"  Shape: {val_images.shape}")
    print(f"  Min: {val_images.min().item():.4f}, Max: {val_images.max().item():.4f}")
    print(f"  Mean: {val_images.mean().item():.4f}, Std: {val_images.std().item():.4f}")
    
    # 检查normalize参数
    train_transform_str = str(train_loader.dataset.transform)
    val_transform_str = str(val_loader.dataset.transform)
    
    if "Normalize" in train_transform_str and "Normalize" in val_transform_str:
        print("\n✓ 训练集和验证集都使用了Normalize")
    else:
        print("\n❌ 警告: Normalize可能缺失!")
        print(f"  训练集有Normalize: {'Normalize' in train_transform_str}")
        print(f"  验证集有Normalize: {'Normalize' in val_transform_str}")


def check_2_train_accuracy_sanity(model, train_loader, val_loader, device):
    """检查2: 训练集准确率是否可信 - 在训练集上用验证transforms评估"""
    print("\n" + "="*80)
    print("检查 2: 训练集准确率可信度检查")
    print("="*80)
    print("在训练集上使用验证transforms（无数据增强）评估准确率...")
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Train Eval"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    train_eval_acc = 100. * correct / total
    print(f"\n✓ 训练集评估准确率（用验证transforms）: {train_eval_acc:.2f}%")
    print(f"\n分析:")
    if train_eval_acc > 85:
        print(f"  - 训练集评估准确率 {train_eval_acc:.2f}% 很高")
        print(f"  - 这意味着报告的93%训练准确率是可信的")
        print(f"  - 问题更可能是: 类别映射不一致、数据集标签错误")
    elif train_eval_acc < 70:
        print(f"  - 训练集评估准确率 {train_eval_acc:.2f}% 偏低")
        print(f"  - 报告的93%可能是统计口径问题（mixup/cutmix等）")
        print(f"  - 模型实际学习效果可能不如预期")
    else:
        print(f"  - 训练集评估准确率 {train_eval_acc:.2f}% 中等")
        print(f"  - 需要进一步检查其他问题")


def check_3_class_index_mapping(train_loader, val_loader):
    """检查3: 类别索引映射是否一致（最致命的问题）"""
    print("\n" + "="*80)
    print("检查 3: 类别索引映射一致性")
    print("="*80)
    
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    # 检查是否有class_to_idx属性
    if not hasattr(train_dataset, 'class_to_idx'):
        print("⚠ 警告: 训练集没有class_to_idx属性")
        return
    
    if not hasattr(val_dataset, 'class_to_idx'):
        print("⚠ 警告: 验证集没有class_to_idx属性")
        return
    
    train_class_to_idx = train_dataset.class_to_idx
    val_class_to_idx = val_dataset.class_to_idx
    
    print(f"\n训练集类别数: {len(train_class_to_idx)}")
    print(f"验证集类别数: {len(val_class_to_idx)}")
    
    # 打印前10个类别映射
    print("\n训练集 class_to_idx (前10个):")
    for i, (class_name, idx) in enumerate(sorted(train_class_to_idx.items())[:10]):
        print(f"  {class_name}: {idx}")
    
    print("\n验证集 class_to_idx (前10个):")
    for i, (class_name, idx) in enumerate(sorted(val_class_to_idx.items())[:10]):
        print(f"  {class_name}: {idx}")
    
    # 检查是否一致
    if train_class_to_idx == val_class_to_idx:
        print("\n✓ 训练集和验证集的class_to_idx完全一致")
    else:
        print("\n❌ 严重警告: 训练集和验证集的class_to_idx不一致!")
        
        # 找出差异
        train_classes = set(train_class_to_idx.keys())
        val_classes = set(val_class_to_idx.keys())
        
        only_in_train = train_classes - val_classes
        only_in_val = val_classes - train_classes
        
        if only_in_train:
            print(f"\n  只在训练集的类别 ({len(only_in_train)}): {list(only_in_train)[:5]}")
        if only_in_val:
            print(f"  只在验证集的类别 ({len(only_in_val)}): {list(only_in_val)[:5]}")
        
        # 检查共同类别的索引是否一致
        common_classes = train_classes & val_classes
        mismatched = []
        for class_name in common_classes:
            if train_class_to_idx[class_name] != val_class_to_idx[class_name]:
                mismatched.append(class_name)
        
        if mismatched:
            print(f"\n  索引不匹配的类别 ({len(mismatched)}):")
            for class_name in mismatched[:5]:
                print(f"    {class_name}: train={train_class_to_idx[class_name]}, "
                      f"val={val_class_to_idx[class_name]}")
        
        print("\n  ⚠️ 这很可能是导致验证准确率低的根本原因!")


def check_4_visualize_samples(val_loader, num_samples=16):
    """检查4: 可视化验证集样本，检查标签是否正确"""
    print("\n" + "="*80)
    print("检查 4: 验证集样本可视化")
    print("="*80)
    
    val_dataset = val_loader.dataset
    
    # 获取类别名称
    if hasattr(val_dataset, 'classes'):
        classes = val_dataset.classes
    else:
        classes = [f"Class_{i}" for i in range(100)]
    
    print(f"\n获取前{num_samples}个验证样本...")
    
    # 获取样本
    images_list = []
    labels_list = []
    paths_list = []
    
    for i in range(min(num_samples, len(val_dataset))):
        img, label = val_dataset[i]
        images_list.append(img)
        labels_list.append(label)
        
        # 尝试获取路径
        if hasattr(val_dataset, 'samples'):
            path, _ = val_dataset.samples[i]
            paths_list.append(path)
    
    # 打印信息
    print("\n前16个样本:")
    for i in range(len(labels_list)):
        label_idx = labels_list[i] if isinstance(labels_list[i], int) else labels_list[i].item()
        class_name = classes[label_idx] if label_idx < len(classes) else f"Unknown_{label_idx}"
        
        print(f"  样本 {i:2d}: 标签={label_idx:3d}, 类名={class_name}")
        if i < len(paths_list):
            print(f"           路径={Path(paths_list[i]).name}")
    
    # 统计标签分布
    label_counts = {}
    for label in labels_list:
        label_idx = label if isinstance(label, int) else label.item()
        label_counts[label_idx] = label_counts.get(label_idx, 0) + 1
    
    print(f"\n前{num_samples}个样本的标签分布:")
    for label_idx, count in sorted(label_counts.items()):
        print(f"  标签 {label_idx}: {count} 个样本")
    
    print("\n提示: 请检查类名和文件路径是否匹配")


def check_5_label_distribution(train_loader, val_loader):
    """检查5: 标签分布统计"""
    print("\n" + "="*80)
    print("检查 5: 标签分布统计")
    print("="*80)
    
    print("\n统计训练集标签分布...")
    train_labels = []
    for _, labels in tqdm(train_loader, desc="Train labels"):
        train_labels.extend(labels.tolist())
    
    print("\n统计验证集标签分布...")
    val_labels = []
    for _, labels in tqdm(val_loader, desc="Val labels"):
        val_labels.extend(labels.tolist())
    
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    
    print(f"\n训练集:")
    print(f"  样本数: {len(train_labels)}")
    print(f"  标签范围: [{train_labels.min()}, {train_labels.max()}]")
    print(f"  唯一标签数: {len(np.unique(train_labels))}")
    
    print(f"\n验证集:")
    print(f"  样本数: {len(val_labels)}")
    print(f"  标签范围: [{val_labels.min()}, {val_labels.max()}]")
    print(f"  唯一标签数: {len(np.unique(val_labels))}")
    
    # 检查标签范围是否合理
    train_unique = set(np.unique(train_labels).tolist())
    val_unique = set(np.unique(val_labels).tolist())
    
    only_in_train = train_unique - val_unique
    only_in_val = val_unique - train_unique
    
    if only_in_train:
        print(f"\n⚠ 只在训练集出现的标签 ({len(only_in_train)}): {sorted(only_in_train)[:10]}")
    if only_in_val:
        print(f"⚠ 只在验证集出现的标签 ({len(only_in_val)}): {sorted(only_in_val)[:10]}")
    
    if not only_in_train and not only_in_val:
        print("\n✓ 训练集和验证集的标签集合完全一致")


def main():
    print("="*80)
    print("训练问题诊断脚本")
    print("="*80)
    print("\n将执行以下检查:")
    print("1. 验证模式和Transforms")
    print("2. 训练集准确率可信度（用验证transforms）")
    print("3. 类别索引映射一致性 ⚠️ 最关键")
    print("4. 验证集样本可视化")
    print("5. 标签分布统计")
    
    # 配置
    dataset = "imagenet100"
    train_dir = "/root/autodl-tmp/imagenet/train"
    val_dir = "/root/autodl-tmp/imagenet/val"
    batch_size = 64
    num_workers = 4
    
    print(f"\n数据集配置:")
    print(f"  Dataset: {dataset}")
    print(f"  Train dir: {train_dir}")
    print(f"  Val dir: {val_dir}")
    
    # 创建数据加载器（使用验证transforms）
    print("\n正在创建数据加载器...")
    
    # 为训练集也使用验证transforms来做准确率检查
    from data.dataset import get_imagenet_val_transform
    from torchvision import datasets
    
    val_transform = get_imagenet_val_transform(224)
    
    train_dataset_eval = datasets.ImageFolder(train_dir, transform=val_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    from torch.utils.data import DataLoader
    train_loader_eval = DataLoader(
        train_dataset_eval,
        batch_size=batch_size,
        shuffle=False,
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
    
    # 也创建正常的训练loader用于对比
    train_loader, _, _, _ = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset=dataset,
        use_memory_fs=False
    )
    
    # 执行检查
    check_1_eval_mode_and_transforms(train_loader, val_loader)
    
    check_3_class_index_mapping(train_loader, val_loader)
    
    check_4_visualize_samples(val_loader)
    
    check_5_label_distribution(train_loader, val_loader)
    
    # 如果用户提供模型路径，可以执行检查2
    print("\n" + "="*80)
    print("检查 2: 训练集准确率评估")
    print("="*80)
    print("\n要执行此检查，需要提供训练好的模型路径")
    print("用法示例:")
    print("  python diagnose_training_issue.py --model_path nas_results/trained_models/best/model_xxx/best_model.pth")
    
    print("\n" + "="*80)
    print("诊断完成")
    print("="*80)
    print("\n请重点关注:")
    print("1. 检查3的结果 - class_to_idx是否一致")
    print("2. 检查4的结果 - 样本标签是否正确")
    print("3. 检查5的结果 - 标签范围和分布是否合理")


if __name__ == '__main__':
    main()
