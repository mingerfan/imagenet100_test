#!/usr/bin/env python3
"""
带模型的诊断脚本 - 检查训练集用验证transforms的真实准确率
"""

import os
import sys
import torch
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import create_dataloaders
from network_gen import create_network
from network_gen.network_config import NetworkConfig
import json


def evaluate_with_val_transforms(model, loader, device):
    """用验证transforms评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='带模型的训练问题诊断')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型检查点路径（.pth文件）')
    parser.add_argument('--config_path', type=str, default=None,
                       help='网络配置文件路径（.json文件，如果checkpoint中没有config）')
    parser.add_argument('--dataset', type=str, default='imagenet100',
                       help='数据集类型（会根据模型自动检测，通常不需要手动指定）')
    parser.add_argument('--train_dir', type=str,
                       default='/home/xuming/Documents/dataset/ImageNet_100/train',
                       help='训练集目录（请确保与模型训练时使用的数据集一致）')
    parser.add_argument('--val_dir', type=str,
                       default='/home/xuming/Documents/dataset/ImageNet_100/val',
                       help='验证集目录（请确保与模型训练时使用的数据集一致）')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    print("="*80)
    print("带模型的诊断 - 检查训练集真实准确率")
    print("="*80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 先加载checkpoint来检测类别数
    print(f"\n预加载checkpoint以检测配置...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 检测实际类别数
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if 'fc.weight' in state_dict:
        detected_num_classes = state_dict['fc.weight'].shape[0]
        print(f"  检测到模型类别数: {detected_num_classes}")
        
        # 根据类别数推断数据集
        if detected_num_classes == 1000:
            print(f"  → 这是ImageNet-1k模型 (1000类)")
            if args.dataset == 'imagenet100':
                print(f"  ⚠️  参数指定的是imagenet100，但模型是1000类")
                print(f"  → 自动切换到imagenet1k数据集")
                args.dataset = 'imagenet1k'
        elif detected_num_classes == 100:
            print(f"  → 这是ImageNet-100模型 (100类)")
            if args.dataset == 'imagenet1k':
                print(f"  ⚠️  参数指定的是imagenet1k，但模型是100类")
                print(f"  → 自动切换到imagenet100数据集")
                args.dataset = 'imagenet100'
        else:
            print(f"  → 检测到 {detected_num_classes} 类模型")
    
    # 创建数据加载器（都使用验证transforms）
    print(f"\n创建数据加载器（使用 {args.dataset} 数据集）...")
    print(f"  训练集: {args.train_dir}")
    print(f"  验证集: {args.val_dir}")
    
    from data.dataset import get_imagenet_val_transform
    from torchvision import datasets
    from torch.utils.data import DataLoader
    
    val_transform = get_imagenet_val_transform(224)
    
    train_dataset_eval = datasets.ImageFolder(args.train_dir, transform=val_transform)
    val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)
    
    print(f"  训练集样本数: {len(train_dataset_eval)}")
    print(f"  验证集样本数: {len(val_dataset)}")
    print(f"  数据集类别数: {len(train_dataset_eval.classes)}")
    
    # 验证数据集类别数与模型是否匹配
    if 'detected_num_classes' in locals():
        if detected_num_classes != len(train_dataset_eval.classes):
            print(f"\n  ❌ 错误: 模型类别数 ({detected_num_classes}) 与数据集类别数 ({len(train_dataset_eval.classes)}) 不匹配!")
            print(f"  请检查:")
            print(f"    1. 数据集路径是否正确")
            print(f"    2. 模型是否用于当前数据集")
            return
    
    train_loader_eval = DataLoader(
        train_dataset_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 加载模型
    print(f"\n加载模型配置和权重...")
    
    # 尝试从checkpoint或单独的config文件加载配置
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        print("  从checkpoint中加载配置")
    elif args.config_path:
        with open(args.config_path) as f:
            data = json.load(f)
            config_dict = data.get('config', data)
        print(f"  从配置文件加载: {args.config_path}")
    else:
        print("❌ 错误: 需要提供config_path")
        print("   checkpoint中没有config字段，请用--config_path指定配置文件")
        return
    
    # 创建模型
    config = NetworkConfig.from_dict(config_dict)
    
    # 从checkpoint中检测实际的类别数
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if 'fc.weight' in state_dict:
        actual_num_classes = state_dict['fc.weight'].shape[0]
        if actual_num_classes != config.num_classes:
            print(f"  ⚠️  检测到类别数不匹配:")
            print(f"     配置文件: {config.num_classes} 类")
            print(f"     checkpoint: {actual_num_classes} 类")
            print(f"  → 自动调整为 {actual_num_classes} 类")
            config.num_classes = actual_num_classes
    
    model = create_network(config)
    
    # 加载权重
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("  ✓ 模型加载成功")
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"  ❌ 模型加载失败: 权重形状不匹配")
            print(f"     {e}")
            print(f"\n  提示: checkpoint可能是用不同的num_classes训练的")
            return
        else:
            raise
    
    model = model.to(device)
    
    # 评估
    print("\n" + "="*80)
    print("评估: 训练集（用验证transforms）")
    print("="*80)
    train_eval_acc = evaluate_with_val_transforms(model, train_loader_eval, device)
    print(f"\n训练集评估准确率: {train_eval_acc:.2f}%")
    
    print("\n" + "="*80)
    print("评估: 验证集")
    print("="*80)
    val_acc = evaluate_with_val_transforms(model, val_loader, device)
    print(f"\n验证集准确率: {val_acc:.2f}%")
    
    # 分析
    print("\n" + "="*80)
    print("分析结果")
    print("="*80)
    
    gap = train_eval_acc - val_acc
    
    print(f"\n训练集评估准确率: {train_eval_acc:.2f}%")
    print(f"验证集准确率:     {val_acc:.2f}%")
    print(f"差距:             {gap:.2f}%")
    
    print("\n诊断:")
    
    if train_eval_acc > 85 and val_acc < 60:
        print("❌ 训练集很高但验证集很低 → 类别映射问题或数据集标签错误")
        print("   建议: 运行 python quick_diagnosis.py 检查class_to_idx")
    elif train_eval_acc < 70:
        print("⚠️  训练集评估准确率偏低 → 报告的93%训练准确率可能不准确")
        print("   可能原因:")
        print("   - 使用了mixup/cutmix但统计方式有问题")
        print("   - 模型实际学习效果不好")
    elif gap > 20:
        print("⚠️  训练验证差距过大 → 严重过拟合")
        print("   建议:")
        print("   - 增强正则化（weight decay, dropout）")
        print("   - 减少训练epoch")
        print("   - 使用更强的数据增强")
    elif gap < 10:
        print("✓ 训练验证差距正常")
        if val_acc < 70:
            print("  但验证准确率偏低，可能是:")
            print("  - 网络结构能力不足")
            print("  - 训练不充分")
            print("  - 超参数需要调整")
    else:
        print("⚠️  中等程度过拟合")
        print("   可以通过调整正则化和数据增强改善")


if __name__ == '__main__':
    main()
