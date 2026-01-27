#!/usr/bin/env python3
"""
从ImageNet1K随机选择100个类，生成ImageNet100数据集
并导出标准的train.txt和val.txt标签文件
"""

import os
import random
import json
from pathlib import Path
from collections import defaultdict

def select_imagenet100_classes(imagenet1k_root, output_root, seed=42):
    """
    从ImageNet1K随机选择100个类，创建ImageNet100数据集
    
    Args:
        imagenet1k_root: ImageNet1K根目录路径
        output_root: ImageNet100输出目录路径  
        seed: 随机种子，确保可复现
    """
    
    imagenet1k_path = Path(imagenet1k_root)
    output_path = Path(output_root)
    
    # 检查源目录
    train_dir = imagenet1k_path / "train"
    val_dir = imagenet1k_path / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print(f"❌ ImageNet1K目录结构错误")
        print(f"   需要: {imagenet1k_root}/train/ 和 {imagenet1k_root}/val/")
        return False
    
    # 获取所有类（从train目录）
    all_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print(f"📁 发现 {len(all_classes)} 个类")
    
    # 随机选择100个类
    random.seed(seed)
    selected_classes = sorted(random.sample(all_classes, 100))
    
    print(f"🎲 随机选择100个类 (seed={seed})")
    print(f"   前10个类: {selected_classes[:10]}")
    
    # 创建输出目录
    output_train = output_path / "train"
    output_val = output_path / "val"
    output_train.mkdir(parents=True, exist_ok=True)
    output_val.mkdir(parents=True, exist_ok=True)
    
    # 创建类ID到索引的映射
    class_to_idx = {class_name: idx for idx, class_name in enumerate(selected_classes)}
    
    # 生成符号链接和标签文件
    train_labels = []
    val_labels = []
    
    stats = {"train_files": 0, "val_files": 0, "missing_classes": []}
    
    for class_name in selected_classes:
        class_idx = class_to_idx[class_name]
        
        # 处理训练集
        src_train_class = train_dir / class_name
        dst_train_class = output_train / class_name
        
        if src_train_class.exists():
            # 创建符号链接
            if not dst_train_class.exists():
                os.symlink(src_train_class, dst_train_class)
            
            # 收集训练文件标签
            for img_file in sorted(src_train_class.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    rel_path = f"train/{class_name}/{img_file.name}"
                    train_labels.append(f"{rel_path} {class_idx}")
                    stats["train_files"] += 1
        else:
            stats["missing_classes"].append(f"train/{class_name}")
        
        # 处理验证集
        src_val_class = val_dir / class_name
        dst_val_class = output_val / class_name
        
        if src_val_class.exists():
            # 创建符号链接
            if not dst_val_class.exists():
                os.symlink(src_val_class, dst_val_class)
            
            # 收集验证文件标签
            for img_file in sorted(src_val_class.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    rel_path = f"val/{class_name}/{img_file.name}"
                    val_labels.append(f"{rel_path} {class_idx}")
                    stats["val_files"] += 1
        else:
            stats["missing_classes"].append(f"val/{class_name}")
    
    # 写入标签文件
    with open(output_path / "train.txt", 'w') as f:
        f.write('\n'.join(train_labels))
    
    with open(output_path / "val.txt", 'w') as f:
        f.write('\n'.join(val_labels))
    
    # 保存类名映射
    class_info = {
        "selected_classes": selected_classes,
        "class_to_idx": class_to_idx,
        "seed": seed,
        "stats": stats
    }
    
    with open(output_path / "class_info.json", 'w') as f:
        json.dump(class_info, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print(f"\n✅ ImageNet100生成完成")
    print(f"📁 输出目录: {output_path}")
    print(f"🏋️  训练文件: {stats['train_files']} 张")
    print(f"✅ 验证文件: {stats['val_files']} 张")
    print(f"📄 标签文件: train.txt, val.txt")
    print(f"📋 类信息: class_info.json")
    
    if stats["missing_classes"]:
        print(f"⚠️  缺失类: {len(stats['missing_classes'])} 个")
        for missing in stats["missing_classes"]:
            print(f"   - {missing}")
    
    # 显示标签文件格式示例
    print(f"\n📝 标签文件格式示例:")
    print(f"   train.txt:")
    for line in train_labels[:3]:
        print(f"      {line}")
    print(f"   val.txt:")
    for line in val_labels[:3]:
        print(f"      {line}")
    
    return True


def load_and_verify_imagenet100(output_root):
    """验证生成的ImageNet100数据集"""
    
    output_path = Path(output_root)
    
    # 检查文件是否存在
    required_files = ["train.txt", "val.txt", "class_info.json"]
    for file in required_files:
        if not (output_path / file).exists():
            print(f"❌ 缺失文件: {file}")
            return False
    
    # 读取类信息
    with open(output_path / "class_info.json") as f:
        class_info = json.load(f)
    
    # 统计标签文件
    with open(output_path / "train.txt") as f:
        train_lines = f.readlines()
    
    with open(output_path / "val.txt") as f:
        val_lines = f.readlines()
    
    print(f"📊 数据集验证")
    print(f"   类数: {len(class_info['selected_classes'])}")
    print(f"   训练样本: {len(train_lines)}")
    print(f"   验证样本: {len(val_lines)}")
    print(f"   随机种子: {class_info['seed']}")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python select_imagenet100.py <imagenet1k_root> <output_root> [seed]")
        print("\n示例:")
        print("  python select_imagenet100.py /data/imagenet1k /data/imagenet100")
        print("  python select_imagenet100.py /data/imagenet1k /data/imagenet100 42")
        print("\n功能:")
        print("  - 从ImageNet1K随机选择100个类")
        print("  - 创建符号链接到新目录")
        print("  - 生成train.txt和val.txt标签文件")
        print("  - 保存类信息到class_info.json")
        sys.exit(1)
    
    imagenet1k_root = sys.argv[1]
    output_root = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    
    print(f"🚀 开始生成ImageNet100\n")
    
    success = select_imagenet100_classes(imagenet1k_root, output_root, seed)
    
    if success:
        print(f"\n🔍 验证生成结果")
        load_and_verify_imagenet100(output_root)
    
    sys.exit(0 if success else 1)