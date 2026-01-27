#!/usr/bin/env python3
"""
根据train.txt和val.txt标签文件从ImageNet1K构建ImageNet100
"""

import os
import json
from pathlib import Path
from collections import defaultdict

def build_imagenet100_from_labels(train_txt, val_txt, imagenet1k_root, output_root):
    """
    根据标签文件从ImageNet1K构建ImageNet100
    
    Args:
        train_txt: train.txt标签文件路径
        val_txt: val.txt标签文件路径  
        imagenet1k_root: ImageNet1K根目录路径
        output_root: ImageNet100输出目录路径
    """
    
    train_txt_path = Path(train_txt)
    val_txt_path = Path(val_txt)
    imagenet1k_path = Path(imagenet1k_root)
    output_path = Path(output_root)
    
    # 检查输入文件
    if not train_txt_path.exists():
        print(f"❌ train.txt不存在: {train_txt}")
        return False
        
    if not val_txt_path.exists():
        print(f"❌ val.txt不存在: {val_txt}")
        return False
    
    # 检查源目录
    train_dir = imagenet1k_path / "train"
    val_dir = imagenet1k_path / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print(f"❌ ImageNet1K目录结构错误")
        print(f"   需要: {imagenet1k_root}/train/ 和 {imagenet1k_root}/val/")
        return False
    
    print(f"📋 解析标签文件...")
    
    # 解析train.txt获取所需的类
    train_classes = set()
    with open(train_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # 格式: train/n01440764/filename.jpg 0
                path_part = line.split()[0]
                # 提取类名 (train/n01440764/filename.jpg -> n01440764)
                parts = path_part.split('/')
                if len(parts) >= 2:
                    class_name = parts[1]  # train/CLASS/file
                    train_classes.add(class_name)
    
    # 解析val.txt获取所需的类
    val_classes = set()
    with open(val_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # 格式: val/n01440764/filename.jpg 0
                path_part = line.split()[0]
                # 提取类名
                parts = path_part.split('/')
                if len(parts) >= 2:
                    class_name = parts[1]  # val/CLASS/file
                    val_classes.add(class_name)
    
    # 合并所有需要的类
    all_needed_classes = sorted(train_classes | val_classes)
    
    print(f"📁 发现需要的类: {len(all_needed_classes)} 个")
    print(f"   训练集类: {len(train_classes)} 个")
    print(f"   验证集类: {len(val_classes)} 个")
    print(f"   前10个类: {all_needed_classes[:10]}")
    
    # 创建输出目录
    output_train = output_path / "train"
    output_val = output_path / "val"
    output_train.mkdir(parents=True, exist_ok=True)
    output_val.mkdir(parents=True, exist_ok=True)
    
    # 复制标签文件到输出目录
    import shutil
    shutil.copy2(train_txt_path, output_path / "train.txt")
    shutil.copy2(val_txt_path, output_path / "val.txt")
    
    stats = {
        "train_classes": 0,
        "val_classes": 0,
        "train_files": 0,
        "val_files": 0,
        "missing_classes": []
    }
    
    print(f"\n🔗 开始创建符号链接...")
    
    # 为每个类创建符号链接
    for class_name in all_needed_classes:
        
        # 处理训练集
        if class_name in train_classes:
            src_train_class = train_dir / class_name
            dst_train_class = output_train / class_name
            
            if src_train_class.exists():
                # 创建符号链接
                if not dst_train_class.exists():
                    os.symlink(src_train_class, dst_train_class)
                    print(f"   ✓ 训练集: {class_name}")
                
                # 统计文件数
                file_count = len([f for f in src_train_class.iterdir() if f.is_file()])
                stats["train_files"] += file_count
                stats["train_classes"] += 1
            else:
                stats["missing_classes"].append(f"train/{class_name}")
                print(f"   ⚠️  训练集缺失: {class_name}")
        
        # 处理验证集
        if class_name in val_classes:
            src_val_class = val_dir / class_name
            dst_val_class = output_val / class_name
            
            if src_val_class.exists():
                # 创建符号链接
                if not dst_val_class.exists():
                    os.symlink(src_val_class, dst_val_class)
                    print(f"   ✓ 验证集: {class_name}")
                
                # 统计文件数
                file_count = len([f for f in src_val_class.iterdir() if f.is_file()])
                stats["val_files"] += file_count
                stats["val_classes"] += 1
            else:
                stats["missing_classes"].append(f"val/{class_name}")
                print(f"   ⚠️  验证集缺失: {class_name}")
    
    # 保存构建信息
    build_info = {
        "source": {
            "train_txt": str(train_txt_path),
            "val_txt": str(val_txt_path),
            "imagenet1k_root": str(imagenet1k_path)
        },
        "classes": all_needed_classes,
        "stats": stats
    }
    
    with open(output_path / "build_info.json", 'w') as f:
        json.dump(build_info, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print(f"\n✅ ImageNet100构建完成")
    print(f"📁 输出目录: {output_path}")
    print(f"🏋️  训练集类: {stats['train_classes']} 个，{stats['train_files']} 张图片")
    print(f"✅ 验证集类: {stats['val_classes']} 个，{stats['val_files']} 张图片")
    print(f"📄 标签文件: train.txt, val.txt")
    print(f"📋 构建信息: build_info.json")
    
    if stats["missing_classes"]:
        print(f"\n⚠️  缺失类: {len(stats['missing_classes'])} 个")
        for missing in stats["missing_classes"][:10]:
            print(f"   - {missing}")
        if len(stats["missing_classes"]) > 10:
            print(f"   ... 还有 {len(stats['missing_classes']) - 10} 个")
    
    return len(stats["missing_classes"]) == 0


def verify_imagenet100_build(output_root):
    """验证构建的ImageNet100数据集"""
    
    output_path = Path(output_root)
    
    # 检查必要文件
    required_files = ["train.txt", "val.txt", "build_info.json"]
    for file in required_files:
        if not (output_path / file).exists():
            print(f"❌ 缺失文件: {file}")
            return False
    
    # 读取构建信息
    with open(output_path / "build_info.json") as f:
        build_info = json.load(f)
    
    # 验证目录结构
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    
    if train_dir.exists():
        train_classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    else:
        train_classes = []
    
    if val_dir.exists():
        val_classes = [d.name for d in val_dir.iterdir() if d.is_dir()]
    else:
        val_classes = []
    
    # 统计实际文件数
    actual_train_files = 0
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            actual_train_files += len([f for f in class_dir.iterdir() if f.is_file()])
    
    actual_val_files = 0
    for class_dir in val_dir.iterdir():
        if class_dir.is_dir():
            actual_val_files += len([f for f in class_dir.iterdir() if f.is_file()])
    
    print(f"📊 数据集验证")
    print(f"   训练集: {len(train_classes)} 个类，{actual_train_files} 张图片")
    print(f"   验证集: {len(val_classes)} 个类，{actual_val_files} 张图片")
    print(f"   标签文件: 存在")
    
    # 与预期对比
    expected_stats = build_info.get("stats", {})
    if expected_stats:
        print(f"\n📋 与预期对比:")
        print(f"   训练集文件: {actual_train_files} / {expected_stats.get('train_files', 0)}")
        print(f"   验证集文件: {actual_val_files} / {expected_stats.get('val_files', 0)}")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("用法: python build_from_labels.py <train.txt> <val.txt> <imagenet1k_root> <output_root>")
        print("\n示例:")
        print("  python build_from_labels.py train.txt val.txt /data/imagenet1k /data/imagenet100")
        print("\n功能:")
        print("  - 从train.txt和val.txt解析需要的类")
        print("  - 从ImageNet1K创建符号链接")
        print("  - 复制标签文件到输出目录")
        print("  - 生成build_info.json记录构建过程")
        sys.exit(1)
    
    train_txt = sys.argv[1]
    val_txt = sys.argv[2]
    imagenet1k_root = sys.argv[3]
    output_root = sys.argv[4]
    
    print(f"🚀 开始从标签文件构建ImageNet100\n")
    
    success = build_imagenet100_from_labels(train_txt, val_txt, imagenet1k_root, output_root)
    
    if success:
        print(f"\n🔍 验证构建结果")
        verify_imagenet100_build(output_root)
    
    sys.exit(0 if success else 1)