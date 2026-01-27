#!/usr/bin/env python3
"""
简单版本：根据类名列表从ImageNet1K构建ImageNet100
"""

import os
from pathlib import Path

def build_from_class_list(class_list_file, imagenet1k_root, output_root):
    """根据类名列表构建ImageNet100"""
    
    # 读取类名列表
    with open(class_list_file, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    print(f"📋 读取类名列表: {len(class_names)} 个类")
    
    imagenet1k_path = Path(imagenet1k_root)
    output_path = Path(output_root)
    
    # 创建输出目录
    output_train = output_path / "train"
    output_val = output_path / "val"
    output_train.mkdir(parents=True, exist_ok=True)
    output_val.mkdir(parents=True, exist_ok=True)
    
    train_count = 0
    val_count = 0
    missing = []
    
    print(f"🔗 创建符号链接...")
    
    for class_name in class_names:
        # 训练集
        src_train = imagenet1k_path / "train" / class_name
        dst_train = output_train / class_name
        
        if src_train.exists():
            if not dst_train.exists():
                os.symlink(src_train, dst_train)
            train_count += 1
        else:
            missing.append(f"train/{class_name}")
        
        # 验证集
        src_val = imagenet1k_path / "val" / class_name
        dst_val = output_val / class_name
        
        if src_val.exists():
            if not dst_val.exists():
                os.symlink(src_val, dst_val)
            val_count += 1
        else:
            missing.append(f"val/{class_name}")
    
    print(f"✅ 构建完成")
    print(f"   训练集: {train_count} 个类")
    print(f"   验证集: {val_count} 个类")
    
    if missing:
        print(f"⚠️  缺失: {len(missing)} 个")
        for m in missing[:5]:
            print(f"     {m}")
    
    return len(missing) == 0

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("用法: python build_simple.py <class_list.txt> <imagenet1k_root> <output_root>")
        print("示例: python build_simple.py selected_classes.txt /data/imagenet1k /data/imagenet100")
        sys.exit(1)
    
    class_list_file = sys.argv[1]
    imagenet1k_root = sys.argv[2] 
    output_root = sys.argv[3]
    
    build_from_class_list(class_list_file, imagenet1k_root, output_root)