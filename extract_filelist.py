#!/usr/bin/env python3
"""
从ImageNet100中提取文件清单
在源服务器上运行：python extract_filelist.py /path/to/imagenet100
输出可粘贴到其他服务器的清单文件
"""

import os
import sys
from pathlib import Path
import json

def extract_filelist(imagenet100_root, output_file="imagenet100_filelist.txt"):
    """
    提取ImageNet100的完整文件清单
    格式：每行一个相对路径
    支持两种结构：
    1. ImageNet_100/ -> n01234567/ -> images.jpg
    2. ImageNet_100/ -> train/val -> n01234567/ -> images.jpg
    """
    imagenet100_root = Path(imagenet100_root)
    
    if not imagenet100_root.exists():
        print(f"❌ 路径不存在: {imagenet100_root}")
        return False
    
    files = []
    classes = []
    
    # 检查第一层是否是 train/val 结构
    try:
        entries = list(imagenet100_root.iterdir())
    except Exception as e:
        print(f"❌ 无法读取目录: {e}")
        return False
    
    # 判断是否为 train/val 结构
    first_level_dirs = [e for e in entries if os.path.isdir(str(e))]
    first_level_names = {d.name for d in first_level_dirs}
    
    is_train_val_structure = first_level_names == {"train", "val"} or \
                             first_level_names == {"train"} or \
                             first_level_names == {"val"}
    
    if is_train_val_structure:
        # 遍历 train/val 子目录
        print(f"📁 检测到 train/val 结构，正在遍历...")
        for subset_dir in sorted(first_level_dirs):
            try:
                class_dirs = list(subset_dir.iterdir())
            except Exception as e:
                print(f"⚠  无法读取子集目录 {subset_dir.name}: {e}")
                continue
            
            for class_dir in sorted(class_dirs):
                if not os.path.isdir(str(class_dir)):
                    continue
                
                class_name = class_dir.name
                if class_name not in classes:
                    classes.append(class_name)
                
                try:
                    img_files = list(class_dir.iterdir())
                except Exception as e:
                    print(f"⚠  无法读取类目录 {subset_dir.name}/{class_name}: {e}")
                    continue
                
                for img_file in sorted(img_files):
                    if os.path.isfile(str(img_file)):
                        rel_path = f"{subset_dir.name}/{class_name}/{img_file.name}"
                        files.append(rel_path)
    else:
        # 直接遍历第一层作为类目录
        print(f"📁 检测到直接结构，正在遍历...")
        for class_dir in sorted(entries):
            if not os.path.isdir(str(class_dir)):
                continue
            
            class_name = class_dir.name
            classes.append(class_name)
            
            try:
                img_files = list(class_dir.iterdir())
            except Exception as e:
                print(f"⚠  无法读取类目录 {class_name}: {e}")
                continue
            
            for img_file in sorted(img_files):
                if os.path.isfile(str(img_file)):
                    rel_path = f"{class_name}/{img_file.name}"
                    files.append(rel_path)
    
    # 保存为可复制的文本格式
    with open(output_file, 'w', encoding='utf-8') as f:
        # 头部信息
        f.write(f"# ImageNet100 文件清单\n")
        f.write(f"# 总类数: {len(classes)}\n")
        f.write(f"# 总文件数: {len(files)}\n")
        f.write(f"# 格式: 类名/文件名\n")
        f.write(f"# ===== 类目录列表 =====\n")
        
        for class_name in classes:
            f.write(f"# {class_name}\n")
        
        f.write(f"# ===== 完整文件列表 =====\n")
        for file_path in files:
            f.write(f"{file_path}\n")
    
    print(f"✓ 文件清单已提取")
    print(f"  总类数: {len(classes)}")
    print(f"  总文件数: {len(files)}")
    print(f"  输出文件: {output_file}")
    print(f"\n🔹 下一步: 复制 {output_file} 的内容到剪贴板，粘贴到目标服务器")
    
    # 同时保存JSON格式用于编程访问
    json_file = output_file.replace('.txt', '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "classes": classes,
            "files": files,
            "total_classes": len(classes),
            "total_files": len(files)
        }, f, indent=2)
    
    print(f"  JSON版本: {json_file}")
    
    return True


def extract_filelist_compact(imagenet100_root, output_file="imagenet100_filelist_compact.txt"):
    """
    提取紧凑格式的文件清单
    支持两种结构：
    1. ImageNet_100/ -> n01234567/ -> images.jpg
    2. ImageNet_100/ -> train/val -> n01234567/ -> images.jpg
    """
    imagenet100_root = Path(imagenet100_root)
    
    if not imagenet100_root.exists():
        print(f"❌ 路径不存在: {imagenet100_root}")
        return False
    
    class_info = {}
    
    try:
        entries = list(imagenet100_root.iterdir())
    except Exception as e:
        print(f"❌ 无法读取目录: {e}")
        return False
    
    # 判断是否为 train/val 结构
    first_level_dirs = [e for e in entries if os.path.isdir(str(e))]
    first_level_names = {d.name for d in first_level_dirs}
    
    is_train_val_structure = first_level_names == {"train", "val"} or \
                             first_level_names == {"train"} or \
                             first_level_names == {"val"}
    
    if is_train_val_structure:
        # 遍历 train/val 子目录
        print(f"📁 检测到 train/val 结构，正在遍历...")
        for subset_dir in sorted(first_level_dirs):
            try:
                class_dirs = list(subset_dir.iterdir())
            except Exception as e:
                print(f"⚠  无法读取子集目录 {subset_dir.name}: {e}")
                continue
            
            for class_dir in sorted(class_dirs):
                if not os.path.isdir(str(class_dir)):
                    continue
                
                class_name = class_dir.name
                
                try:
                    img_files = [f.name for f in class_dir.iterdir() if os.path.isfile(str(f))]
                    if img_files:
                        if class_name not in class_info:
                            class_info[class_name] = []
                        class_info[class_name].extend(sorted(img_files))
                except Exception as e:
                    print(f"⚠  无法读取类目录 {subset_dir.name}/{class_name}: {e}")
                    continue
    else:
        # 直接遍历第一层作为类目录
        print(f"📁 检测到直接结构，正在遍历...")
        for class_dir in sorted(first_level_dirs):
            class_name = class_dir.name
            
            try:
                img_files = [f.name for f in class_dir.iterdir() if os.path.isfile(str(f))]
                if img_files:
                    class_info[class_name] = sorted(img_files)
            except Exception as e:
                print(f"⚠  无法读取类目录 {class_name}: {e}")
                continue
    
    # 保存为JSON（易于粘贴和处理）
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(class_info, f, indent=2, ensure_ascii=False)
    
    total_files = sum(len(files) for files in class_info.values())
    
    print(f"✓ 紧凑清单已生成")
    print(f"  总类数: {len(class_info)}")
    print(f"  总文件数: {total_files}")
    print(f"  输出文件: {output_file}")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python extract_filelist.py /path/to/imagenet100")
        print("\n示例:")
        print("  python extract_filelist.py /mnt/data/imagenet100")
        sys.exit(1)
    
    imagenet100_root = sys.argv[1]
    
    print(f"📁 正在扫描: {imagenet100_root}\n")
    
    extract_filelist(imagenet100_root)
    print()
    extract_filelist_compact(imagenet100_root)
