#!/usr/bin/env python3
"""
从文件清单在目标服务器上重建ImageNet100
在目标服务器上运行：python rebuild_from_filelist.py imagenet100_filelist_compact.json /path/to/imagenet1k /path/to/output
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List


def rebuild_from_filelist(
    filelist_file: str,
    imagenet1k_root: str,
    output_root: str,
    use_symlink: bool = True,
    use_hardlink: bool = False,
    copy_files: bool = False
):
    """
    从文件清单重建ImageNet100
    支持两种清单格式：
    1. {"class_id": ["file1.jpg", "file2.jpg"]} 
    2. {"class_id": ["train/file1.jpg", "val/file2.jpg"]}
    
    Args:
        filelist_file: 文件清单JSON文件
        imagenet1k_root: 源ImageNet1K路径
        output_root: 输出ImageNet100路径
        use_symlink: 使用符号链接（推荐，最节省空间）
        use_hardlink: 使用硬链接（节省空间但不跨文件系统）
        copy_files: 复制文件（最安全但占用空间）
    """
    
    filelist_path = Path(filelist_file)
    imagenet1k_path = Path(imagenet1k_root)
    output_path = Path(output_root)
    
    # 验证输入
    if not filelist_path.exists():
        print(f"❌ 文件清单不存在: {filelist_file}")
        return False
    
    if not imagenet1k_path.exists():
        print(f"❌ ImageNet1K路径不存在: {imagenet1k_root}")
        return False
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 读取清单
    with open(filelist_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 兼容两种JSON格式
    if isinstance(data, dict):
        # 检查是否是旧格式（包含 "classes" 和 "files" 字段）
        if "classes" in data and "files" in data:
            # 旧格式：转换为新格式
            print(f"📋 检测到旧JSON格式，正在转换...")
            classes = data.get("classes", [])
            files = data.get("files", [])
            
            file_dict = {}
            for file_path in files:
                # 解析路径获取类名
                parts = file_path.split('/')
                if len(parts) >= 2:
                    # train/val 结构
                    class_name = parts[-2]
                else:
                    # 直接结构
                    class_name = parts[0]
                
                if class_name not in file_dict:
                    file_dict[class_name] = []
                file_dict[class_name].append(file_path)
        else:
            # 新格式：直接使用
            file_dict = data
    else:
        print(f"❌ JSON格式错误")
        return False
    
    print(f"📋 读取清单: {len(file_dict)} 个类")
    print(f"📁 源ImageNet1K: {imagenet1k_root}")
    print(f"📁 输出目录: {output_root}")
    print(f"🔗 链接方式: ", end="")
    
    if use_symlink:
        print("符号链接")
        link_method = "symlink"
    elif use_hardlink:
        print("硬链接")
        link_method = "hardlink"
    else:
        print("复制文件")
        link_method = "copy"
    
    print()
    
    stats = {
        "total_classes": 0,
        "total_files": 0,
        "success": 0,
        "missing": 0,
        "errors": []
    }
    
    # 遍历清单中的每个类
    for class_name, files in sorted(file_dict.items()):
        src_class_path = imagenet1k_path / class_name
        dst_class_path = output_path / class_name
        
        # 首先尝试直接路径
        if not os.path.isdir(str(src_class_path)):
            # 如果不存在，尝试 train/class 和 val/class
            train_class_path = imagenet1k_path / "train" / class_name
            val_class_path = imagenet1k_path / "val" / class_name
            
            if os.path.isdir(str(train_class_path)):
                src_class_path = train_class_path
            elif os.path.isdir(str(val_class_path)):
                src_class_path = val_class_path
            else:
                print(f"⚠  类不存在: {class_name}")
                stats["missing"] += 1
                stats["errors"].append(f"Missing class: {class_name}")
                continue
        
        stats["total_classes"] += 1
        
        # 创建目标类目录
        dst_class_path.mkdir(parents=True, exist_ok=True)
        
        # 直接复制类目录下的所有文件，不按清单逐个匹配
        try:
            all_files_in_class = list(src_class_path.iterdir())
        except Exception as e:
            print(f"⚠  无法读取类目录 {class_name}: {e}")
            continue
        
        for src_file in sorted(all_files_in_class):
            if not os.path.isfile(str(src_file)):
                continue
            
            filename = src_file.name
            dst_file = dst_class_path / filename
            
            stats["total_files"] += 1
            
            try:
                # 如果目标文件已存在则跳过
                if dst_file.exists():
                    stats["success"] += 1
                    continue
                
                # 使用指定的链接方式
                if link_method == "symlink":
                    os.symlink(src_file, dst_file)
                elif link_method == "hardlink":
                    os.link(src_file, dst_file)
                else:  # copy
                    import shutil
                    shutil.copy2(src_file, dst_file)
                
                stats["success"] += 1
                
            except Exception as e:
                print(f"❌ 处理失败: {class_name}/{filename}")
                print(f"   错误: {e}")
                stats["errors"].append(f"Error: {class_name}/{filename} - {str(e)}")
    
    # 打印总结
    print("\n" + "="*50)
    print("📊 重建完成统计")
    print("="*50)
    print(f"✓ 成功处理的文件: {stats['success']}")
    print(f"⚠  缺失的文件: {len(stats['errors']) - stats['missing']}")
    print(f"⚠  缺失的类: {stats['missing']}")
    print(f"📁 总类数: {stats['total_classes']}")
    print(f"📄 总文件数: {stats['total_files']}")
    
    if stats["errors"]:
        print(f"\n❌ 遇到 {len(stats['errors'])} 个错误:")
        for error in stats["errors"][:10]:  # 只显示前10个
            print(f"   - {error}")
        if len(stats["errors"]) > 10:
            print(f"   ... 还有 {len(stats['errors']) - 10} 个错误")
    
    success_rate = stats["success"] / stats["total_files"] * 100 if stats["total_files"] > 0 else 0
    print(f"\n✓ 成功率: {success_rate:.1f}%")
    
    return stats["success"] == stats["total_files"]


def create_test_filelist(imagenet100_root: str, output_file: str = "imagenet100_filelist_compact.json"):
    """
    创建测试用的紧凑清单（仅包含每个类的前3张图片）
    用于测试重建流程
    """
    imagenet100_root = Path(imagenet100_root)
    
    file_dict = {}
    for class_dir in sorted(imagenet100_root.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        files = sorted([f.name for f in class_dir.iterdir() if f.is_file()])[:3]
        if files:
            file_dict[class_name] = files
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(file_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 测试清单已生成: {output_file}")
    print(f"  类数: {len(file_dict)}, 每个类3张图片")
    return output_file


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("用法: python rebuild_from_filelist.py <filelist.json> <imagenet1k_path> <output_path> [--symlink|--hardlink|--copy]")
        print("\n示例:")
        print("  python rebuild_from_filelist.py imagenet100_filelist_compact.json /mnt/imagenet1k /mnt/imagenet100")
        print("  python rebuild_from_filelist.py imagenet100_filelist_compact.json /data/imagenet1k /data/imagenet100 --copy")
        print("\n选项:")
        print("  --symlink   使用符号链接（默认，最节省空间）")
        print("  --hardlink  使用硬链接（节省空间）")
        print("  --copy      复制文件（最安全，占用完整空间）")
        sys.exit(1)
    
    filelist_file = sys.argv[1]
    imagenet1k_root = sys.argv[2]
    output_root = sys.argv[3]
    
    # 解析链接方式
    use_symlink = True
    use_hardlink = False
    copy_files = False
    
    if len(sys.argv) > 4:
        method = sys.argv[4].lower()
        if method == "--symlink":
            use_symlink = True
        elif method == "--hardlink":
            use_symlink = False
            use_hardlink = True
        elif method == "--copy":
            use_symlink = False
            copy_files = True
    
    print(f"🚀 开始重建ImageNet100\n")
    
    success = rebuild_from_filelist(
        filelist_file,
        imagenet1k_root,
        output_root,
        use_symlink=use_symlink,
        use_hardlink=use_hardlink,
        copy_files=copy_files
    )
    
    sys.exit(0 if success else 1)
