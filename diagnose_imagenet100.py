#!/usr/bin/env python3
"""
诊断ImageNet100目录结构
帮助定位为什么提取失败的问题
"""

import os
import sys
from pathlib import Path


def diagnose(imagenet100_root):
    """诊断目录结构"""
    root = Path(imagenet100_root)
    
    if not root.exists():
        print(f"❌ 路径不存在: {imagenet100_root}")
        return
    
    print(f"🔍 诊断目录: {imagenet100_root}\n")
    
    # 1. 检查目录本身
    print(f"📁 目录信息:")
    print(f"  是否存在: {root.exists()}")
    print(f"  是否是目录: {root.is_dir()}")
    print(f"  是否是符号链接: {root.is_symlink()}")
    if root.is_symlink():
        print(f"  符号链接指向: {root.resolve()}")
    print(f"  权限: {oct(os.stat(str(root)).st_mode)}")
    print()
    
    # 2. 列出顶级条目
    print(f"📂 顶级条目 (前20个):")
    try:
        entries = sorted(list(root.iterdir()))[:20]
        for i, entry in enumerate(entries, 1):
            is_dir = os.path.isdir(str(entry))
            is_link = entry.is_symlink()
            target = f" -> {entry.resolve()}" if is_link else ""
            print(f"  {i:2}. {'📁' if is_dir else '📄'} {entry.name}{target}")
    except Exception as e:
        print(f"  ❌ 无法列出条目: {e}")
    print()
    
    # 3. 统计各类型
    print(f"📊 统计信息:")
    try:
        all_entries = list(root.iterdir())
        dirs = [e for e in all_entries if os.path.isdir(str(e))]
        files = [e for e in all_entries if os.path.isfile(str(e))]
        links = [e for e in all_entries if e.is_symlink()]
        
        print(f"  总条目数: {len(all_entries)}")
        print(f"  目录数: {len(dirs)}")
        print(f"  文件数: {len(files)}")
        print(f"  符号链接数: {len(links)}")
        print()
    except Exception as e:
        print(f"  ❌ 统计失败: {e}")
        return
    
    # 4. 检查第一个类
    if dirs:
        first_class = dirs[0]
        print(f"🔍 检查第一个类: {first_class.name}")
        print(f"  是否是目录: {os.path.isdir(str(first_class))}")
        print(f"  是否是符号链接: {first_class.is_symlink()}")
        if first_class.is_symlink():
            print(f"  符号链接指向: {first_class.resolve()}")
        
        try:
            class_entries = list(first_class.iterdir())[:10]
            print(f"  包含的文件 (前10个):")
            for entry in class_entries:
                is_file = os.path.isfile(str(entry))
                print(f"    {'✓' if is_file else '✗'} {entry.name}")
            print(f"  总文件数: {len(list(first_class.iterdir()))}")
        except Exception as e:
            print(f"  ❌ 无法读取: {e}")
        print()
    
    # 5. 验证修复后的脚本能正确识别
    print(f"🧪 测试修复脚本:")
    try:
        from extract_filelist import extract_filelist, extract_filelist_compact
        
        # 创建临时清单
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_txt = os.path.join(tmpdir, "test.txt")
            tmp_json = os.path.join(tmpdir, "test.json")
            
            result = extract_filelist(imagenet100_root, tmp_txt)
            if result:
                with open(tmp_txt) as f:
                    lines = f.readlines()
                    file_lines = [l for l in lines if not l.startswith("#")]
                    print(f"  ✓ 提取成功，找到 {len(file_lines)} 个文件")
            else:
                print(f"  ❌ 提取失败")
    except Exception as e:
        print(f"  ❌ 测试异常: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python diagnose_imagenet100.py /path/to/imagenet100")
        sys.exit(1)
    
    diagnose(sys.argv[1])
