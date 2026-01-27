#!/usr/bin/env python3
"""
简单版本：从ImageNet1K随机选择100个类，只保存类名列表
"""

import random
from pathlib import Path

def select_classes_simple(imagenet1k_root, output_root, seed=42):
    """随机选择100个类，保存类名列表"""
    
    train_dir = Path(imagenet1k_root) / "train"
    output_path = Path(output_root)
    
    if not train_dir.exists():
        print(f"❌ 找不到: {train_dir}")
        return False
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有类
    all_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print(f"📁 发现 {len(all_classes)} 个类")
    
    # 随机选择100个
    random.seed(seed)
    selected = sorted(random.sample(all_classes, 100))
    
    # 保存到文件
    output_file = output_path / "selected_classes.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(selected))
    
    print(f"🎲 随机选择100个类 (seed={seed})")
    print(f"📄 保存到: {output_file}")
    print(f"   前10个: {selected[:10]}")
    
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python select_classes.py <imagenet1k_root> <output_root> [seed]")
        print("示例: python select_classes.py /data/imagenet1k /data/imagenet100")
        sys.exit(1)
    
    imagenet1k_root = sys.argv[1]
    output_root = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    
    select_classes_simple(imagenet1k_root, output_root, seed)