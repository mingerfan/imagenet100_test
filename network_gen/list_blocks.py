#!/usr/bin/env python3
"""
列出所有可用的Block名称

使用方法:
    python list_blocks.py
    python list_blocks.py --filter mbconv
    python list_blocks.py --filter poly4
"""

import argparse
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_gen.search_space import UNIFIED_BLOCKS, BLOCK_NAME_TO_ID


def list_blocks(filter_str: str = None):
    """列出所有block，可选过滤"""

    print("=" * 80)
    print("FHE-NAS 可用的 Block 类型")
    print("=" * 80)

    if filter_str:
        print(f"过滤条件: 包含 '{filter_str}'")
        print()

    # 按类别分组
    categories = {
        'MBConv1': (0, 3),
        'MBConv4': (4, 7),
        'GatedMBConv1': (8, 11),
        'GatedMBConv4': (12, 15),
        'BasicBlock': (16, 17),
        'BasicSelfGatedBlock': (18, 19),
        'FullGatedBasicBlock': (20, 21),
        'ReLU MBConv': (22, 25),
    }

    total_shown = 0

    for category_name, (start_id, end_id) in categories.items():
        # 收集该类别的blocks
        blocks_in_category = []
        for block_id in range(start_id, end_id + 1):
            spec = UNIFIED_BLOCKS[block_id]
            if filter_str is None or filter_str.lower() in spec.name.lower():
                blocks_in_category.append(spec)

        # 只显示有内容的类别
        if blocks_in_category:
            print(f"\n{category_name}:")
            print("-" * 80)
            for spec in blocks_in_category:
                print(f"  [{spec.id:2d}] {spec.name:30s} - {spec.description}")
                total_shown += 1

    print()
    print("=" * 80)
    if filter_str:
        print(f"共显示 {total_shown} 个block（过滤后）")
    else:
        print(f"共 {len(UNIFIED_BLOCKS)} 个block")
    print()

    # 显示使用示例
    print("在YAML配置中使用示例:")
    print("-" * 80)
    print("# 方式1: 使用数字ID")
    print("blocks:")
    print("  allowed_block_ids: [0, 1, 4, 5]")
    print()
    print("# 方式2: 使用block名称（推荐）")
    print("blocks:")
    print("  allowed_block_ids:")
    print("    - mbconv1_poly4")
    print("    - mbconv1_swish")
    print("    - mbconv4_poly4")
    print("    - mbconv4_swish")
    print()
    print("# 方式3: 混合使用")
    print("blocks:")
    print('  allowed_block_ids: [0, 1, "basic_poly4", "basic_swish"]')
    print()


def main():
    parser = argparse.ArgumentParser(
        description="列出所有可用的Block类型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python list_blocks.py                # 列出所有block
  python list_blocks.py --filter mbconv   # 只显示MBConv相关
  python list_blocks.py --filter poly4    # 只显示使用Poly4的block
  python list_blocks.py --filter gated    # 只显示门控block
        """
    )
    parser.add_argument(
        '--filter', '-f',
        type=str,
        help='过滤条件（包含指定字符串的block）'
    )

    args = parser.parse_args()
    list_blocks(args.filter)


if __name__ == "__main__":
    main()
