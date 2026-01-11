#!/usr/bin/env python
"""
简单示例：快速上手FHE统计分析工具

展示三种常见使用方式：
1. 分析单个模型
2. 比较多个模型
3. 使用批量分析工具
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torchvision
from fhe_statistics import FheInfo, analyze_model, compare_networks


def example_1_single_model():
    """示例1：分析单个模型"""
    print("\n" + "="*80)
    print("示例1：分析单个模型 - ResNet18")
    print("="*80 + "\n")

    # 加载模型
    model = torchvision.models.resnet18()

    # 分析模型（会自动打印统计结果并生成图表）
    fhe_info = analyze_model(
        model=model,
        model_name="ResNet18",
        output_folder="fhe_statistics/results",
        plot_folder="fhe_statistics/results",
        input_shape=(1, 3, 224, 224),
        print_detailed=False,  # 设置为True可查看详细的逐层统计
        optimize_boot=True     # 使用动态规划优化boot插入
    )

    print(f"\n分析完成！")
    print(f"  总延迟（含boot）: {(fhe_info.total_latency + fhe_info.total_boot_latency)/1e6:.2f}M")
    print(f"  Boot次数: {fhe_info.total_boot_count}")
    print(f"  最大深度: {fhe_info.get_max_depth()}")


def example_2_compare_models():
    """示例2：比较多个模型"""
    print("\n" + "="*80)
    print("示例2：比较多个模型")
    print("="*80 + "\n")

    # 准备要比较的模型列表
    models = [
        ("ResNet18", torchvision.models.resnet18()),
        ("MobileNetV2", torchvision.models.mobilenet_v2()),
    ]

    # 批量比较（会自动生成横向比较图）
    compare_networks(
        models=models,
        plot_folder="fhe_statistics/results",
        input_shape=(1, 3, 224, 224)
    )

    print(f"\n比较完成！结果保存在 fhe_statistics/results/")


def example_3_batch_analysis():
    """示例3：使用批量分析工具（推荐用于大规模对比）"""
    print("\n" + "="*80)
    print("示例3：批量分析工具")
    print("="*80 + "\n")

    from fhe_statistics.batch_analyzer import BatchAnalyzer

    print("1. 首先，查看配置文件中的所有模型：")
    print("   python fhe_statistics/batch_analyzer.py --list\n")

    print("2. 然后，编辑 batch_analysis_config.yaml 启用想要分析的模型\n")

    print("3. 运行批量分析：")
    print("   python fhe_statistics/batch_analyzer.py\n")

    print("4. 或者只分析特定模型：")
    print("   python fhe_statistics/batch_analyzer.py --models ResNet18 ResNet34\n")

    # 也可以在代码中使用
    # analyzer = BatchAnalyzer("fhe_statistics/batch_analysis_config.yaml")
    # analyzer.run(specific_models=["ResNet18", "MobileNetV2"])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='FHE统计分析工具示例')
    parser.add_argument('--example', type=int, choices=[1, 2, 3], default=1,
                       help='运行哪个示例 (1: 单模型, 2: 多模型比较, 3: 批量分析说明)')

    args = parser.parse_args()

    if args.example == 1:
        example_1_single_model()
    elif args.example == 2:
        example_2_compare_models()
    elif args.example == 3:
        example_3_batch_analysis()
