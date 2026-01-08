#!/usr/bin/env python3
"""
生成网络综合对比图表
包括：
1. 网络综合对比图（6个子图）：FHE延迟、ImageNet-1K准确率、FLOPs、最大深度、浅层延迟百分比、浅层FLOPs百分比
2. 每个网络的深度-FLOPs分布图
"""

import sys
import os

# 添加父目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision
from fhe_statistics.statistics_fn import (
    FheInfo, 
    IMAGENET1K_ACCURACY, 
    generate_unique_filename
)


def analyze_all_networks():
    """分析所有网络并生成综合对比图表"""
    
    # 定义要分析的模型
    models_to_test = [
        ("ResNet18", torchvision.models.resnet18),
        ("ResNet34", torchvision.models.resnet34),
        ("ResNet50", torchvision.models.resnet50),
        ("VGG16", torchvision.models.vgg16),
        ("MobileNetV2", torchvision.models.mobilenet_v2),
    ]

    # 尝试添加 EfficientNet
    try:
        models_to_test.append(("EfficientNet_B0", torchvision.models.efficientnet_b0))
    except AttributeError:
        print("EfficientNet not available in this torchvision version")

    # 设置输出文件夹（使用绝对路径）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "results")
    os.makedirs(output_folder, exist_ok=True)

    # 存储所有网络的FheInfo对象
    network_infos = {}

    print("=" * 80)
    print("开始分析所有网络...")
    print("=" * 80)

    # 第一步：分析每个网络
    for name, model_fn in models_to_test:
        print(f"\n{'='*80}")
        print(f"正在分析: {name}")
        print(f"{'='*80}")
        
        try:
            # 创建模型并分析
            model = model_fn()
            fhe_info = FheInfo(model, input_shape=(1, 3, 224, 224), model_name=name)
            fhe_info.run_statistics()
            
            # 打印基本统计信息
            print(f"\n{name} 统计摘要:")
            print(f"  总延迟（含boot）: {fhe_info.total_latency + fhe_info.total_boot_latency:.2f}")
            print(f"  总FLOPs: {fhe_info.get_flops_count():,}")
            print(f"  最大深度: {fhe_info.get_max_depth()}")
            print(f"  ImageNet-1K准确率: {IMAGENET1K_ACCURACY.get(name, 0):.1f}%")
            
            # 计算浅层指标
            shallow_metrics = fhe_info.get_shallow_layer_metrics(shallow_threshold=0.2)
            print(f"  浅层深度阈值: {shallow_metrics['shallow_depth_threshold']}")
            print(f"  浅层延迟占比: {shallow_metrics['shallow_latency_pct']:.2f}%")
            print(f"  浅层FLOPs占比: {shallow_metrics['shallow_flops_pct']:.2f}%")
            
            # 保存FheInfo对象
            network_infos[name] = fhe_info
            
            # 为每个网络生成深度-FLOPs分布图
            print(f"\n生成 {name} 的深度-FLOPs分布图...")
            fhe_info.plot_depth_flops_distribution(
                bin_size=10, 
                max_bins=30, 
                plot_folder=output_folder, 
                show=False
            )
            
        except Exception as e:
            print(f"错误: 分析 {name} 失败")
            print(f"  {e}")
            import traceback
            traceback.print_exc()

    # 第二步：生成网络综合对比图
    if network_infos:
        print(f"\n{'='*80}")
        print("生成网络综合对比图（6个子图）...")
        print(f"{'='*80}")
        
        try:
            FheInfo.plot_network_comprehensive_comparison(
                network_infos, 
                plot_folder=output_folder, 
                show=False
            )
            print("\n网络综合对比图已生成！")
        except Exception as e:
            print(f"错误: 生成综合对比图失败")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n没有成功分析的网络，无法生成综合对比图")

    # 第三步：生成网络分组对比图（每个网络显示所有6个指标）
    if network_infos:
        print(f"\n{'='*80}")
        print("生成网络分组对比图（每个网络显示所有6个指标）...")
        print(f"{'='*80}")
        
        try:
            FheInfo.plot_network_grouped_comparison(
                network_infos, 
                plot_folder=output_folder, 
                show=False
            )
            print("\n网络分组对比图已生成！")
        except Exception as e:
            print(f"错误: 生成分组对比图失败")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n没有成功分析的网络，无法生成分组对比图")

    # 第四步：生成摘要报告
    if network_infos:
        print(f"\n{'='*80}")
        print("生成摘要报告...")
        print(f"{'='*80}")
        
        summary_lines = []
        summary_lines.append("=" * 100)
        summary_lines.append("网络综合对比摘要")
        summary_lines.append("=" * 100)
        summary_lines.append(f"{'Network':<20} {'FLOPs(M)':>12} {'FHE Latency(M)':>16} {'MaxDepth':>10} {'Accuracy(%)':>12} {'ShallowLat(%)':>14} {'ShallowFLOPs(%)':>15}")
        summary_lines.append("-" * 100)
        
        for name in sorted(network_infos.keys()):
            info = network_infos[name]
            flops = info.get_flops_count() / 1e6
            latency = (info.total_latency + info.total_boot_latency) / 1e6
            max_depth = info.get_max_depth()
            accuracy = IMAGENET1K_ACCURACY.get(name, 0)
            shallow_metrics = info.get_shallow_layer_metrics(shallow_threshold=0.2)
            
            line = f"{name:<20} {flops:>12.2f} {latency:>16.2f} {max_depth:>10} {accuracy:>12.1f} {shallow_metrics['shallow_latency_pct']:>14.2f} {shallow_metrics['shallow_flops_pct']:>15.2f}"
            summary_lines.append(line)
        
        summary_lines.append("=" * 100)
        
        summary = "\n".join(summary_lines)
        print(summary)
        
        # 保存摘要报告
        summary_file = generate_unique_filename("network_comparison_summary", "txt", output_folder)
        with open(summary_file, "w") as f:
            f.write(summary)
        print(f"\n摘要报告已保存到: {summary_file}")

    print(f"\n{'='*80}")
    print("所有分析完成！")
    print(f"{'='*80}")


if __name__ == "__main__":
    analyze_all_networks()