"""
计算多个模型的zero-cost代理指标

计算以下模型的 SynFlow 和 ZEN 分数：
- resnet-18
- resnet-34
- resnet-50
- mobilenet-v2
- efficientnet-b0
"""

import torch
import torch.nn as nn
from torchvision import models
import sys
import os
import time

# 导入zero_cost_proxy中的计算函数
from zero_cost_proxy import compute_synflow, compute_zen_score


def get_model(model_name: str, num_classes: int = 100) -> nn.Module:
    """
    获取模型实例（不加载预训练权重）

    Args:
        model_name: 模型名称
        num_classes: 输出类别数

    Returns:
        PyTorch模型
    """
    if model_name == 'resnet-18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet-34':
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet-50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet-v2':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet-b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    return model


def print_synflow_results(model_name: str, results: dict):
    """
    格式化打印SynFlow结果

    Args:
        model_name: 模型名称
        results: 计算结果字典
    """
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"{'='*60}")
    print(f"SynFlow 分数: {results['synflow_score']:.6f}")
    print(f"梯度 L2 范数: {results['synflow_grad_norm']:.6f}")
    print(f"参数数量: {results['synflow_params']:,}")
    print(f"{'='*60}\n")


def print_zen_results(model_name: str, results: dict):
    """
    格式化打印ZEN结果

    Args:
        model_name: 模型名称
        results: 计算结果字典
    """
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"{'='*60}")
    print(f"ZEN 分数: {results['zen_score']:.4g}")
    print(f"标准差: {results['std_zen_score']:.6f}")
    print(f"95% 置信区间精度: {results['zen_precision']:.6f}")
    print(f"{'='*60}\n")


def main():
    """主函数"""
    # 配置参数
    resolution = 224  # ImageNet标准分辨率
    batch_size = 16
    gpu = 0
    fp16 = False

    # ZEN 特定参数
    zen_repeat = 32
    zen_mixup_gamma = 1e-2

    # 要计算的模型列表
    model_names = [
        'resnet-18',
        'resnet-34',
        'resnet-50',
        'mobilenet-v2',
        'efficientnet-b0'
    ]

    print("="*80)
    print("开始计算模型的 zero-cost 代理指标")
    print("="*80)
    print(f"分辨率: {resolution}, 批次大小: {batch_size}")
    print(f"ZEN 重复次数: {zen_repeat}, Mixup 系数: {zen_mixup_gamma}")
    print("="*80)

    # SynFlow 计算
    print("\n\n>>> 计算 SynFlow 分数 <<<\n")
    for model_name in model_names:
        print(f"正在加载 {model_name}...")

        try:
            # 获取模型
            model = get_model(model_name, num_classes=100)

            # 计算SynFlow指标
            results = compute_synflow(
                model=model,
                gpu=gpu,
                resolution=resolution,
                batch_size=batch_size,
                fp16=fp16
            )

            # 打印结果
            print_synflow_results(model_name, results)

        except Exception as e:
            print(f"计算 {model_name} 的 SynFlow 时出错: {e}")
            import traceback
            traceback.print_exc()

    # ZEN Score 计算
    print("\n\n>>> 计算 ZEN 分数 <<<\n")
    for model_name in model_names:
        print(f"正在加载 {model_name}...")

        try:
            # 获取模型
            model = get_model(model_name, num_classes=100)

            # 计算ZEN指标
            start_time = time.time()
            results = compute_zen_score(
                model=model,
                gpu=gpu,
                resolution=resolution,
                batch_size=batch_size,
                repeat=zen_repeat,
                mixup_gamma=zen_mixup_gamma,
                fp16=fp16
            )
            time_cost = (time.time() - start_time) / zen_repeat

            # 打印结果
            print_zen_results(model_name, results)
            print(f"平均计算时间: {time_cost:.4g} 秒/次")

        except Exception as e:
            print(f"计算 {model_name} 的 ZEN 时出错: {e}")
            import traceback
            traceback.print_exc()

    print("\n所有模型的 zero-cost 计算完成！")


if __name__ == '__main__':
    main()