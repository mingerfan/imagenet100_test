#!/usr/bin/env python3
"""测试 gate_net.py 中的特殊模型是否能正确进行 FHE 统计"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '.')

from models import get_model, MODEL_REGISTRY
from fhe_statistics.statistics_fn import FheInfo


def test_single_model(model_name, input_shape=(1, 3, 64, 64), num_classes=100, verbose=True):
    """测试单个模型的FHE统计

    Args:
        model_name: 模型名称
        input_shape: 输入形状（使用较小尺寸加快测试）
        num_classes: 类别数
        verbose: 是否打印详细信息

    Returns:
        bool: 测试是否成功
    """
    try:
        if verbose:
            print(f"\n{'='*80}")
            print(f"测试模型: {model_name}")
            print(f"{'='*80}")

        # 创建模型
        model = get_model(model_name, num_classes=num_classes)
        model.eval()

        if verbose:
            # 计算参数量
            param_count = sum(p.numel() for p in model.parameters())
            print(f"参数量: {param_count:,} ({param_count/1e6:.2f}M)")

        # 创建 FheInfo 并运行统计
        fhe_info = FheInfo(model, input_shape=input_shape, model_name=model_name)
        fhe_info.run_statistics()

        # 打印统计结果
        if verbose:
            fhe_info.print_statistics()

        # 验证统计结果的基本有效性
        assert fhe_info.total_latency > 0, "总延迟应该大于0"
        assert len(fhe_info.op_stats) > 0, "应该有操作统计"
        assert fhe_info.get_max_depth() > 0, "最大深度应该大于0"

        if verbose:
            print(f"✓ {model_name} 测试通过")

        return True

    except Exception as e:
        print(f"✗ {model_name} 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_gate_net_models(sample_size=None, verbose=True):
    """测试所有 gate_net 模型

    Args:
        sample_size: 如果指定，只测试前 N 个模型
        verbose: 是否打印详细信息

    Returns:
        dict: 测试结果统计
    """
    print("="*80)
    print("测试 gate_net.py 中的所有模型")
    print("="*80)

    # 获取所有注册的模型
    all_models = MODEL_REGISTRY.list_models()

    # 过滤出 gate_net 相关的模型（以 'resnet-' 开头的）
    gate_net_models = [m for m in all_models if m.startswith('resnet-')]

    print(f"\n找到 {len(gate_net_models)} 个 gate_net 模型:")
    for i, model_name in enumerate(gate_net_models, 1):
        print(f"  {i}. {model_name}")

    # 如果指定了样本数量，只测试部分模型
    if sample_size is not None:
        gate_net_models = gate_net_models[:sample_size]
        print(f"\n只测试前 {sample_size} 个模型")

    # 测试结果统计
    results = {
        'total': len(gate_net_models),
        'passed': 0,
        'failed': 0,
        'failed_models': []
    }

    # 测试每个模型
    for i, model_name in enumerate(gate_net_models, 1):
        print(f"\n[{i}/{len(gate_net_models)}] ", end="")

        success = test_single_model(
            model_name,
            input_shape=(1, 3, 64, 64),  # 使用较小尺寸加快测试
            num_classes=100,
            verbose=verbose
        )

        if success:
            results['passed'] += 1
        else:
            results['failed'] += 1
            results['failed_models'].append(model_name)

    # 打印测试总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"总计: {results['total']} 个模型")
    print(f"通过: {results['passed']} 个")
    print(f"失败: {results['failed']} 个")

    if results['failed'] > 0:
        print("\n失败的模型:")
        for model_name in results['failed_models']:
            print(f"  - {model_name}")

    print("="*80)

    return results


def test_activation_preservation():
    """测试激活函数是否被正确保留（不被拆分）"""
    print("\n" + "="*80)
    print("测试激活函数保留性")
    print("="*80)

    # 测试几个代表性模型
    test_models = [
        'resnet-basic-relu-layer1block1',
        'resnet-basic-swish-layer1block1',
        'resnet-basic-learnableswish-layer1block1',
        'resnet-basic-learnablerelu-layer1block1',
        'resnet-basic-stablepoly4-layer1block1',
    ]

    for model_name in test_models:
        print(f"\n测试 {model_name}...")

        # 创建模型
        model = get_model(model_name, num_classes=100)
        model.eval()

        # 创建 FheInfo
        fhe_info = FheInfo(model, input_shape=(1, 3, 64, 64), model_name=model_name)

        # 检查 FX 图中的节点
        activation_types = set()
        has_fine_grained_ops = False

        for node in fhe_info.traced.graph.nodes:
            if node.op == "call_module":
                module = fhe_info.traced.get_submodule(str(node.target))
                module_type = type(module).__name__

                # 记录激活函数类型
                if module_type in ['LearnableSwish', 'Swish', 'LearnableRelu', 'StablePoly4', 'Relu']:
                    activation_types.add(module_type)

            elif node.op == "call_function":
                # 检查是否有细粒度操作（sigmoid, mul, maximum等）
                func_name = str(node.target).lower()
                if any(op in func_name for op in ['sigmoid', 'maximum']):
                    has_fine_grained_ops = True
                    print(f"  ✗ 发现细粒度操作: {node.name} ({node.target})")

        # 打印结果
        if activation_types:
            print(f"  ✓ 找到激活函数: {', '.join(activation_types)}")

        if not has_fine_grained_ops:
            print(f"  ✓ 未发现细粒度操作（激活函数未被拆分）")
        else:
            print(f"  ✗ 发现细粒度操作（激活函数可能被拆分）")


def quick_test():
    """快速测试：只测试几个代表性模型"""
    print("="*80)
    print("快速测试模式：测试代表性模型")
    print("="*80)

    # 选择几个代表性模型
    representative_models = [
        'resnet-basic-relu-layer1block1',
        'resnet-basic-swish-layer1block1',
        'resnet-basic_self_gated-learnableswish-layer1block1',
        'resnet-bottleneck-stablepoly4-layer1block1',
        'resnet-bottleneck_self_gated-relu-layer1block1',
    ]

    results = {
        'total': len(representative_models),
        'passed': 0,
        'failed': 0,
        'failed_models': []
    }

    for i, model_name in enumerate(representative_models, 1):
        print(f"\n[{i}/{len(representative_models)}] ", end="")

        success = test_single_model(
            model_name,
            input_shape=(1, 3, 64, 64),
            num_classes=100,
            verbose=True
        )

        if success:
            results['passed'] += 1
        else:
            results['failed'] += 1
            results['failed_models'].append(model_name)

    print("\n" + "="*80)
    print("快速测试总结")
    print("="*80)
    print(f"通过: {results['passed']}/{results['total']}")
    print(f"失败: {results['failed']}/{results['total']}")
    print("="*80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='测试 gate_net 模型的 FHE 统计')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'all', 'single', 'activation'],
                       help='测试模式: quick=快速测试, all=测试所有模型, single=测试单个模型, activation=测试激活函数保留性')
    parser.add_argument('--model', type=str, default='resnet-basic-relu-layer1block1',
                       help='当mode=single时，指定要测试的模型名称')
    parser.add_argument('--sample', type=int, default=None,
                       help='当mode=all时，指定测试的样本数量（默认测试全部）')
    parser.add_argument('--quiet', action='store_true',
                       help='安静模式，不打印详细信息')

    args = parser.parse_args()

    verbose = not args.quiet

    if args.mode == 'quick':
        quick_test()
    elif args.mode == 'all':
        test_all_gate_net_models(sample_size=args.sample, verbose=verbose)
    elif args.mode == 'single':
        test_single_model(args.model, verbose=verbose)
    elif args.mode == 'activation':
        test_activation_preservation()
