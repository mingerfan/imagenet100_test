#!/usr/bin/env python3
"""
测试变异修复是否正确

验证：
1. Stride变异后通道数是否正确重算
2. CT Policy变异后通道数是否正确重算
3. Population年龄管理是否正确
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_gen import RandomNetworkGenerator
from nas_evolution.mutations import MutationOperator
from nas_evolution.population import Population
from network_gen.search_space import ChannelCalculator


def test_stride_mutation_channel_recalc():
    """测试stride变异后通道数是否正确重算"""
    print("=" * 80)
    print("测试1: Stride变异后通道数重算")
    print("=" * 80)

    # 生成随机网络
    generator = RandomNetworkGenerator(seed=42)
    config = generator.generate_random_config()

    print(f"\n原始配置:")
    print(f"  Stride code: {config.stride_code}")
    print(f"  Strides: {[b.stride for b in config.blocks]}")
    print(f"  Channels: {[b.out_channels for b in config.blocks]}")
    print(f"  CT policies: {config.ct_policies}")

    # 计算预期的通道数（用于对比）
    calculator = ChannelCalculator(ct_slots=32768, input_size=224)
    strides_before = [b.stride for b in config.blocks]
    expected_channels_before, _, _ = calculator.compute_channels_sequence(
        strides=strides_before,
        ct_policies=config.ct_policies,
        initial_ct_count=config.initial_ct_count
    )

    print(f"\n验证原始通道数:")
    actual_channels_before = [b.out_channels for b in config.blocks]
    match_before = actual_channels_before == expected_channels_before
    print(f"  实际通道数: {actual_channels_before}")
    print(f"  预期通道数: {expected_channels_before}")
    print(f"  匹配: {'✅' if match_before else '❌'}")

    # 应用stride变异
    mutator = MutationOperator()
    mutated_config = mutator._mutate_stride(config)

    if mutated_config is None:
        # 如果没有可用的变异，使用原配置
        mutated_config = config

    print(f"\n变异后配置:")
    print(f"  Stride code: {mutated_config.stride_code}")
    print(f"  Strides: {[b.stride for b in mutated_config.blocks]}")
    print(f"  Channels: {[b.out_channels for b in mutated_config.blocks]}")

    # 验证变异后的通道数
    strides_after = [b.stride for b in mutated_config.blocks]
    expected_channels_after, _, _ = calculator.compute_channels_sequence(
        strides=strides_after,
        ct_policies=mutated_config.ct_policies,
        initial_ct_count=mutated_config.initial_ct_count
    )

    print(f"\n验证变异后通道数:")
    actual_channels_after = [b.out_channels for b in mutated_config.blocks]
    match_after = actual_channels_after == expected_channels_after
    print(f"  实际通道数: {actual_channels_after}")
    print(f"  预期通道数: {expected_channels_after}")
    print(f"  匹配: {'✅' if match_after else '❌'}")

    return match_after


def test_ct_policy_mutation_channel_recalc():
    """测试CT policy变异后通道数是否正确重算"""
    print("\n" + "=" * 80)
    print("测试2: CT Policy变异后通道数重算")
    print("=" * 80)

    # 生成随机网络
    generator = RandomNetworkGenerator(seed=123)
    config = generator.generate_random_config()

    print(f"\n原始配置:")
    print(f"  CT policies: {config.ct_policies}")
    print(f"  Strides: {[b.stride for b in config.blocks]}")
    print(f"  Channels: {[b.out_channels for b in config.blocks]}")

    # 应用CT policy变异
    mutator = MutationOperator()
    mutator._mutate_ct_policy(config)

    print(f"\n变异后配置:")
    print(f"  CT policies: {config.ct_policies}")
    print(f"  Channels: {[b.out_channels for b in config.blocks]}")

    # 验证变异后的通道数
    calculator = ChannelCalculator(ct_slots=32768, input_size=224)
    strides = [b.stride for b in config.blocks]
    expected_channels, _, _ = calculator.compute_channels_sequence(
        strides=strides,
        ct_policies=config.ct_policies,
        initial_ct_count=config.initial_ct_count
    )

    print(f"\n验证变异后通道数:")
    actual_channels = [b.out_channels for b in config.blocks]
    match = actual_channels == expected_channels
    print(f"  实际通道数: {actual_channels}")
    print(f"  预期通道数: {expected_channels}")
    print(f"  匹配: {'✅' if match else '❌'}")

    return match


def test_population_age_management():
    """测试Population年龄管理是否正确"""
    print("\n" + "=" * 80)
    print("测试3: Population年龄管理")
    print("=" * 80)

    # 创建population
    pop = Population(max_size=5)

    # 添加3个个体
    generator = RandomNetworkGenerator()
    for i in range(3):
        config = generator.generate_random_config()
        scores = {'expressivity': 0.5, 'progressivity': 0.5, 'trainability': 0.5, 'fhe_latency': 1000}
        pop.add(config, scores, aznas_fitness=0.0, generation=i)

    print(f"\n添加3个个体后:")
    for i, ind in enumerate(pop.individuals):
        print(f"  Individual {i}: age={ind.age}, generation={ind.generation}")

    # 验证年龄
    expected_ages = [2, 1, 0]  # 第一个个体经历了2次递增，第二个1次，第三个0次
    actual_ages = [ind.age for ind in pop.individuals]

    print(f"\n验证年龄:")
    print(f"  实际年龄: {actual_ages}")
    print(f"  预期年龄: {expected_ages}")
    match = actual_ages == expected_ages
    print(f"  匹配: {'✅' if match else '❌'}")

    # 再添加2个个体，触发FIFO移除
    print(f"\n再添加2个个体（触发FIFO）:")
    for i in range(2):
        config = generator.generate_random_config()
        scores = {'expressivity': 0.5, 'progressivity': 0.5, 'trainability': 0.5, 'fhe_latency': 1000}
        pop.add(config, scores, aznas_fitness=0.0, generation=3+i)

    print(f"\n当前population:")
    for i, ind in enumerate(pop.individuals):
        print(f"  Individual {i}: age={ind.age}, generation={ind.generation}")

    # 验证最新个体的年龄为0
    newest_age = pop.individuals[-1].age
    print(f"\n最新个体年龄: {newest_age}")
    print(f"预期: 0")
    print(f"匹配: {'✅' if newest_age == 0 else '❌'}")

    return match and (newest_age == 0)


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("NAS进化算法修复验证测试")
    print("=" * 80)

    results = []

    # 测试1: Stride变异
    try:
        result1 = test_stride_mutation_channel_recalc()
        results.append(("Stride变异通道数重算", result1))
    except Exception as e:
        print(f"\n❌ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Stride变异通道数重算", False))

    # 测试2: CT Policy变异
    try:
        result2 = test_ct_policy_mutation_channel_recalc()
        results.append(("CT Policy变异通道数重算", result2))
    except Exception as e:
        print(f"\n❌ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("CT Policy变异通道数重算", False))

    # 测试3: Population年龄管理
    try:
        result3 = test_population_age_management()
        results.append(("Population年龄管理", result3))
    except Exception as e:
        print(f"\n❌ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Population年龄管理", False))

    # 总结
    print("\n" + "=" * 80)
    print("测试结果总结")
    print("=" * 80)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print(f"\n总体结果: {'✅ 所有测试通过' if all_passed else '❌ 部分测试失败'}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
