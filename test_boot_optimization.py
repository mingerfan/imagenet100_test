"""测试boot优化器的效果"""

import torch
import torchvision
from fhe_statistics.statistics_fn import analyze_model


def test_boot_optimization():
    """对比优化前后的boot成本"""

    print("=" * 80)
    print("Boot优化器测试")
    print("=" * 80)

    # 使用ResNet18作为测试模型
    model = torchvision.models.resnet18()

    print("\n【测试1】不使用boot优化（简单策略）")
    print("-" * 80)
    fhe_info_simple = analyze_model(
        model,
        model_name="ResNet18_SimpleBootStrategy",
        optimize_boot=False  # 关闭优化
    )

    simple_total = fhe_info_simple.total_latency + fhe_info_simple.total_boot_latency
    simple_boot = fhe_info_simple.total_boot_latency
    simple_boot_count = fhe_info_simple.total_boot_count

    print("\n【测试2】使用boot优化（动态规划）")
    print("-" * 80)
    fhe_info_optimized = analyze_model(
        model,
        model_name="ResNet18_OptimizedBootStrategy",
        optimize_boot=True  # 启用优化
    )

    optimized_total = fhe_info_optimized.total_latency + fhe_info_optimized.total_boot_latency
    optimized_boot = fhe_info_optimized.total_boot_latency
    optimized_boot_count = fhe_info_optimized.total_boot_count

    print("\n" + "=" * 80)
    print("优化效果对比")
    print("=" * 80)
    print(f"{'指标':<30} {'简单策略':>20} {'动态规划优化':>20} {'改进':>10}")
    print("-" * 80)
    print(f"{'Boot次数':<30} {simple_boot_count:>20} {optimized_boot_count:>20} {optimized_boot_count - simple_boot_count:>10}")
    print(f"{'Boot延迟':<30} {simple_boot:>20.2f} {optimized_boot:>20.2f} {(1 - optimized_boot/simple_boot)*100:>9.1f}%")
    print(f"{'总延迟（含boot）':<30} {simple_total:>20.2f} {optimized_total:>20.2f} {(1 - optimized_total/simple_total)*100:>9.1f}%")
    print("=" * 80)

    if optimized_boot < simple_boot:
        print(f"\n✓ 优化成功！Boot成本降低了 {(1 - optimized_boot/simple_boot)*100:.1f}%")
        print(f"  绝对节省: {simple_boot - optimized_boot:.2f}")
    elif optimized_boot == simple_boot:
        print(f"\n✓ 优化器找到了与简单策略相同的解（已是最优）")
    else:
        print(f"\n✗ 警告：优化后成本反而增加了！")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_boot_optimization()
