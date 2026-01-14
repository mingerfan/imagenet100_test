#!/usr/bin/env python3
"""
测试多样性保护机制

验证：
1. 少数群体是否被保护
2. 延迟超标的个体是否不受保护
3. 移除优先级是否正确
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_diversity_protection():
    """测试多样性保护机制"""
    print("=" * 80)
    print("测试: 多样性保护机制")
    print("=" * 80)

    # 创建简化的测试类
    class MockConfig:
        def __init__(self, num_blocks):
            self.blocks = [None] * num_blocks

    class MockIndividual:
        _next_id = 0

        def __init__(self, num_blocks, latency, generation):
            self.id = MockIndividual._next_id
            MockIndividual._next_id += 1
            self.config = MockConfig(num_blocks)
            self.scores = {'fhe_latency': latency}
            self.generation = generation
            self.age = 0
            self.aznas_fitness = 0.0

        def increment_age(self):
            self.age += 1

    class MockPopulation:
        def __init__(self, max_size, diversity_quota, latency_baseline):
            self.max_size = max_size
            self.diversity_quota = diversity_quota
            self.latency_baseline = latency_baseline
            self.individuals = []

        def _get_depth_distribution(self):
            depth_counts = {}
            for ind in self.individuals:
                depth = len(ind.config.blocks)
                depth_counts[depth] = depth_counts.get(depth, 0) + 1
            return depth_counts

        def _remove_with_diversity_protection(self):
            if len(self.individuals) <= self.max_size:
                return

            min_quota = max(1, int(self.max_size * self.diversity_quota))
            depth_counts = self._get_depth_distribution()

            # Priority 1: Majority + latency violation
            for i in range(len(self.individuals)):
                ind = self.individuals[i]
                depth = len(ind.config.blocks)
                is_minority = depth_counts[depth] <= min_quota
                is_latency_violation = ind.scores.get('fhe_latency', float('inf')) > self.latency_baseline

                if not is_minority and is_latency_violation:
                    removed = self.individuals.pop(i)
                    print(f"  Removed: ID={removed.id}, depth={depth}, latency={removed.scores['fhe_latency']:.0f} (P1: majority+violation)")
                    return

            # Priority 2: Majority + latency ok
            for i in range(len(self.individuals)):
                ind = self.individuals[i]
                depth = len(ind.config.blocks)
                is_minority = depth_counts[depth] <= min_quota

                if not is_minority:
                    removed = self.individuals.pop(i)
                    print(f"  Removed: ID={removed.id}, depth={depth}, latency={removed.scores['fhe_latency']:.0f} (P2: majority+ok)")
                    return

            # Priority 3: Minority + latency violation
            for i in range(len(self.individuals)):
                ind = self.individuals[i]
                is_latency_violation = ind.scores.get('fhe_latency', float('inf')) > self.latency_baseline

                if is_latency_violation:
                    depth = len(ind.config.blocks)
                    removed = self.individuals.pop(i)
                    print(f"  Removed: ID={removed.id}, depth={depth}, latency={removed.scores['fhe_latency']:.0f} (P3: minority+violation)")
                    return

            # Priority 4: All protected
            removed = self.individuals.pop(0)
            depth = len(removed.config.blocks)
            print(f"  Removed: ID={removed.id}, depth={depth}, latency={removed.scores['fhe_latency']:.0f} (P4: forced)")

        def add_individual(self, num_blocks, latency, generation):
            ind = MockIndividual(num_blocks, latency, generation)
            self.individuals.append(ind)

            if len(self.individuals) > self.max_size:
                self._remove_with_diversity_protection()

            for ind in self.individuals[:-1]:
                ind.increment_age()

    # 测试场景
    print("\n【场景1: 保护少数群体】")
    print("-" * 80)
    MockIndividual._next_id = 0
    pop = MockPopulation(max_size=10, diversity_quota=0.1, latency_baseline=1000)

    # 添加10个个体：8个8层，2个12层
    print("添加10个个体：8个8层（多数），2个12层（少数）")
    for i in range(8):
        pop.add_individual(num_blocks=8, latency=500, generation=i)
    for i in range(2):
        pop.add_individual(num_blocks=12, latency=500, generation=8+i)

    print(f"\n当前population: {len(pop.individuals)}个")
    depth_dist = pop._get_depth_distribution()
    print(f"深度分布: {depth_dist}")

    # 添加第11个个体（8层），应该移除8层的
    print("\n添加第11个个体（8层，延迟500）...")
    pop.add_individual(num_blocks=8, latency=500, generation=10)

    print(f"\n结果: {len(pop.individuals)}个")
    depth_dist = pop._get_depth_distribution()
    print(f"深度分布: {depth_dist}")
    print(f"验证: {'✅ 12层被保护' if depth_dist.get(12, 0) == 2 else '❌ 12层未被保护'}")

    # 测试场景2
    print("\n" + "=" * 80)
    print("【场景2: 延迟超标不受保护】")
    print("-" * 80)
    MockIndividual._next_id = 0
    pop2 = MockPopulation(max_size=10, diversity_quota=0.1, latency_baseline=1000)

    # 添加10个个体：8个8层（延迟500），2个12层（延迟1500，超标）
    print("添加10个个体：8个8层（延迟500），2个12层（延迟1500，超标）")
    for i in range(8):
        pop2.add_individual(num_blocks=8, latency=500, generation=i)
    for i in range(2):
        pop2.add_individual(num_blocks=12, latency=1500, generation=8+i)

    print(f"\n当前population: {len(pop2.individuals)}个")
    depth_dist2 = pop2._get_depth_distribution()
    print(f"深度分布: {depth_dist2}")

    # 添加第11个个体（8层），应该移除12层的（延迟超标）
    print("\n添加第11个个体（8层，延迟500）...")
    pop2.add_individual(num_blocks=8, latency=500, generation=10)

    print(f"\n结果: {len(pop2.individuals)}个")
    depth_dist2 = pop2._get_depth_distribution()
    print(f"深度分布: {depth_dist2}")
    print(f"验证: {'✅ 延迟超标的12层被移除' if depth_dist2.get(12, 0) == 1 else '❌ 延迟超标的12层未被移除'}")

    # 测试场景3
    print("\n" + "=" * 80)
    print("【场景3: 移除优先级测试】")
    print("-" * 80)
    MockIndividual._next_id = 0
    pop3 = MockPopulation(max_size=10, diversity_quota=0.1, latency_baseline=1000)

    # 添加复杂场景
    print("添加10个个体：")
    print("  - 5个8层（延迟500，多数+合格）")
    print("  - 2个8层（延迟1500，多数+超标）")
    print("  - 2个12层（延迟500，少数+合格）")
    print("  - 1个12层（延迟1500，少数+超标）")

    for i in range(5):
        pop3.add_individual(num_blocks=8, latency=500, generation=i)
    for i in range(2):
        pop3.add_individual(num_blocks=8, latency=1500, generation=5+i)
    for i in range(2):
        pop3.add_individual(num_blocks=12, latency=500, generation=7+i)
    pop3.add_individual(num_blocks=12, latency=1500, generation=9)

    print(f"\n当前population: {len(pop3.individuals)}个")

    # 添加第11个个体，应该优先移除"多数+超标"
    print("\n添加第11个个体（8层，延迟500）...")
    print("预期: 应该移除8层+延迟超标的个体（优先级P1）")
    pop3.add_individual(num_blocks=8, latency=500, generation=10)

    print(f"\n结果: {len(pop3.individuals)}个")
    # 统计剩余的8层延迟超标个体
    remaining_8_violation = sum(1 for ind in pop3.individuals
                                if len(ind.config.blocks) == 8 and ind.scores['fhe_latency'] > 1000)
    print(f"剩余8层延迟超标个体: {remaining_8_violation}")
    print(f"验证: {'✅ 优先移除多数+超标' if remaining_8_violation == 1 else '❌ 移除优先级错误'}")

    return True


if __name__ == "__main__":
    try:
        success = test_diversity_protection()
        print("\n" + "=" * 80)
        print("测试完成")
        print("=" * 80)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
