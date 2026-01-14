#!/usr/bin/env python3
"""
测试Population年龄管理修复（不依赖torch）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_population_age_management():
    """测试Population年龄管理是否正确"""
    print("=" * 80)
    print("测试: Population年龄管理修复验证")
    print("=" * 80)

    # 手动创建简化的Individual和Population类来测试
    class SimpleIndividual:
        _next_id = 0

        def __init__(self, generation):
            self.id = SimpleIndividual._next_id
            SimpleIndividual._next_id += 1
            self.generation = generation
            self.age = 0

        def increment_age(self):
            self.age += 1

    class SimplePopulation:
        def __init__(self, max_size):
            self.max_size = max_size
            self.individuals = []

        def add_old_version(self, generation):
            """旧版本（有bug）"""
            individual = SimpleIndividual(generation)
            self.individuals.append(individual)

            if len(self.individuals) > self.max_size:
                self.individuals.pop(0)

            # ❌ Bug: 包括新添加的个体
            for ind in self.individuals:
                ind.increment_age()

        def add_new_version(self, generation):
            """新版本（已修复）"""
            individual = SimpleIndividual(generation)
            self.individuals.append(individual)

            if len(self.individuals) > self.max_size:
                self.individuals.pop(0)

            # ✅ Fix: 只递增现有个体，排除新添加的
            for ind in self.individuals[:-1]:
                ind.increment_age()

    # 测试旧版本
    print("\n【旧版本 - 有Bug】")
    print("-" * 80)
    pop_old = SimplePopulation(max_size=5)

    for i in range(3):
        pop_old.add_old_version(generation=i)

    print("添加3个个体后的年龄:")
    for i, ind in enumerate(pop_old.individuals):
        print(f"  Individual {i}: age={ind.age}, generation={ind.generation}")

    old_ages = [ind.age for ind in pop_old.individuals]
    print(f"\n实际年龄: {old_ages}")
    print(f"预期年龄: [2, 1, 0]")
    print(f"结果: {'❌ 错误 - 最新个体年龄应该是0，但实际是1' if old_ages[-1] != 0 else '✅ 正确'}")

    # 测试新版本
    print("\n【新版本 - 已修复】")
    print("-" * 80)
    SimpleIndividual._next_id = 0  # 重置ID
    pop_new = SimplePopulation(max_size=5)

    for i in range(3):
        pop_new.add_new_version(generation=i)

    print("添加3个个体后的年龄:")
    for i, ind in enumerate(pop_new.individuals):
        print(f"  Individual {i}: age={ind.age}, generation={ind.generation}")

    new_ages = [ind.age for ind in pop_new.individuals]
    expected_ages = [2, 1, 0]
    print(f"\n实际年龄: {new_ages}")
    print(f"预期年龄: {expected_ages}")
    match = new_ages == expected_ages
    print(f"结果: {'✅ 正确' if match else '❌ 错误'}")

    # 测试FIFO移除
    print("\n【测试FIFO移除机制】")
    print("-" * 80)
    print("继续添加2个个体（触发FIFO移除）...")

    for i in range(2):
        pop_new.add_new_version(generation=3+i)

    print("\n当前population (max_size=5):")
    for i, ind in enumerate(pop_new.individuals):
        print(f"  Individual {i}: id={ind.id}, age={ind.age}, generation={ind.generation}")

    newest_age = pop_new.individuals[-1].age
    print(f"\n最新个体年龄: {newest_age}")
    print(f"预期: 0")
    print(f"结果: {'✅ 正确' if newest_age == 0 else '❌ 错误'}")

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"旧版本（有bug）: ❌ 新个体年龄被错误递增")
    print(f"新版本（已修复）: {'✅ 通过' if match and newest_age == 0 else '❌ 失败'}")

    return match and newest_age == 0


if __name__ == "__main__":
    success = test_population_age_management()
    sys.exit(0 if success else 1)
