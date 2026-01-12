#!/usr/bin/env python3
"""
测试配置系统

验证配置文件加载和网络生成是否正常工作
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_gen.generator_config import (
    GeneratorConfig,
    ConfigManager,
    create_default_imagenet_config,
    create_default_cifar10_config,
)
from network_gen.network_generator import RandomNetworkGenerator


def test_config_loading():
    """测试配置文件加载"""
    print("=" * 60)
    print("测试1: 配置文件加载")
    print("=" * 60)

    # 测试ImageNet配置
    print("\n加载ImageNet配置...")
    imagenet_config = GeneratorConfig.from_yaml("network_gen/configs/imagenet_224.yaml")
    print(imagenet_config.summary())

    # 测试CIFAR-10配置
    print("\n加载CIFAR-10配置...")
    cifar10_config = GeneratorConfig.from_yaml("network_gen/configs/cifar10_32.yaml")
    print(cifar10_config.summary())

    print("\n✓ 配置文件加载成功")


def test_generator_with_config():
    """测试使用配置生成网络"""
    print("\n" + "=" * 60)
    print("测试2: 使用配置生成网络")
    print("=" * 60)

    # 加载ImageNet配置
    config = GeneratorConfig.from_yaml("network_gen/configs/imagenet_224.yaml")

    # 创建生成器
    generator = RandomNetworkGenerator(config=config, seed=42)

    # 生成单个网络
    print("\n生成单个网络配置...")
    network_config = generator.generate_random_config()
    print(network_config.summary())

    print("\n✓ 网络生成成功")


def test_cifar10_constraints():
    """测试CIFAR-10约束是否生效"""
    print("\n" + "=" * 60)
    print("测试3: CIFAR-10约束验证")
    print("=" * 60)

    # 加载CIFAR-10配置
    config = GeneratorConfig.from_yaml("network_gen/configs/cifar10_32.yaml")

    # 创建生成器
    generator = RandomNetworkGenerator(config=config, seed=42)

    # 生成多个网络并验证约束
    print("\n生成5个网络并验证约束...")
    for i in range(5):
        network_config = generator.generate_random_config()

        # 检查前两层的stride
        stride_0 = network_config.blocks[0].stride
        stride_1 = network_config.blocks[1].stride

        print(f"\n网络 {i+1}: {network_config.name}")
        print(f"  Block数量: {network_config.num_blocks}")
        print(f"  第1个block stride: {stride_0} (应该为1)")
        print(f"  第2个block stride: {stride_1} (应该为1)")

        # 验证约束
        assert stride_0 == 1, f"第1个block的stride应该为1，但是是{stride_0}"
        assert stride_1 == 1, f"第2个block的stride应该为1，但是是{stride_1}"
        assert network_config.num_blocks in [6, 8, 10, 12], \
            f"Block数量应该在[6,8,10,12]中，但是是{network_config.num_blocks}"

    print("\n✓ CIFAR-10约束验证通过")


def test_config_manager():
    """测试配置管理器"""
    print("\n" + "=" * 60)
    print("测试4: 配置管理器")
    print("=" * 60)

    # 创建测试配置
    config = create_default_imagenet_config()
    config.output.base_dir = "/tmp/test_network_gen/imagenet_test"

    # 创建配置管理器
    manager = ConfigManager(config)
    print(f"\n输出目录: {manager.output_dir}")

    # 生成一个网络
    generator = RandomNetworkGenerator(config=config, seed=42)
    network_config = generator.generate_random_config()

    # 保存网络配置
    print("\n保存网络配置...")
    path = manager.save_network_config(network_config, overwrite=True)
    print(f"已保存到: {path}")

    # 加载网络配置
    print("\n加载网络配置...")
    loaded_config = manager.load_network_config(network_config.name)
    print(f"已加载: {loaded_config.name}")

    # 验证
    assert loaded_config.name == network_config.name
    assert len(loaded_config.blocks) == len(network_config.blocks)

    print("\n配置管理器摘要:")
    print(manager.summary())

    print("\n✓ 配置管理器测试通过")


def test_batch_generation():
    """测试批量生成"""
    print("\n" + "=" * 60)
    print("测试5: 批量生成")
    print("=" * 60)

    # 使用CIFAR-10配置
    config = GeneratorConfig.from_yaml("network_gen/configs/cifar10_32.yaml")
    config.output.base_dir = "/tmp/test_network_gen/cifar10_test"

    # 创建生成器和管理器
    generator = RandomNetworkGenerator(config=config, seed=42)
    manager = ConfigManager(config)

    # 生成批量配置
    print("\n生成10个网络配置...")
    batch = generator.generate_batch(
        num_configs=10,
        batch_name="test_batch",
        description="Test batch",
        unique=True,
    )

    print(f"成功生成 {len(batch)} 个配置")
    print(batch.summary())

    # 保存批量配置
    print("\n保存批量配置...")
    batch_path = manager.save_batch(batch, overwrite=True)
    print(f"已保存到: {batch_path}")

    # 加载批量配置
    print("\n加载批量配置...")
    loaded_batch = manager.load_batch("test_batch")
    print(f"已加载 {len(loaded_batch)} 个配置")

    # 验证
    assert len(loaded_batch) == len(batch)

    print("\n✓ 批量生成测试通过")


def main():
    """运行所有测试"""
    print("开始测试配置系统...")
    print()

    try:
        test_config_loading()
        test_generator_with_config()
        test_cifar10_constraints()
        test_config_manager()
        test_batch_generation()

        print("\n" + "=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ 测试失败!")
        print("=" * 60)
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
