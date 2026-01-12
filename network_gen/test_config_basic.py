#!/usr/bin/env python3
"""
简化的配置系统测试

只测试配置文件的加载和验证，不需要torch
"""

import sys
import os
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_yaml_files():
    """测试YAML配置文件是否存在和格式正确"""
    print("=" * 60)
    print("测试1: YAML配置文件")
    print("=" * 60)

    configs_dir = Path("network_gen/configs")

    # 检查配置文件是否存在
    expected_files = [
        "imagenet_224.yaml",
        "cifar10_32.yaml",
        "README.md",
    ]

    print("\n检查配置文件...")
    for filename in expected_files:
        filepath = configs_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename} 存在")
        else:
            print(f"  ✗ {filename} 不存在")
            return False

    print("\n✓ 所有配置文件都存在")
    return True


def test_yaml_loading():
    """测试YAML文件能否正确加载"""
    print("\n" + "=" * 60)
    print("测试2: YAML文件加载")
    print("=" * 60)

    try:
        import yaml

        configs = [
            "network_gen/configs/imagenet_224.yaml",
            "network_gen/configs/cifar10_32.yaml",
        ]

        for config_path in configs:
            print(f"\n加载 {config_path}...")
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # 检查必需字段
            required_fields = ['name', 'description', 'dataset', 'search_space', 'output']
            for field in required_fields:
                if field not in data:
                    print(f"  ✗ 缺少字段: {field}")
                    return False
                print(f"  ✓ {field}: {data[field] if field in ['name', 'description'] else '...'}")

        print("\n✓ YAML文件加载成功")
        return True

    except Exception as e:
        print(f"\n✗ YAML文件加载失败: {e}")
        return False


def test_config_structure():
    """测试配置文件结构"""
    print("\n" + "=" * 60)
    print("测试3: 配置文件结构")
    print("=" * 60)

    try:
        import yaml

        # 测试ImageNet配置
        print("\n检查ImageNet配置结构...")
        with open("network_gen/configs/imagenet_224.yaml", 'r', encoding='utf-8') as f:
            imagenet_config = yaml.safe_load(f)

        assert imagenet_config['name'] == 'imagenet_224'
        assert imagenet_config['dataset']['input_size'] == 224
        assert imagenet_config['dataset']['num_classes'] == 100
        assert imagenet_config['search_space']['stem']['enabled'] == True
        print("  ✓ ImageNet配置结构正确")

        # 测试CIFAR-10配置
        print("\n检查CIFAR-10配置结构...")
        with open("network_gen/configs/cifar10_32.yaml", 'r', encoding='utf-8') as f:
            cifar10_config = yaml.safe_load(f)

        assert cifar10_config['name'] == 'cifar10_32'
        assert cifar10_config['dataset']['input_size'] == 32
        assert cifar10_config['dataset']['num_classes'] == 10
        assert cifar10_config['search_space']['stem']['enabled'] == False
        assert cifar10_config['search_space']['stride']['num_strides'] == 2

        # 检查前两层约束
        first_layers = cifar10_config['search_space']['blocks']['first_layers_constraints']
        assert first_layers is not None
        assert len(first_layers) == 2
        assert first_layers[0]['position'] == 0
        assert first_layers[0]['stride'] == 1
        assert first_layers[1]['position'] == 1
        assert first_layers[1]['stride'] == 1
        print("  ✓ CIFAR-10配置结构正确")
        print("  ✓ 前两层stride约束正确")

        print("\n✓ 配置文件结构验证通过")
        return True

    except Exception as e:
        print(f"\n✗ 配置文件结构验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """测试目录结构"""
    print("\n" + "=" * 60)
    print("测试4: 目录结构")
    print("=" * 60)

    expected_files = [
        "network_gen/generator_config.py",
        "network_gen/network_generator.py",
        "network_gen/network_config.py",
        "network_gen/search_space.py",
        "network_gen/batch_generator.py",
        "network_gen/configs/README.md",
        "network_gen/USAGE.md",
    ]

    print("\n检查文件...")
    all_exist = True
    for filepath in expected_files:
        if Path(filepath).exists():
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} 不存在")
            all_exist = False

    if all_exist:
        print("\n✓ 所有文件都存在")
    else:
        print("\n✗ 部分文件缺失")

    return all_exist


def main():
    """运行所有测试"""
    print("开始测试配置系统（简化版，不需要torch）...")
    print()

    results = []

    try:
        results.append(("YAML文件存在性", test_yaml_files()))
        results.append(("YAML文件加载", test_yaml_loading()))
        results.append(("配置文件结构", test_config_structure()))
        results.append(("目录结构", test_directory_structure()))

        print("\n" + "=" * 60)
        print("测试结果摘要")
        print("=" * 60)

        all_passed = True
        for test_name, passed in results:
            status = "✓ 通过" if passed else "✗ 失败"
            print(f"{test_name:20s}: {status}")
            if not passed:
                all_passed = False

        print("=" * 60)

        if all_passed:
            print("✓ 所有测试通过!")
            print("\n提示: 完整的功能测试需要安装torch，请运行:")
            print("  python network_gen/test_config_system.py")
        else:
            print("✗ 部分测试失败!")
            sys.exit(1)

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ 测试过程出错!")
        print("=" * 60)
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
