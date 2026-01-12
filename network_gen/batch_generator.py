#!/usr/bin/env python3
"""
批量网络生成脚本

用于批量生成网络配置，支持：
1. 随机生成指定数量的网络
2. 通过配置文件控制生成约束
3. 不同配置文件生成的网络保存到不同文件夹
4. 保存配置到JSON文件
5. 验证生成的网络可以正确构建
6. 输出网络统计信息
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_gen.search_space import SearchSpace
from network_gen.network_config import NetworkConfig, NetworkConfigBatch
from network_gen.network_generator import (
    RandomNetworkGenerator,
    NetworkBuilder,
    create_network,
)
from network_gen.generator_config import GeneratorConfig, ConfigManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量生成FHE-NAS网络配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用配置文件生成
  python batch_generator.py --config configs/imagenet_224.yaml -n 100

  # 使用CIFAR-10配置
  python batch_generator.py --config configs/cifar10_32.yaml -n 50

  # 使用命令行参数生成（向后兼容）
  python batch_generator.py -n 50 --input-size 224
        """
    )

    # 配置文件参数（新增）
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="生成器配置文件路径 (YAML格式)"
    )

    parser.add_argument(
        "-n", "--num",
        type=int,
        default=50,
        help="生成的网络数量 (默认: 50)"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出目录 (默认: 从配置文件读取，或使用 ./generated_networks)"
    )

    parser.add_argument(
        "--batch-name",
        type=str,
        default=None,
        help="批次名称 (默认: 使用时间戳)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="验证生成的网络可以正确构建和前向传播"
    )

    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="除了批量文件外，也保存每个配置的单独JSON文件"
    )

    # 以下参数仅在不使用配置文件时生效
    parser.add_argument(
        "--ct-slots",
        type=int,
        default=32768,
        help="CT槽位数 (默认: 32768，仅在不使用配置文件时有效)"
    )

    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="输入图像大小 (默认: 224，仅在不使用配置文件时有效)"
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=100,
        help="分类数量 (默认: 100，仅在不使用配置文件时有效)"
    )

    return parser.parse_args()


def verify_network(config: NetworkConfig, input_size: int = 224) -> dict:
    """
    验证网络可以正确构建和前向传播

    Returns:
        dict: 包含验证结果和网络信息
    """
    result = {
        "name": config.name,
        "success": False,
        "error": None,
        "num_params": 0,
        "output_shape": None,
    }

    try:
        # 构建网络
        model = create_network(config)
        model.eval()

        # 计算参数量
        result["num_params"] = sum(p.numel() for p in model.parameters())

        # 测试前向传播
        x = torch.randn(1, 3, input_size, input_size)
        with torch.no_grad():
            y = model(x)
        result["output_shape"] = list(y.shape)

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    args = parse_args()

    # 加载或创建配置
    if args.config:
        # 使用配置文件
        print(f"加载配置文件: {args.config}")
        gen_config = GeneratorConfig.from_yaml(args.config)
        print(gen_config.summary())

        # 输出目录：优先使用命令行参数，否则使用配置文件
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = gen_config.get_output_dir()

        input_size = gen_config.dataset.input_size
        num_classes = gen_config.dataset.num_classes

        # 创建配置管理器
        if args.output:
            # 如果命令行指定了输出目录，更新配置
            gen_config.output.base_dir = args.output
        config_manager = ConfigManager(gen_config)

    else:
        # 使用命令行参数（向后兼容）
        print("未指定配置文件，使用命令行参数")
        gen_config = None
        config_manager = None

        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path("./generated_networks")

        input_size = args.input_size
        num_classes = args.num_classes

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 批次名称
    if args.batch_name is None:
        args.batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n" + "=" * 60)
    print("FHE-NAS 批量网络生成器")
    print("=" * 60)
    print(f"生成数量: {args.num}")
    print(f"输出目录: {output_dir}")
    print(f"批次名称: {args.batch_name}")
    print(f"随机种子: {args.seed}")
    if gen_config:
        print(f"使用配置: {gen_config.name}")
        print(f"数据集: {gen_config.dataset}")
    else:
        print(f"CT槽位数: {args.ct_slots}")
        print(f"输入大小: {input_size}")
        print(f"分类数: {num_classes}")
    print("=" * 60)

    # 创建生成器
    if gen_config:
        # 使用配置文件
        generator = RandomNetworkGenerator(
            config=gen_config,
            seed=args.seed,
        )
    else:
        # 使用命令行参数
        search_space = SearchSpace(
            ct_slots=args.ct_slots,
            input_size=input_size,
        )
        print("\n搜索空间信息:")
        print(search_space.summary())

        generator = RandomNetworkGenerator(
            search_space=search_space,
            seed=args.seed,
        )

    # 生成批量配置
    print(f"\n正在生成 {args.num} 个网络配置...")
    batch = generator.generate_batch(
        num_configs=args.num,
        batch_name=args.batch_name,
        description=f"随机生成于 {datetime.now().isoformat()}",
        unique=True,
    )
    print(f"成功生成 {len(batch)} 个配置")

    # 验证网络
    if args.verify:
        print("\n验证网络构建...")
        verify_results = []
        success_count = 0
        total_params = 0

        for i, config in enumerate(batch):
            result = verify_network(config, input_size)
            verify_results.append(result)

            if result["success"]:
                success_count += 1
                total_params += result["num_params"]
                status = f"OK ({result['num_params']:,} params)"
            else:
                status = f"FAILED: {result['error']}"

            print(f"  [{i+1}/{len(batch)}] {config.name}: {status}")

        print(f"\n验证结果: {success_count}/{len(batch)} 成功")
        if success_count > 0:
            avg_params = total_params / success_count
            print(f"平均参数量: {avg_params:,.0f}")

    # 保存批量配置
    if config_manager:
        # 使用配置管理器保存
        batch_path = config_manager.save_batch(batch, overwrite=True)
        print(f"\n批量配置已保存: {batch_path}")
    else:
        # 直接保存到输出目录
        batch_file = output_dir / f"{args.batch_name}.json"
        batch.save(str(batch_file))
        print(f"\n批量配置已保存: {batch_file}")

    # 保存单独的配置文件
    if args.save_individual:
        if config_manager:
            # 使用配置管理器保存
            for i, config in enumerate(batch):
                try:
                    config_manager.save_network_config(config, overwrite=True)
                except Exception as e:
                    print(f"保存配置 {config.name} 时出错: {e}")
            print(f"单独配置已保存到: {config_manager.output_dir}/")
        else:
            # 直接保存到子目录
            individual_dir = output_dir / args.batch_name
            individual_dir.mkdir(parents=True, exist_ok=True)

            for config in batch:
                config_file = individual_dir / f"{config.name}.json"
                config.save(str(config_file))

            print(f"单独配置已保存: {individual_dir}/")

    # 打印批量统计
    print("\n" + batch.summary())

    # 打印配置管理器摘要
    if config_manager:
        print("\n" + config_manager.summary())

    print("\n完成!")


if __name__ == "__main__":
    main()
