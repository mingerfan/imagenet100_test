"""
主训练脚本
支持多GPU并行训练和增量训练
"""

import argparse
import os
import sys
from trainers import MultiGPUManager
from utils import load_config, get_model_configs


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ImageNet-100 多模型训练脚本')
    
    # 配置文件
    parser.add_argument(
        '--config',
        type=str,
        default='configs/models_list.yaml',
        help='模型配置文件路径'
    )
    
    # 数据集路径
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/home/xuming/Documents/dataset/ImageNet_100/train',
        help='训练集目录'
    )
    parser.add_argument(
        '--val_dir',
        type=str,
        default='/home/xuming/Documents/dataset/ImageNet_100/val',
        help='验证集目录'
    )
    
    # 结果目录
    parser.add_argument(
        '--result_dir',
        type=str,
        default='./results',
        help='结果保存目录'
    )
    
    # GPU设置
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3],
        help='使用的GPU设备ID列表，例如: --gpus 0 1 2 3'
    )
    
    # 训练选项
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新训练所有模型（包括已训练的）'
    )
    parser.add_argument(
        '--no_parallel',
        action='store_true',
        help='禁用并行训练，使用串行模式'
    )
    parser.add_argument(
        '--no_cache',
        action='store_true',
        help='不使用内存缓存数据集'
    )
    
    # 选择特定模型
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='只训练指定的模型，例如: --models resnet18 resnet34'
    )
    
    # 其他参数
    parser.add_argument(
        '--num_classes',
        type=int,
        default=100,
        help='类别数量'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=16,
        help='数据加载worker数量'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("ImageNet-100 多模型训练系统")
    print("=" * 60)
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"⚠ 配置文件不存在: {args.config}")
        print("将使用默认配置")
        config = {'models': []}
    else:
        print(f"\n加载配置文件: {args.config}")
        config = load_config(args.config)
    
    # 获取模型配置
    model_configs = get_model_configs(config)
    
    # 过滤特定模型（如果指定）
    if args.models:
        print(f"\n只训练指定的模型: {args.models}")
        model_configs = [m for m in model_configs if m['name'] in args.models]
        if not model_configs:
            print(f"⚠ 未找到指定的模型: {args.models}")
            print(f"可用的模型: {[m['name'] for m in get_model_configs(config)]}")
            sys.exit(1)
    
    # 显示将要训练的模型
    print(f"\n将要训练的模型: {len(model_configs)} 个")
    for i, model_config in enumerate(model_configs, 1):
        print(f"  {i}. {model_config['name']}")
        print(f"     - Epochs: {model_config.get('epochs', 60)}")
        print(f"     - Batch Size: {model_config.get('batch_size', 128)}")
        print(f"     - Learning Rate: {model_config.get('learning_rate', 0.001)}")
        print(f"     - Pretrained: {model_config.get('params', {}).get('pretrained', True)}")
    
    # 检查数据集目录
    if not os.path.exists(args.train_dir):
        print(f"\n⚠ 训练集目录不存在: {args.train_dir}")
        sys.exit(1)
    if not os.path.exists(args.val_dir):
        print(f"\n⚠ 验证集目录不存在: {args.val_dir}")
        sys.exit(1)
    
    # 创建多GPU管理器
    print(f"\n创建多GPU训练管理器...")
    manager = MultiGPUManager(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        result_dir=args.result_dir,
        gpus=args.gpus,
        num_classes=args.num_classes,
        default_num_workers=args.num_workers,
        use_cache=not args.no_cache
    )
    
    # 训练模型
    print(f"\n{'=' * 60}")
    print(f"开始训练 - 并行模式: {'否' if args.no_parallel else '是'}")
    print(f"强制重新训练: {'是' if args.force else '否'}")
    print(f"{'=' * 60}")
    
    results = manager.train_models(
        model_configs=model_configs,
        force=args.force,
        parallel=not args.no_parallel
    )
    
    # 最终总结
    print(f"\n{'=' * 60}")
    print("训练完成总结")
    print(f"{'=' * 60}")
    
    if results:
        print(f"\n成功训练的模型: {len(results)} 个")
        for model_name, acc in results.items():
            print(f"  ✓ {model_name}: {acc:.2f}%")
    else:
        print("\n没有模型被训练（可能都已存在）")
        if not args.force:
            print("提示: 使用 --force 参数可以强制重新训练所有模型")
    
    # 统计跳过的模型
    skipped = [m['name'] for m in model_configs 
              if m['name'] not in results and not args.force]
    if skipped:
        print(f"\n跳过的模型（已存在结果）: {len(skipped)} 个")
        for model_name in skipped:
            print(f"  - {model_name}")
    
    print(f"\n结果保存在: {args.result_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()