"""
主训练脚本
支持多GPU并行训练和增量训练
"""

import argparse
import os
import sys
from trainers import MultiGPUManager
from utils import load_config, get_model_configs, set_random_seed
from models import MODEL_REGISTRY
from data import get_dataset_info, normalize_dataset_name


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多模型训练脚本')
    
    # 配置文件
    parser.add_argument(
        '--config',
        type=str,
        default='configs/models_list.yaml',
        help='模型配置文件路径'
    )
    
    # 数据集类型
    parser.add_argument(
        '--dataset',
        type=str,
        default='imagenet100',
        help='数据集类型: imagenet100/imagenet1k/cifar10/cifar100'
    )

    # 数据集路径
    parser.add_argument(
        '--train_dir',
        type=str,
        default=None,
        help='训练集目录（ImageFolder）或 CIFAR 根目录'
    )
    parser.add_argument(
        '--val_dir',
        type=str,
        default=None,
        help='验证集目录（ImageFolder）或 CIFAR 根目录'
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
        '--no_checkpoint',
        dest='save_checkpoints',
        action='store_false',
        help='禁用检查点保存（仅保存最佳模型）'
    )
    parser.set_defaults(save_checkpoints=True)
    parser.add_argument(
        '--use_memory_fs',
        action='store_true',
        default=True,
        help='使用内存文件系统（/dev/shm）加速数据加载（推荐，避免并发内存问题）'
    )
    parser.add_argument(
        '--no_memory_fs',
        dest='use_memory_fs',
        action='store_false',
        help='禁用内存文件系统'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='允许下载数据集（仅 CIFAR 有效）'
    )
    parser.add_argument(
        '--input_size',
        type=int,
        default=None,
        help='输入图像大小（可选，覆盖默认值）'
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
        default=None,
        help='类别数量'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        type=int,
        default=0 if os.name == 'nt' else 16,
        help='数据加载worker数量'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    set_random_seed(args.seed)

    dataset_name = normalize_dataset_name(args.dataset)
    dataset_info = get_dataset_info(dataset_name)
    
    print("=" * 60)
    print("多模型训练系统")
    print("=" * 60)
    
    # 设置默认数据路径
    if args.train_dir is None:
        if dataset_name == 'imagenet100':
            args.train_dir = '/home/xuming/Documents/dataset/ImageNet_100/train'
        elif dataset_name in ('cifar10', 'cifar100'):
            args.train_dir = './data'
        else:
            print("⚠ ImageNet-1k 需要显式指定 --train_dir")
            sys.exit(1)

    if args.val_dir is None:
        if dataset_name == 'imagenet100':
            args.val_dir = '/home/xuming/Documents/dataset/ImageNet_100/val'
        elif dataset_name in ('cifar10', 'cifar100'):
            args.val_dir = args.train_dir
        else:
            print("⚠ ImageNet-1k 需要显式指定 --val_dir")
            sys.exit(1)

    if args.num_classes is None:
        args.num_classes = dataset_info['num_classes']

    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"⚠ 配置文件不存在: {args.config}")
        print("将使用默认配置")
        config = {'models': []}
    else:
        print(f"\n加载配置文件: {args.config}")
        config = load_config(args.config)
    
    # 获取已注册的模型列表（用于正则匹配）
    registered_models = MODEL_REGISTRY.list_models()
    print(f"\n已注册的模型: {len(registered_models)} 个")
    for model_name in registered_models:
        print(f"  - {model_name}")
    
    # 获取模型配置（传入已注册模型列表以支持正则匹配）
    model_configs = get_model_configs(config, registered_models)
    
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
    
    # 检查数据集目录（CIFAR 可通过下载创建）
    if dataset_info['type'] == 'imagefolder':
        if not os.path.exists(args.train_dir):
            print(f"\n⚠ 训练集目录不存在: {args.train_dir}")
            sys.exit(1)
        if not os.path.exists(args.val_dir):
            print(f"\n⚠ 验证集目录不存在: {args.val_dir}")
            sys.exit(1)
    elif not os.path.exists(args.train_dir) and not args.download:
        print(f"\n⚠ CIFAR 根目录不存在: {args.train_dir}")
        print("提示: 使用 --download 允许自动下载")
        sys.exit(1)
    
    # 应用save_checkpoints到所有模型配置
    for model_config in model_configs:
        model_config['save_checkpoints'] = args.save_checkpoints
    
    if not args.save_checkpoints:
        print("\n⚠ 检查点保存已禁用（仅保存最佳模型）")
    
    # 创建多GPU管理器
    print(f"\n创建多GPU训练管理器...")
    manager = MultiGPUManager(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        result_dir=args.result_dir,
        gpus=args.gpus,
        num_classes=args.num_classes,
        default_num_workers=args.num_workers,
        use_memory_fs=args.use_memory_fs,
        dataset=dataset_name,
        download=args.download,
        input_size=args.input_size,
        seed=args.seed
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
    
    # 最终总结（MultiGPUManager已经打印了详细信息）
    print(f"\n{'=' * 60}")
    print("训练完成")
    print(f"{'=' * 60}")
    
    total = len(model_configs)
    success_count = len(results['success'])
    failed_count = len(results['failed'])
    skipped_count = len(results['skipped'])
    
    if success_count == 0 and failed_count == 0 and skipped_count > 0:
        print("\n所有模型都已存在训练结果")
        if not args.force:
            print("提示: 使用 --force 参数可以强制重新训练所有模型")
    
    if failed_count > 0:
        print(f"\n⚠ 注意: {failed_count} 个模型训练失败，详见上方错误信息")
    
    print(f"\n结果保存在: {args.result_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
