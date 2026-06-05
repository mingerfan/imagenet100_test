"""
主训练脚本
支持多GPU并行训练和增量训练
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from trainers import MultiGPUManager
from utils import (
    format_gpu_ids_with_physical,
    format_visible_gpu_mapping,
    load_config,
    get_model_configs,
    get_json_model_configs,
    resolve_gpu_selection,
    set_random_seed,
)
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
        nargs='+',
        default=None,
        help=(
            '使用的GPU设备ID列表/范围，默认使用所有可见GPU并避开physical GPU 0；'
            '例如: --gpus 1 2 3, --gpus 0-7, --gpus all'
        )
    )
    parser.add_argument(
        '--allow_gpu0',
        action='store_true',
        help='允许使用physical GPU 0（不推荐：该卡存在memory/ECC风险）'
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
        '--save_freq',
        type=int,
        default=None,
        help='Checkpoint save frequency (epochs)'
    )
    parser.add_argument(
        '--use_amp',
        action='store_true',
        default=None,
        help='Enable automatic mixed precision'
    )
    parser.add_argument(
        '--no_use_amp',
        dest='use_amp',
        action='store_false',
        help='Disable automatic mixed precision'
    )
    parser.add_argument(
        '--val_fp32',
        dest='val_force_fp32',
        action='store_true',
        default=None,
        help='Force FP32 in validation (disable autocast)'
    )
    parser.add_argument(
        '--no_val_fp32',
        dest='val_force_fp32',
        action='store_false',
        help='Allow AMP in validation'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint if available'
    )
    parser.add_argument(
        '--resume_mode',
        type=str,
        default='auto',
        choices=['auto', 'best', 'last'],
        help='Checkpoint selection mode'
    )
    parser.add_argument(
        '--resume_path',
        type=str,
        default=None,
        help='Explicit checkpoint path to resume from'
    )
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
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    dataset_name = normalize_dataset_name(args.dataset)
    dataset_info = get_dataset_info(dataset_name)
    
    print("=" * 60)
    print("多模型训练系统")
    print("=" * 60)

    try:
        gpu_selection = resolve_gpu_selection(args.gpus, allow_gpu0=args.allow_gpu0)
    except ValueError as exc:
        print(f"❌ GPU参数错误: {exc}")
        sys.exit(1)
    args.gpus = gpu_selection.selected

    if gpu_selection.skipped and not args.allow_gpu0:
        skipped = format_gpu_ids_with_physical(
            gpu_selection.skipped,
            gpu_selection.visible_to_physical,
        )
        print(f"⚠ 已过滤 physical GPU 0: {skipped}；如确需使用请加 --allow_gpu0")
        if not args.gpus:
            print("❌ 过滤 GPU0 后没有剩余 GPU。请指定非 GPU0 的设备，或显式 --allow_gpu0。")
            sys.exit(1)
    print(f"请求GPU: {gpu_selection.requested}")
    print(f"实际GPU: {format_gpu_ids_with_physical(args.gpus, gpu_selection.visible_to_physical)}")
    print(f"CUDA_VISIBLE_DEVICES: {gpu_selection.cuda_visible_devices}")
    print(f"可见GPU映射: {format_visible_gpu_mapping(gpu_selection.visible_to_physical)}")

    # 设置默认数据路径
    if args.train_dir is None:
        if dataset_name == 'imagenet100':
            candidates = [
                '/home/xuming/Documents/dataset/imagenet_100/train',
                '/home/xuming/Documents/dataset/ImageNet_100/train',
            ]
            args.train_dir = next((path for path in candidates if os.path.exists(path)), candidates[0])
        elif dataset_name in ('cifar10', 'cifar100'):
            args.train_dir = './data'
        else:
            print("⚠ ImageNet-1k 需要显式指定 --train_dir")
            sys.exit(1)

    if args.val_dir is None:
        if dataset_name == 'imagenet100':
            candidates = [
                '/home/xuming/Documents/dataset/imagenet_100/val',
                '/home/xuming/Documents/dataset/ImageNet_100/val',
            ]
            args.val_dir = next((path for path in candidates if os.path.exists(path)), candidates[0])
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
        # Use per-config result subfolder
        config_tag = Path(args.config).stem
        args.result_dir = os.path.join(args.result_dir, config_tag)
    
    # 获取已注册的模型列表（用于正则匹配）
    registered_models = MODEL_REGISTRY.list_models()
    print(f"\n已注册的模型: {len(registered_models)} 个")
    for model_name in registered_models:
        print(f"  - {model_name}")
    
    # 获取模型配置（传入已注册模型列表以支持正则匹配）
    model_configs = get_model_configs(config, registered_models)
    json_model_configs = get_json_model_configs(config)
    if json_model_configs:
        model_configs.extend(json_model_configs)
    
    # 过滤特定模型（如果指定）
    if args.models:
        print(f"\n只训练指定的模型: {args.models}")
        model_configs = [m for m in model_configs if m['name'] in args.models]
        if not model_configs:
            all_candidates = get_model_configs(config, registered_models)
            all_candidates.extend(get_json_model_configs(config))
            print(f"⚠ 未找到指定的模型: {args.models}")
            print(f"可用的模型: {[m['name'] for m in all_candidates]}")
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
    if args.resume_path and len(model_configs) > 1:
        print("Warning: resume_path is set for multiple models; the same path will be applied to all models.")

    for model_config in model_configs:
        model_config['save_checkpoints'] = args.save_checkpoints
        if args.save_freq is not None:
            model_config['save_freq'] = args.save_freq
        if args.use_amp is not None:
            model_config['use_amp'] = args.use_amp
        if args.val_force_fp32 is not None:
            model_config['val_force_fp32'] = args.val_force_fp32
        if args.resume:
            model_config['resume'] = True
        if args.resume_mode:
            model_config['resume_mode'] = args.resume_mode
        if args.resume_path:
            model_config['resume_path'] = args.resume_path
    
    if not args.save_checkpoints:
        print("\n⚠ 检查点保存已禁用（仅保存最佳模型）")
    
    # 创建多GPU管理器
    print(f"\n创建多GPU训练管理器...")
    manager = MultiGPUManager(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        result_dir=args.result_dir,
        gpus=args.gpus,
        excluded_gpus=[] if args.allow_gpu0 else [0],
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
    if failed_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
