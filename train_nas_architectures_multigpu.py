#!/usr/bin/env python3
"""Train NAS architectures on multiple GPUs (via shared training framework)."""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainers import MultiGPUManager
from data import get_dataset_info, normalize_dataset_name
from utils import set_random_seed, load_config
from utils.nas_training import (
    load_nas_architectures,
    build_nas_model_configs,
    build_nas_results,
    save_results,
)


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def apply_yaml_config(args):
    if not args.config:
        return
    if not os.path.exists(args.config):
        print(f"Warning: config file not found: {args.config}")
        return
    config = load_config(args.config) or {}
    if not isinstance(config, dict):
        print("Warning: config is not a mapping; ignoring.")
        return
    training_config = config.get("training", config)
    if not isinstance(training_config, dict):
        print("Warning: training config is not a mapping; ignoring.")
        return
    if "save_checkpoints" in training_config and "--no_checkpoint" not in sys.argv:
        args.save_checkpoints = _coerce_bool(training_config.get("save_checkpoints"))
    if "save_freq" in training_config and "--save_freq" not in sys.argv:
        try:
            args.save_freq = int(training_config.get("save_freq"))
        except (TypeError, ValueError):
            print("Warning: invalid save_freq in config; using CLI/default.")


def parse_args():
    parser = argparse.ArgumentParser(description='Train NAS architectures on multiple GPUs')

    # Config
    parser.add_argument('--config', type=str, default=None,
                       help='Optional YAML config file')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='imagenet100',
                       help='Dataset type: imagenet100/imagenet1k/cifar10/cifar100')

    # Data paths
    parser.add_argument('--nas_results', type=str, default='nas_results',
                       help='NAS results directory')
    parser.add_argument('--train_dir', type=str,
                       default=None,
                       help='Training data directory (ImageFolder) or CIFAR root')
    parser.add_argument('--val_dir', type=str,
                       default=None,
                       help='Validation data directory (ImageFolder) or CIFAR root')

    # Training settings
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=0 if os.name == 'nt' else 4,
                       help='Number of data loading workers per GPU')

    # Device settings
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3],
                       help='GPU device IDs to use')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    parser.add_argument('--no_use_amp', dest='use_amp', action='store_false',
                       help='Disable automatic mixed precision')
    parser.add_argument('--use_memory_fs', action='store_true', default=True,
                       help='Use memory filesystem for faster data loading')
    parser.add_argument('--no_memory_fs', dest='use_memory_fs', action='store_false',
                       help='Disable memory filesystem')
    parser.add_argument('--download', action='store_true',
                       help='Allow dataset download (CIFAR only)')
    parser.add_argument('--input_size', type=int, default=None,
                       help='Input image size override')

    # Save settings
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--no_checkpoint', dest='save_checkpoints', action='store_false',
                       help='Disable checkpoint saving')
    parser.set_defaults(save_checkpoints=True)

    # Selection
    parser.add_argument('--categories', type=str, nargs='+',
                       default=['best', 'middle', 'worst'],
                       help='Categories to train')
    parser.add_argument('--max_per_category', type=int, default=None,
                       help='Maximum architectures per category')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()


def main():
    args = parse_args()
    apply_yaml_config(args)
    set_random_seed(args.seed)

    print("="*80)
    print("Multi-GPU NAS Architecture Training")
    print("="*80)

    dataset_name = normalize_dataset_name(args.dataset)
    dataset_info = get_dataset_info(dataset_name)
    args.dataset = dataset_name
    args.dataset_num_classes = dataset_info['num_classes']

    # Resolve default paths
    if args.train_dir is None:
        if dataset_name == 'imagenet100':
            args.train_dir = '/home/xuming/Documents/dataset/ImageNet_100/train'
        elif dataset_name in ('cifar10', 'cifar100'):
            args.train_dir = './data'
        else:
            print("ImageNet-1k requires --train_dir")
            sys.exit(1)

    if args.val_dir is None:
        if dataset_name == 'imagenet100':
            args.val_dir = '/home/xuming/Documents/dataset/ImageNet_100/val'
        elif dataset_name in ('cifar10', 'cifar100'):
            args.val_dir = args.train_dir
        else:
            print("ImageNet-1k requires --val_dir")
            sys.exit(1)

    # Check paths
    if not os.path.exists(args.nas_results):
        print(f"NAS results directory not found: {args.nas_results}")
        sys.exit(1)

    if dataset_info['type'] == 'imagefolder':
        if not os.path.exists(args.train_dir):
            print(f"Training directory not found: {args.train_dir}")
            sys.exit(1)
        if not os.path.exists(args.val_dir):
            print(f"Validation directory not found: {args.val_dir}")
            sys.exit(1)
    elif not os.path.exists(args.train_dir) and not args.download:
        print(f"CIFAR root directory not found: {args.train_dir}")
        print("Hint: use --download to allow automatic download")
        sys.exit(1)

    # Load architectures
    print(f"\nLoading architectures from {args.nas_results}...")
    architectures = load_nas_architectures(args.nas_results, args.categories)

    # Limit per category if specified
    if args.max_per_category:
        filtered = []
        for category in args.categories:
            cat_archs = [a for a in architectures if a['category'] == category]
            filtered.extend(cat_archs[:args.max_per_category])
        architectures = filtered

    print(f"Found {len(architectures)} architectures to train:")
    for category in ['best', 'middle', 'worst']:
        count = sum(1 for a in architectures if a['category'] == category)
        if count > 0:
            print(f"  - {category}: {count}")

    if not architectures:
        print("No architectures found. Exiting.")
        return

    training_cfg = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_workers': args.num_workers,
        'save_checkpoints': args.save_checkpoints,
        'save_freq': args.save_freq,
        'use_amp': args.use_amp,
    }

    result_root = os.path.join(args.nas_results, 'trained_models')
    os.makedirs(result_root, exist_ok=True)

    model_configs, arch_map = build_nas_model_configs(
        architectures,
        args.dataset_num_classes,
        training_cfg,
        result_root,
    )

    manager = MultiGPUManager(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        result_dir=result_root,
        gpus=args.gpus,
        num_classes=args.dataset_num_classes,
        default_epochs=args.epochs,
        default_batch_size=args.batch_size,
        default_lr=args.learning_rate,
        default_num_workers=args.num_workers,
        use_memory_fs=args.use_memory_fs,
        dataset=args.dataset,
        download=args.download,
        input_size=args.input_size,
        seed=args.seed,
    )

    results = manager.train_models(
        model_configs=model_configs,
        force=True,
        parallel=True,
        return_details=True,
    )

    print(f"\n{'='*80}")
    print("Training Complete")
    print(f"{'='*80}")

    total = len(model_configs)
    success_count = len(results.get('success', {}))
    failed_count = len(results.get('failed', {}))

    print(f"\nTotal: {total} architectures")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")

    if results.get('failed'):
        print("\nFailed architectures:")
        for model_name, error in results['failed'].items():
            print(f"  - {model_name}")
            print(f"    Error: {error[:100]}...")

    details = results.get('details', {})
    nas_results = build_nas_results(details, arch_map)

    if nas_results:
        save_results(nas_results, args.nas_results)
    else:
        print("\nNo successful architectures; skipping result export")

    print("\nAccuracy by Category:")
    for category in ['best', 'middle', 'worst']:
        cat_results = [r for r in nas_results if r['category'] == category]
        if cat_results:
            mean_acc = sum(r['best_val_acc'] for r in cat_results) / len(cat_results)
            print(f"  {category:8s}: {mean_acc:.2f}% (n={len(cat_results)})")

    print(f"\nResults saved in: {args.nas_results}")
    print("  - training_results.csv: Detailed results")
    print("  - training_summary.json: Summary statistics")
    print("  - trained_models/: Model checkpoints and logs")


if __name__ == '__main__':
    main()
