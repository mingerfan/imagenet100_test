#!/usr/bin/env python3
"""Train NAS architectures from evolution search

This script trains the architectures sampled from NAS evolution
and saves results for correlation analysis.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
import csv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from network_gen import create_network
from network_gen.network_config import NetworkConfig
from trainers import Trainer
from trainers.multi_gpu_manager import create_smart_optimizer
from data import create_dataloaders, get_dataset_info, normalize_dataset_name
from utils import set_random_seed, load_config


def load_nas_architectures(nas_result_dir: str):
    """Load all NAS architectures from result directory

    Args:
        nas_result_dir: Directory containing NAS results

    Returns:
        List of (category, arch_id, config, scores) tuples
    """
    architectures = []

    for category in ['best', 'middle', 'worst']:
        model_dir = Path(nas_result_dir) / f'{category}_models'
        if not model_dir.exists():
            continue

        for json_file in sorted(model_dir.glob('*.json')):
            with open(json_file) as f:
                data = json.load(f)

            # Extract architecture ID from filename
            arch_id = json_file.stem  # e.g., "rank1_fitness-1.234"

            architectures.append({
                'category': category,
                'arch_id': arch_id,
                'config': data['config'],
                'scores': data['scores'],
                'aznas_fitness': data.get('aznas_fitness', 0.0),
                'generation': data.get('generation', 0)
            })

    return architectures


def train_architecture(arch_info, train_loader, val_loader, result_dir, device, args):
    """Train a single architecture

    Args:
        arch_info: Architecture information dict
        train_loader: Training data loader
        val_loader: Validation data loader
        result_dir: Directory to save results
        device: Training device
        args: Command line arguments

    Returns:
        Dict with training results
    """
    category = arch_info['category']
    arch_id = arch_info['arch_id']

    print(f"\n{'='*80}")
    print(f"Training: {category}/{arch_id}")
    print(f"{'='*80}")
    print(f"AZ-NAS Fitness: {arch_info['aznas_fitness']:.4f}")
    print(f"Expressivity: {arch_info['scores']['expressivity']:.4f}")
    print(f"Progressivity: {arch_info['scores']['progressivity']:.4f}")
    print(f"Trainability: {arch_info['scores']['trainability']:.4f}")
    print(f"FHE Latency: {arch_info['scores']['fhe_latency']:.0f}")

    # Create model from config
    try:
        config = NetworkConfig.from_dict(arch_info['config'])
        if args.dataset_num_classes and config.num_classes != args.dataset_num_classes:
            print(f"⚠ Adjusting num_classes: {config.num_classes} -> {args.dataset_num_classes}")
            config.num_classes = args.dataset_num_classes
        model = create_network(config)
        model = model.to(device)
        print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Setup training
    criterion = nn.CrossEntropyLoss()

    # Use smart optimizer (handles different parameter types)
    optimizer = create_smart_optimizer(model, lr=args.learning_rate)

    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01
    )

    # Create result directory for this architecture
    arch_result_dir = os.path.join(result_dir, category, arch_id)
    os.makedirs(arch_result_dir, exist_ok=True)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        result_dir=arch_result_dir,
        epochs=args.epochs,
        scheduler=scheduler,
        use_amp=args.use_amp,
        save_freq=args.save_freq,
        save_checkpoints=args.save_checkpoints,
        grad_clip_max_norm=1.0
    )

    # Train
    try:
        trainer.train()
        best_acc = trainer.best_acc
        print(f"✓ Training completed. Best accuracy: {best_acc:.2f}%")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Return results
    return {
        'category': category,
        'arch_id': arch_id,
        'aznas_fitness': arch_info['aznas_fitness'],
        'scores': arch_info['scores'],
        'generation': arch_info['generation'],
        'best_val_acc': best_acc,
        'train_time': sum(trainer.history['epoch_time']) if trainer.history['epoch_time'] else 0,
        'final_train_loss': trainer.history['train_loss'][-1] if trainer.history['train_loss'] else 0,
        'final_val_loss': trainer.history['val_loss'][-1] if trainer.history['val_loss'] else 0,
    }


def save_results(results, nas_result_dir):
    """Save training results to CSV

    Args:
        results: List of result dicts
        nas_result_dir: NAS result directory
    """
    # Save detailed results
    csv_path = os.path.join(nas_result_dir, 'training_results.csv')

    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'category', 'arch_id', 'aznas_fitness',
            'expressivity', 'progressivity', 'trainability', 'fhe_latency',
            'fhe_boot_count', 'fhe_max_depth',
            'generation', 'best_val_acc', 'train_time',
            'final_train_loss', 'final_val_loss'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                'category': result['category'],
                'arch_id': result['arch_id'],
                'aznas_fitness': result['aznas_fitness'],
                'expressivity': result['scores']['expressivity'],
                'progressivity': result['scores']['progressivity'],
                'trainability': result['scores']['trainability'],
                'fhe_latency': result['scores']['fhe_latency'],
                'fhe_boot_count': result['scores']['fhe_boot_count'],
                'fhe_max_depth': result['scores']['fhe_max_depth'],
                'generation': result['generation'],
                'best_val_acc': result['best_val_acc'],
                'train_time': result['train_time'],
                'final_train_loss': result['final_train_loss'],
                'final_val_loss': result['final_val_loss']
            })

    print(f"\n✓ Results saved to: {csv_path}")

    # Save summary JSON
    summary = {
        'total_architectures': len(results),
        'by_category': {},
        'correlation_ready': True,
        'timestamp': datetime.now().isoformat()
    }

    for category in ['best', 'middle', 'worst']:
        cat_results = [r for r in results if r['category'] == category]
        if cat_results:
            summary['by_category'][category] = {
                'count': len(cat_results),
                'mean_accuracy': sum(r['best_val_acc'] for r in cat_results) / len(cat_results),
                'best_accuracy': max(r['best_val_acc'] for r in cat_results),
                'worst_accuracy': min(r['best_val_acc'] for r in cat_results)
            }

    json_path = os.path.join(nas_result_dir, 'training_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary saved to: {json_path}")


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
    parser = argparse.ArgumentParser(description='Train NAS architectures')

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
                       help='Number of training epochs (default: 25 for quick evaluation)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')

    # Device settings
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
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
    print("NAS Architecture Training")
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
            print("❌ ImageNet-1k requires --train_dir")
            sys.exit(1)

    if args.val_dir is None:
        if dataset_name == 'imagenet100':
            args.val_dir = '/home/xuming/Documents/dataset/ImageNet_100/val'
        elif dataset_name in ('cifar10', 'cifar100'):
            args.val_dir = args.train_dir
        else:
            print("❌ ImageNet-1k requires --val_dir")
            sys.exit(1)

    # Check paths
    if not os.path.exists(args.nas_results):
        print(f"❌ NAS results directory not found: {args.nas_results}")
        sys.exit(1)

    if dataset_info['type'] == 'imagefolder':
        if not os.path.exists(args.train_dir):
            print(f"❌ Training directory not found: {args.train_dir}")
            sys.exit(1)
        if not os.path.exists(args.val_dir):
            print(f"❌ Validation directory not found: {args.val_dir}")
            sys.exit(1)
    elif not os.path.exists(args.train_dir) and not args.download:
        print(f"❌ CIFAR root directory not found: {args.train_dir}")
        print("提示: 使用 --download 允许自动下载")
        sys.exit(1)

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load architectures
    print(f"\nLoading architectures from {args.nas_results}...")
    architectures = load_nas_architectures(args.nas_results)

    # Filter by category
    architectures = [a for a in architectures if a['category'] in args.categories]

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
        print(f"  - {category}: {count}")

    # Create data loaders using existing data module
    print(f"\nCreating data loaders...")
    train_loader, val_loader, _, _ = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        use_memory_fs=args.use_memory_fs,
        dataset=args.dataset,
        download=args.download,
        input_size=args.input_size,
        seed=args.seed
    )
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    # Create result directory
    result_dir = os.path.join(args.nas_results, 'trained_models')
    os.makedirs(result_dir, exist_ok=True)

    # Train architectures
    print(f"\n{'='*80}")
    print(f"Starting training: {len(architectures)} architectures")
    print(f"{'='*80}")

    results = []
    for i, arch_info in enumerate(architectures, 1):
        print(f"\n[{i}/{len(architectures)}] {arch_info['category']}/{arch_info['arch_id']}")

        result = train_architecture(
            arch_info=arch_info,
            train_loader=train_loader,
            val_loader=val_loader,
            result_dir=result_dir,
            device=device,
            args=args
        )

        if result:
            results.append(result)

    # Save results
    print(f"\n{'='*80}")
    print("Training Complete")
    print(f"{'='*80}")

    print(f"\nSuccessfully trained: {len(results)}/{len(architectures)}")
    save_results(results, args.nas_results)

    # Print summary
    print("\nAccuracy by Category:")
    for category in ['best', 'middle', 'worst']:
        cat_results = [r for r in results if r['category'] == category]
        if cat_results:
            mean_acc = sum(r['best_val_acc'] for r in cat_results) / len(cat_results)
            print(f"  {category:8s}: {mean_acc:.2f}% (n={len(cat_results)})")

    print(f"\nResults saved in: {args.nas_results}")
    print("  - training_results.csv: Detailed results")
    print("  - training_summary.json: Summary statistics")
    print(f"  - trained_models/: Model checkpoints and logs")


if __name__ == '__main__':
    main()
