"""
Utilities for training NAS architectures via the shared training framework.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv


def load_nas_architectures(
    nas_result_dir: str,
    categories: Optional[List[str]] = None,
) -> List[Dict]:
    """Load NAS architectures and metadata from result directory."""
    architectures = []
    categories = categories or ['best', 'middle', 'worst']

    for category in categories:
        model_dir = Path(nas_result_dir) / f'{category}_models'
        if not model_dir.exists():
            continue

        for json_file in sorted(model_dir.glob('*.json')):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            architectures.append({
                'category': category,
                'arch_id': json_file.stem,
                'json_path': str(json_file),
                'scores': data.get('scores', {}) if isinstance(data, dict) else {},
                'aznas_fitness': data.get('aznas_fitness', 0.0) if isinstance(data, dict) else 0.0,
                'generation': data.get('generation', 0) if isinstance(data, dict) else 0,
            })

    return architectures


def build_nas_model_configs(
    architectures: List[Dict],
    dataset_num_classes: int,
    training_config: Dict,
    result_root: str,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Build model_configs for MultiGPUManager and a lookup map for metadata."""
    model_configs = []
    arch_map = {}

    for arch in architectures:
        name = f"{arch['category']}/{arch['arch_id']}"
        result_dir = os.path.join(result_root, arch['category'], arch['arch_id'])

        model_configs.append({
            'name': name,
            'class': 'nas-json',
            'params': {
                'json_path': arch['json_path'],
                'num_classes': dataset_num_classes,
            },
            'epochs': training_config.get('epochs'),
            'batch_size': training_config.get('batch_size'),
            'learning_rate': training_config.get('learning_rate'),
            'num_workers': training_config.get('num_workers'),
            'save_checkpoints': training_config.get('save_checkpoints', True),
            'save_freq': training_config.get('save_freq', 10),
            'use_amp': training_config.get('use_amp', True),
            'result_dir': result_dir,
            'trainer_kwargs': {
                'val_batch_stats_path': os.path.join(result_dir, 'val_batch_stats.csv'),
                'val_batch_stats_anomaly_only': True,
            },
        })

        arch_map[name] = arch

    return model_configs, arch_map


def build_nas_results(details: Dict[str, Dict], arch_map: Dict[str, Dict]) -> List[Dict]:
    """Merge training details with NAS metadata for CSV/summary export."""
    results = []
    for name, detail in details.items():
        arch = arch_map.get(name)
        if not arch:
            continue
        results.append({
            'category': arch['category'],
            'arch_id': arch['arch_id'],
            'aznas_fitness': arch.get('aznas_fitness', 0.0),
            'scores': arch.get('scores', {}),
            'generation': arch.get('generation', 0),
            'best_val_acc': detail.get('best_acc', 0.0),
            'train_time': detail.get('train_time', 0),
            'final_train_loss': detail.get('final_train_loss', 0),
            'final_val_loss': detail.get('final_val_loss', 0),
        })
    return results


def save_results(results: List[Dict], nas_result_dir: str) -> None:
    """Save training results to CSV and summary JSON."""
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
            scores = result.get('scores', {})
            writer.writerow({
                'category': result['category'],
                'arch_id': result['arch_id'],
                'aznas_fitness': result['aznas_fitness'],
                'expressivity': scores.get('expressivity', 0.0),
                'progressivity': scores.get('progressivity', 0.0),
                'trainability': scores.get('trainability', 0.0),
                'fhe_latency': scores.get('fhe_latency', 0.0),
                'fhe_boot_count': scores.get('fhe_boot_count', 0),
                'fhe_max_depth': scores.get('fhe_max_depth', 0),
                'generation': result['generation'],
                'best_val_acc': result['best_val_acc'],
                'train_time': result['train_time'],
                'final_train_loss': result['final_train_loss'],
                'final_val_loss': result['final_val_loss']
                })

    print(f"\nResults saved to: {csv_path}")

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
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {json_path}")
