#!/usr/bin/env python3
"""Analyze stratified sampling results from NAS evolution

This script analyzes the top, middle, and worst architectures saved during
NAS evolution to help evaluate the accuracy of the search metrics.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


def get_fitness(arch: Dict) -> float:
    return float(arch.get('zen_fitness', arch.get('aznas_fitness', 0.0)))


def get_score(arch: Dict, key: str, default: float = 0.0) -> float:
    scores = arch.get('scores', {}) if isinstance(arch, dict) else {}
    value = scores.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def summarize(values: List[float]) -> Dict:
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return {'mean': float('nan'), 'std': float('nan'), 'min': float('nan'), 'max': float('nan')}
    return {
        'mean': np.mean(finite),
        'std': np.std(finite),
        'min': np.min(finite),
        'max': np.max(finite),
    }


def load_architectures(output_dir: str) -> Dict[str, List[Dict]]:
    """Load all sampled architectures

    Args:
        output_dir: Directory containing the results

    Returns:
        Dict with keys 'top', 'middle', 'worst' containing architecture data
    """
    results = {
        'top': [],
        'middle': [],
        'worst': []
    }

    # Load best models
    best_dir = Path(output_dir) / 'best_models'
    if best_dir.exists():
        for filename in sorted(best_dir.glob('*.json')):
            with open(filename) as f:
                data = json.load(f)
                data['category'] = 'top'
                results['top'].append(data)

    # Load middle models
    middle_dir = Path(output_dir) / 'middle_models'
    if middle_dir.exists():
        for filename in sorted(middle_dir.glob('*.json')):
            with open(filename) as f:
                data = json.load(f)
                results['middle'].append(data)

    # Load worst models
    worst_dir = Path(output_dir) / 'worst_models'
    if worst_dir.exists():
        for filename in sorted(worst_dir.glob('*.json')):
            with open(filename) as f:
                data = json.load(f)
                results['worst'].append(data)

    return results


def compute_statistics(architectures: List[Dict]) -> Dict:
    """Compute statistics for a set of architectures

    Args:
        architectures: List of architecture data

    Returns:
        Dict with statistical summaries
    """
    if not architectures:
        return {}

    fitness = [get_fitness(a) for a in architectures]
    zen_score = [get_score(a, 'zen_score', float('-inf')) for a in architectures]
    synflow = [get_score(a, 'synflow_score', float('nan')) for a in architectures]
    params = [get_score(a, 'params', 0.0) for a in architectures]
    flops = [get_score(a, 'flops', 0.0) for a in architectures]
    latency = [get_score(a, 'fhe_latency', float('inf')) for a in architectures]
    boot_count = [get_score(a, 'fhe_boot_count', 0.0) for a in architectures]

    return {
        'count': len(architectures),
        'fitness': summarize(fitness),
        'zen_score': summarize(zen_score),
        'synflow_score': summarize(synflow),
        'params': summarize(params),
        'flops': summarize(flops),
        'latency': summarize(latency),
        'boot_count': summarize(boot_count),
    }


def print_statistics(results: Dict[str, List[Dict]]):
    """Print statistics for all categories

    Args:
        results: Dict with 'top', 'middle', 'worst' architecture lists
    """
    print("\n" + "=" * 80)
    print("STRATIFIED SAMPLING ANALYSIS")
    print("=" * 80)

    for category in ['top', 'middle', 'worst']:
        archs = results[category]
        stats = compute_statistics(archs)

        if not stats:
            continue

        print(f"\n{category.upper()} ARCHITECTURES ({stats['count']} total)")
        print("-" * 80)

        print(f"\nZen Fitness:")
        print(f"  Mean: {stats['fitness']['mean']:.4f} ± {stats['fitness']['std']:.4f}")
        print(f"  Range: [{stats['fitness']['min']:.4f}, {stats['fitness']['max']:.4f}]")

        print(f"\nZEN Score:")
        print(f"  Mean: {stats['zen_score']['mean']:.4f} ± {stats['zen_score']['std']:.4f}")
        print(f"  Range: [{stats['zen_score']['min']:.4f}, {stats['zen_score']['max']:.4f}]")

        print(f"\nSynFlow Score:")
        print(f"  Mean: {stats['synflow_score']['mean']:.4f} ± {stats['synflow_score']['std']:.4f}")
        print(f"  Range: [{stats['synflow_score']['min']:.4f}, {stats['synflow_score']['max']:.4f}]")

        print(f"\nParams:")
        print(f"  Mean: {stats['params']['mean']/1e6:.2f}M ± {stats['params']['std']/1e6:.2f}M")
        print(f"  Range: [{stats['params']['min']/1e6:.2f}M, {stats['params']['max']/1e6:.2f}M]")

        print(f"\nFLOPs:")
        print(f"  Mean: {stats['flops']['mean']/1e6:.2f}M ± {stats['flops']['std']/1e6:.2f}M")
        print(f"  Range: [{stats['flops']['min']/1e6:.2f}M, {stats['flops']['max']/1e6:.2f}M]")

        print(f"\nFHE Latency (ms):")
        print(f"  Mean: {stats['latency']['mean']/1000:.2f} ± {stats['latency']['std']/1000:.2f}")
        print(f"  Range: [{stats['latency']['min']/1000:.2f}, {stats['latency']['max']/1000:.2f}]")

        print(f"\nBootstrap Count:")
        print(f"  Mean: {stats['boot_count']['mean']:.1f} ± {stats['boot_count']['std']:.1f}")
        print(f"  Range: [{stats['boot_count']['min']:.0f}, {stats['boot_count']['max']:.0f}]")


def plot_distributions(results: Dict[str, List[Dict]], output_dir: str):
    """Create distribution plots for all metrics

    Args:
        results: Dict with 'top', 'middle', 'worst' architecture lists
        output_dir: Directory to save plots
    """
    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Stratified Sampling: Metric Distributions', fontsize=16, fontweight='bold')

    metrics = [
        ('zen_fitness', 'Zen Fitness'),
        ('zen_score', 'ZEN Score'),
        ('synflow_score', 'SynFlow Score'),
        ('params', 'Params (M)', 1e6),
        ('fhe_latency', 'FHE Latency (ms)', 1000),
        ('fhe_boot_count', 'Bootstrap Count')
    ]

    colors = {'top': 'green', 'middle': 'orange', 'worst': 'red'}

    for idx, metric_info in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        metric_key = metric_info[0]
        metric_name = metric_info[1]
        scale = metric_info[2] if len(metric_info) > 2 else 1

        # Plot for each category
        for category in ['top', 'middle', 'worst']:
            archs = results[category]
            if not archs:
                continue

            # Extract values
            if metric_key == 'zen_fitness':
                values = [get_fitness(a) for a in archs]
            else:
                values = [get_score(a, metric_key) / scale for a in archs]
            values = [v for v in values if np.isfinite(v)]
            if not values:
                continue

            # Plot
            ax.hist(values, alpha=0.5, label=category, color=colors[category], bins=10)

        ax.set_xlabel(metric_name)
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = Path(output_dir) / 'stratified_sampling_distributions.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")

    # Also create scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Stratified Sampling: Fitness vs Metrics', fontsize=16, fontweight='bold')

    # Fitness vs Latency
    ax = axes[0]
    for category in ['top', 'middle', 'worst']:
        archs = results[category]
        if not archs:
            continue

        fitness = [get_fitness(a) for a in archs]
        latency = [get_score(a, 'fhe_latency') / 1000 for a in archs]

        ax.scatter(latency, fitness, alpha=0.6, label=category, color=colors[category], s=50)

    ax.set_xlabel('FHE Latency (ms)')
    ax.set_ylabel('Zen Fitness')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fitness vs ZEN Score
    ax = axes[1]
    for category in ['top', 'middle', 'worst']:
        archs = results[category]
        if not archs:
            continue

        fitness = [get_fitness(a) for a in archs]
        zen_score = [get_score(a, 'zen_score') for a in archs]

        ax.scatter(zen_score, fitness, alpha=0.6, label=category, color=colors[category], s=50)

    ax.set_xlabel('ZEN Score')
    ax.set_ylabel('Zen Fitness')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = Path(output_dir) / 'stratified_sampling_scatter.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")


def export_for_training(results: Dict[str, List[Dict]], output_dir: str):
    """Export architecture configurations in a format suitable for training

    Args:
        results: Dict with 'top', 'middle', 'worst' architecture lists
        output_dir: Directory to save export file
    """
    export_data = []

    for category in ['top', 'middle', 'worst']:
        for arch in results[category]:
            export_data.append({
                'category': category,
                'config': arch['config'],
                'zen_fitness': get_fitness(arch),
                'scores': arch['scores']
            })

    export_path = Path(output_dir) / 'architectures_for_training.json'
    with open(export_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"\nExported {len(export_data)} architectures for training to: {export_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_sampling.py <output_dir>")
        print("Example: python analyze_sampling.py nas_results/test_run")
        sys.exit(1)

    output_dir = sys.argv[1]

    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} does not exist")
        sys.exit(1)

    # Load architectures
    print(f"Loading architectures from {output_dir}...")
    results = load_architectures(output_dir)

    # Print statistics
    print_statistics(results)

    # Create plots
    try:
        plot_distributions(results, output_dir)
    except Exception as e:
        print(f"\nWarning: Could not create plots: {e}")

    # Export for training
    export_for_training(results, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    total = sum(len(results[cat]) for cat in ['top', 'middle', 'worst'])
    print(f"\nTotal architectures sampled: {total}")
    print(f"  - Top: {len(results['top'])}")
    print(f"  - Middle: {len(results['middle'])}")
    print(f"  - Worst: {len(results['worst'])}")

    print(f"\nFiles saved in: {output_dir}")
    print(f"  - top_models/: Best {len(results['top'])} architectures")
    print(f"  - middle_models/: {len(results['middle'])} randomly sampled from middle 50%")
    print(f"  - worst_models/: {len(results['worst'])} randomly sampled from worst 25%")
    print(f"  - architectures_for_training.json: All configs ready for training")


if __name__ == '__main__':
    main()
