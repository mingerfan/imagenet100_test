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

    # Extract metrics
    fitness = [a['aznas_fitness'] for a in architectures]
    expressivity = [a['scores']['expressivity'] for a in architectures]
    progressivity = [a['scores']['progressivity'] for a in architectures]
    trainability = [a['scores']['trainability'] for a in architectures]
    latency = [a['scores']['fhe_latency'] for a in architectures]
    boot_count = [a['scores']['fhe_boot_count'] for a in architectures]

    return {
        'count': len(architectures),
        'fitness': {
            'mean': np.mean(fitness),
            'std': np.std(fitness),
            'min': np.min(fitness),
            'max': np.max(fitness)
        },
        'expressivity': {
            'mean': np.mean(expressivity),
            'std': np.std(expressivity),
            'min': np.min(expressivity),
            'max': np.max(expressivity)
        },
        'progressivity': {
            'mean': np.mean(progressivity),
            'std': np.std(progressivity),
            'min': np.min(progressivity),
            'max': np.max(progressivity)
        },
        'trainability': {
            'mean': np.mean(trainability),
            'std': np.std(trainability),
            'min': np.min(trainability),
            'max': np.max(trainability)
        },
        'latency': {
            'mean': np.mean(latency),
            'std': np.std(latency),
            'min': np.min(latency),
            'max': np.max(latency)
        },
        'boot_count': {
            'mean': np.mean(boot_count),
            'std': np.std(boot_count),
            'min': np.min(boot_count),
            'max': np.max(boot_count)
        }
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

        print(f"\nAZ-NAS Fitness:")
        print(f"  Mean: {stats['fitness']['mean']:.4f} ± {stats['fitness']['std']:.4f}")
        print(f"  Range: [{stats['fitness']['min']:.4f}, {stats['fitness']['max']:.4f}]")

        print(f"\nExpressivity:")
        print(f"  Mean: {stats['expressivity']['mean']:.4f} ± {stats['expressivity']['std']:.4f}")
        print(f"  Range: [{stats['expressivity']['min']:.4f}, {stats['expressivity']['max']:.4f}]")

        print(f"\nProgressivity:")
        print(f"  Mean: {stats['progressivity']['mean']:.4f} ± {stats['progressivity']['std']:.4f}")
        print(f"  Range: [{stats['progressivity']['min']:.4f}, {stats['progressivity']['max']:.4f}]")

        print(f"\nTrainability:")
        print(f"  Mean: {stats['trainability']['mean']:.4f} ± {stats['trainability']['std']:.4f}")
        print(f"  Range: [{stats['trainability']['min']:.4f}, {stats['trainability']['max']:.4f}]")

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
        ('aznas_fitness', 'AZ-NAS Fitness'),
        ('expressivity', 'Expressivity'),
        ('progressivity', 'Progressivity'),
        ('trainability', 'Trainability'),
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
            if metric_key == 'aznas_fitness':
                values = [a[metric_key] for a in archs]
            else:
                values = [a['scores'][metric_key] / scale for a in archs]

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

        fitness = [a['aznas_fitness'] for a in archs]
        latency = [a['scores']['fhe_latency'] / 1000 for a in archs]

        ax.scatter(latency, fitness, alpha=0.6, label=category, color=colors[category], s=50)

    ax.set_xlabel('FHE Latency (ms)')
    ax.set_ylabel('AZ-NAS Fitness')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fitness vs Expressivity
    ax = axes[1]
    for category in ['top', 'middle', 'worst']:
        archs = results[category]
        if not archs:
            continue

        fitness = [a['aznas_fitness'] for a in archs]
        expressivity = [a['scores']['expressivity'] for a in archs]

        ax.scatter(expressivity, fitness, alpha=0.6, label=category, color=colors[category], s=50)

    ax.set_xlabel('Expressivity')
    ax.set_ylabel('AZ-NAS Fitness')
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
                'aznas_fitness': arch['aznas_fitness'],
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
