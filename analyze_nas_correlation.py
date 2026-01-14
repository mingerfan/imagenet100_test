#!/usr/bin/env python3
"""Analyze correlation between NAS metrics and actual training accuracy

This script computes correlation coefficients between zero-cost proxy metrics
and actual training accuracy to validate the NAS search.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr


def load_training_results(nas_result_dir):
    """Load training results from CSV

    Args:
        nas_result_dir: NAS results directory

    Returns:
        List of result dicts
    """
    csv_path = Path(nas_result_dir) / 'training_results.csv'

    if not csv_path.exists():
        print(f"❌ Training results not found: {csv_path}")
        print("Please run train_nas_architectures.py first")
        sys.exit(1)

    results = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            results.append({
                'category': row['category'],
                'arch_id': row['arch_id'],
                'aznas_fitness': float(row['aznas_fitness']),
                'expressivity': float(row['expressivity']),
                'progressivity': float(row['progressivity']),
                'trainability': float(row['trainability']),
                'fhe_latency': float(row['fhe_latency']),
                'best_val_acc': float(row['best_val_acc']),
                'generation': int(row['generation'])
            })

    return results


def compute_correlations(results):
    """Compute correlation coefficients

    Args:
        results: List of result dicts

    Returns:
        Dict with correlation results
    """
    # Extract metrics
    aznas_fitness = np.array([r['aznas_fitness'] for r in results])
    expressivity = np.array([r['expressivity'] for r in results])
    progressivity = np.array([r['progressivity'] for r in results])
    trainability = np.array([r['trainability'] for r in results])
    fhe_latency = np.array([r['fhe_latency'] for r in results])
    accuracy = np.array([r['best_val_acc'] for r in results])

    # Compute correlations
    correlations = {}

    metrics = {
        'AZ-NAS Fitness': aznas_fitness,
        'Expressivity': expressivity,
        'Progressivity': progressivity,
        'Trainability': trainability,
        'FHE Latency': -fhe_latency  # Negate so lower is better aligns with higher correlation
    }

    for name, values in metrics.items():
        # Filter out invalid values
        valid_mask = np.isfinite(values) & np.isfinite(accuracy)
        valid_values = values[valid_mask]
        valid_accuracy = accuracy[valid_mask]

        if len(valid_values) < 3:
            print(f"Warning: Not enough valid data for {name}")
            continue

        # Pearson correlation (linear)
        pearson_r, pearson_p = pearsonr(valid_values, valid_accuracy)

        # Spearman correlation (rank-based, monotonic)
        spearman_r, spearman_p = spearmanr(valid_values, valid_accuracy)

        # Kendall Tau (rank-based, for small samples)
        kendall_tau, kendall_p = kendalltau(valid_values, valid_accuracy)

        correlations[name] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'kendall_tau': kendall_tau,
            'kendall_p': kendall_p,
            'n_samples': len(valid_values)
        }

    return correlations


def print_correlation_report(correlations, results):
    """Print detailed correlation report

    Args:
        correlations: Dict with correlation results
        results: List of result dicts
    """
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS: NAS Metrics vs Training Accuracy")
    print("="*80)

    print(f"\nTotal architectures: {len(results)}")
    print(f"Accuracy range: [{min(r['best_val_acc'] for r in results):.2f}%, "
          f"{max(r['best_val_acc'] for r in results):.2f}%]")

    print("\n" + "-"*80)
    print(f"{'Metric':<20} {'Pearson':<12} {'Spearman':<12} {'Kendall':<12} {'Samples':<10}")
    print("-"*80)

    for metric_name, corr_data in correlations.items():
        print(f"{metric_name:<20} "
              f"{corr_data['pearson_r']:>6.4f} (p={corr_data['pearson_p']:.4f})  "
              f"{corr_data['spearman_r']:>6.4f} (p={corr_data['spearman_p']:.4f})  "
              f"{corr_data['kendall_tau']:>6.4f} (p={corr_data['kendall_p']:.4f})  "
              f"{corr_data['n_samples']:>7d}")

    print("-"*80)

    # Interpretation
    print("\nInterpretation Guide:")
    print("  Correlation coefficient (ρ):")
    print("    |ρ| > 0.7  : Strong correlation")
    print("    |ρ| > 0.4  : Moderate correlation")
    print("    |ρ| > 0.2  : Weak correlation")
    print("    |ρ| ≤ 0.2  : Very weak/no correlation")
    print("\n  p-value:")
    print("    p < 0.05  : Statistically significant")
    print("    p < 0.01  : Highly significant")

    # Find best metric
    print("\nBest Predictive Metric:")
    best_metric = max(correlations.items(),
                     key=lambda x: abs(x[1]['spearman_r']))
    print(f"  {best_metric[0]}: Spearman ρ = {best_metric[1]['spearman_r']:.4f}")


def analyze_by_category(results):
    """Analyze performance by category

    Args:
        results: List of result dicts
    """
    print("\n" + "="*80)
    print("PERFORMANCE BY CATEGORY")
    print("="*80)

    for category in ['best', 'middle', 'worst']:
        cat_results = [r for r in results if r['category'] == category]
        if not cat_results:
            continue

        accuracies = [r['best_val_acc'] for r in cat_results]
        fitness_scores = [r['aznas_fitness'] for r in cat_results]

        print(f"\n{category.upper()} Category ({len(cat_results)} architectures):")
        print(f"  Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
        print(f"  Range: [{np.min(accuracies):.2f}%, {np.max(accuracies):.2f}%]")
        print(f"  AZ-NAS Fitness: {np.mean(fitness_scores):.4f} ± {np.std(fitness_scores):.4f}")

    # Statistical test between categories
    print("\n" + "-"*80)
    print("Category Comparison:")
    print("-"*80)

    best_acc = [r['best_val_acc'] for r in results if r['category'] == 'best']
    middle_acc = [r['best_val_acc'] for r in results if r['category'] == 'middle']
    worst_acc = [r['best_val_acc'] for r in results if r['category'] == 'worst']

    if best_acc and middle_acc:
        print(f"  Top vs Middle: {np.mean(best_acc) - np.mean(middle_acc):+.2f}%")
    if best_acc and worst_acc:
        print(f"  Top vs Worst:  {np.mean(best_acc) - np.mean(worst_acc):+.2f}%")
    if middle_acc and worst_acc:
        print(f"  Middle vs Worst: {np.mean(middle_acc) - np.mean(worst_acc):+.2f}%")


def save_correlation_results(correlations, results, output_dir):
    """Save correlation analysis to JSON

    Args:
        correlations: Dict with correlation results
        results: List of result dicts
        output_dir: Output directory
    """
    output = {
        'correlations': correlations,
        'by_category': {},
        'summary': {
            'total_architectures': len(results),
            'accuracy_mean': float(np.mean([r['best_val_acc'] for r in results])),
            'accuracy_std': float(np.std([r['best_val_acc'] for r in results])),
            'accuracy_min': float(np.min([r['best_val_acc'] for r in results])),
            'accuracy_max': float(np.max([r['best_val_acc'] for r in results]))
        }
    }

    # By category
    for category in ['best', 'middle', 'worst']:
        cat_results = [r for r in results if r['category'] == category]
        if cat_results:
            output['by_category'][category] = {
                'count': len(cat_results),
                'accuracy_mean': float(np.mean([r['best_val_acc'] for r in cat_results])),
                'accuracy_std': float(np.std([r['best_val_acc'] for r in cat_results])),
                'fitness_mean': float(np.mean([r['aznas_fitness'] for r in cat_results])),
                'fitness_std': float(np.std([r['aznas_fitness'] for r in cat_results]))
            }

    json_path = Path(output_dir) / 'correlation_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Correlation analysis saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze NAS metric correlations')
    parser.add_argument('--nas_results', type=str, default='nas_results',
                       help='NAS results directory')
    args = parser.parse_args()

    print("="*80)
    print("NAS Metrics Correlation Analysis")
    print("="*80)

    # Load results
    print(f"\nLoading training results from {args.nas_results}...")
    results = load_training_results(args.nas_results)
    print(f"✓ Loaded {len(results)} training results")

    # Compute correlations
    print("\nComputing correlations...")
    correlations = compute_correlations(results)

    # Print report
    print_correlation_report(correlations, results)

    # Analyze by category
    analyze_by_category(results)

    # Save results
    save_correlation_results(correlations, results, args.nas_results)

    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80)


if __name__ == '__main__':
    main()
