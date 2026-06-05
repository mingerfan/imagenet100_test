#!/usr/bin/env python3
"""Entry point for running regularized evolution NAS

Usage:
    # Run full evolution
    python nas_evolution/run_evolution.py

    # Run with custom config
    python nas_evolution/run_evolution.py --config nas_evolution/evolution_config.yaml

    # Resume from checkpoint
    python nas_evolution/run_evolution.py --resume nas_results/run_001/checkpoints/checkpoint_gen50.json

    # Run test evolution
    python nas_evolution/run_evolution.py --config nas_evolution/evolution_config_test.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nas_evolution.regularized_evolution import RegularizedEvolution
from nas_evolution.utils import load_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run regularized evolution for Neural Architecture Search'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='nas_evolution/evolution_config.yaml',
        help='Path to evolution configuration YAML file'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint file to resume from'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )

    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='Override single GPU device ID from config'
    )

    parser.add_argument(
        '--gpus',
        nargs='+',
        default=None,
        help='Override GPU list/range from config, e.g. --gpus all, --gpus 0-7, --gpus 1 2 3'
    )

    parser.add_argument(
        '--exclude_gpus',
        nargs='+',
        default=None,
        help='Exclude physical GPU ids, e.g. --exclude_gpus 0 or --exclude_gpus 0,7'
    )

    parser.add_argument(
        '--population_size',
        type=int,
        default=None,
        help='Override population size from config'
    )

    parser.add_argument(
        '--num_generations',
        type=int,
        default=None,
        help='Override number of generations from config'
    )

    parser.add_argument(
        '--network_config',
        type=str,
        default=None,
        help='Override network configuration file (e.g., network_gen/configs/imagenet_224_resnet_style.yaml)'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    if args.gpu is not None and args.gpus is not None:
        print("ERROR: use either --gpu or --gpus, not both")
        return 1

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    if args.output_dir is None:
        config_stem = Path(args.config).stem
        if config.logging.output_dir in {"nas_results", "./nas_results"}:
            config.logging.output_dir = os.path.join(config.logging.output_dir, config_stem)

    # Override config with command line arguments
    if args.output_dir:
        config.logging.output_dir = args.output_dir
    if args.gpus is not None:
        config.evaluation.gpus = args.gpus
        config.evaluation.gpu = None
    if args.gpu is not None:
        config.evaluation.gpu = args.gpu
        if hasattr(config.evaluation, 'gpus'):
            config.evaluation.gpus = None
    if args.exclude_gpus is not None:
        config.evaluation.exclude_gpus = args.exclude_gpus
    if args.population_size:
        config.search.population_size = args.population_size
    if args.num_generations:
        config.search.num_generations = args.num_generations
    if args.network_config:
        config.network_config = args.network_config
        print(f"Overriding network config: {args.network_config}")

    # Create output directory
    os.makedirs(config.logging.output_dir, exist_ok=True)

    # Print configuration
    print("\n" + "=" * 80)
    print("REGULARIZED EVOLUTION FOR NAS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Population size: {config.search.population_size}")
    print(f"  Generations: {config.search.num_generations}")
    print(f"  Tournament sample size: {config.search.sample_size}")
    print(f"  Output directory: {config.logging.output_dir}")
    print(f"  GPU: {getattr(config.evaluation, 'gpu', None)}")
    print(f"  GPUs: {getattr(config.evaluation, 'gpus', None)}")
    print(f"  Exclude GPUs: {getattr(config.evaluation, 'exclude_gpus', None)}")
    print(f"  Resolution: {config.evaluation.resolution}")
    print(f"  Batch size: {config.evaluation.batch_size}")

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")

    print("\n" + "=" * 80 + "\n")

    # Create evolution instance
    evolution = RegularizedEvolution(config)

    # Run evolution
    try:
        best_individuals = evolution.run(resume_from=args.resume)

        # Print summary
        print("\n" + "=" * 80)
        print("EVOLUTION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nBest {len(best_individuals)} architectures:")

        for i, ind in enumerate(best_individuals[:5]):  # Show top 5
            print(f"\nRank {i+1}:")
            print(f"  ZenNAS Fitness: {ind.zen_fitness:.6f}")
            print(f"  ZEN Score: {ind.scores.get('zen_score', 0.0):.4f}")
            print(f"  Params: {ind.scores.get('params', 0):,}")
            print(f"  FLOPs: {ind.scores.get('flops', 0.0):.2e}")
            print(f"  FHE Latency: {ind.scores.get('fhe_latency', 0.0):.0f}")
            print(f"  FHE Boot Count: {ind.scores.get('fhe_boot_count', 0)}")
            print(f"  Generation: {ind.generation}")

        print(f"\nResults saved to: {config.logging.output_dir}")
        print(f"  - evolution.log: Detailed logs")
        print(f"  - evolution_stats.json: Statistics history")
        print(f"  - checkpoints/: Saved checkpoints")
        print(f"  - best_models/: Best architecture configurations")

    except KeyboardInterrupt:
        print("\n\nEvolution interrupted by user")
        print("Checkpoints have been saved and can be resumed")
        return 1

    except Exception as e:
        print(f"\n\nERROR: Evolution failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
