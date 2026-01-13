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
        help='Override GPU device ID from config'
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

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Override config with command line arguments
    if args.output_dir:
        config.logging.output_dir = args.output_dir
    if args.gpu is not None:
        config.evaluation.gpu = args.gpu
    if args.population_size:
        config.search.population_size = args.population_size
    if args.num_generations:
        config.search.num_generations = args.num_generations

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
    print(f"  GPU: {config.evaluation.gpu}")
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
            print(f"  AZ-NAS Fitness: {ind.aznas_fitness:.6f}")
            print(f"  Expressivity: {ind.scores['expressivity']:.4f}")
            print(f"  Progressivity: {ind.scores['progressivity']:.4f}")
            print(f"  Trainability: {ind.scores['trainability']:.4f}")
            print(f"  FHE Latency: {ind.scores['fhe_latency']:.0f}")
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
