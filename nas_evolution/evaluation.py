"""Fitness Evaluation for NAS

Wraps the zero_cost_proxy evaluation and integrates with network_gen to build
and evaluate architectures.
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
from .fitness_function import ZenNASFitnessFunction


class FitnessEvaluator:
    """Evaluates architecture fitness using ZenNAS zero-cost proxy

    Integrates:
    - NetworkBuilder to construct models from NetworkConfig
    - ZenNAS compute_nas_score with FHE latency
    - ZenNAS fitness function for ranking
    """

    def __init__(self, config):
        """Initialize fitness evaluator

        Args:
            config: Configuration object with evaluation settings
        """
        self.config = config
        self.gpu = config.evaluation.gpu
        self.resolution = config.evaluation.resolution
        self.batch_size = config.evaluation.batch_size
        self.use_dataloader = config.evaluation.use_dataloader
        self.synflow_check = getattr(config.evaluation, "synflow_check", False)

        # Initialize fitness function
        self.fitness_fn = ZenNASFitnessFunction()

        # Cache for evaluated architectures (avoid re-evaluation)
        self.eval_cache = {}

        print(f"FitnessEvaluator initialized:")
        print(f"  GPU: {self.gpu}")
        print(f"  Resolution: {self.resolution}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Use dataloader: {self.use_dataloader}")
        print(f"  SynFlow check: {self.synflow_check}")

    def evaluate(self, network_config) -> Dict:
        """Evaluate single architecture

        Args:
            network_config: NetworkConfig object

        Returns:
            Dict with evaluation scores:
                - 'zen_score': float (primary metric)
                - 'std_zen_score': float
                - 'params': int
                - 'flops': float
                - 'fhe_latency': float
                - 'fhe_boot_count': int
                - 'fhe_max_depth': int
                - 'synflow_score': float (if enabled)
                - 'synflow_issue': str or None (if enabled)
        """
        # Check cache (hash by config dict)
        config_str = str(sorted(network_config.to_dict().items()))
        if config_str in self.eval_cache:
            print(f"  Using cached evaluation")
            return self.eval_cache[config_str]

        try:
            # Build model from config
            from network_gen import NetworkBuilder
            builder = NetworkBuilder()
            model = builder.build(network_config)

            # Import zero_cost_proxy functions
            from network_evaluate.zero_cost_proxy import compute_nas_score

            # Compute zero-cost proxy scores with FHE latency
            scores = compute_nas_score(
                model=model,
                gpu=self.gpu,
                trainloader=None,
                resolution=self.resolution,
                batch_size=self.batch_size,
                include_synflow=self.synflow_check
            )

            if self.synflow_check:
                synflow_issue = scores.get('synflow_issue')
                if synflow_issue:
                    print(f"  ⚠ SynFlow issue detected: {synflow_issue}")

            # Cache result
            self.eval_cache[config_str] = scores

            return scores

        except Exception as e:
            print(f"Error evaluating architecture: {e}")
            import traceback
            traceback.print_exc()

            # Return invalid scores
            invalid_scores = {
                'zen_score': float('-inf'),
                'std_zen_score': 0.0,
                'params': 0,
                'flops': 0.0,
                'fhe_latency': float('inf'),
                'fhe_boot_count': 0,
                'fhe_max_depth': 0,
                'fhe_operation_latency': 0.0,
                'fhe_boot_latency': 0.0
            }
            if self.synflow_check:
                invalid_scores.update({
                    'synflow_score': float('nan'),
                    'synflow_grad_norm': float('nan'),
                    'synflow_params': 0,
                    'synflow_issue': 'evaluation_error',
                    'synflow_ok': False
                })
            return invalid_scores

    def evaluate_population(self, population):
        """Evaluate multiple architectures and compute ZenNAS fitness

        Args:
            population: List of (network_config, existing_scores) tuples
                       If existing_scores is None, will evaluate

        Returns:
            List of (scores_dict, fitness) tuples
        """
        results = []

        for i, item in enumerate(population):
            if isinstance(item, tuple):
                network_config, existing_scores = item
            else:
                network_config = item
                existing_scores = None

            print(f"Evaluating architecture {i+1}/{len(population)}...")

            # Evaluate if needed
            if existing_scores is None:
                scores = self.evaluate(network_config)
            else:
                scores = existing_scores

            results.append(scores)

        # Compute ZenNAS fitness for entire population
        zen_scores = self.fitness_fn.compute_fitness(results)

        # Return pairs of (scores, zen_fitness)
        return [(results[i], float(zen_scores[i])) for i in range(len(results))]

    def clear_cache(self):
        """Clear evaluation cache"""
        self.eval_cache = {}
        print("Evaluation cache cleared")

    def get_cache_size(self) -> int:
        """Get number of cached evaluations"""
        return len(self.eval_cache)


def test_evaluator():
    """Test fitness evaluator"""
    import argparse
    from types import SimpleNamespace

    # Create mock config
    config = SimpleNamespace(
        evaluation=SimpleNamespace(
            gpu=0,
            resolution=224,
            batch_size=16,
            use_dataloader=False
        )
    )

    # Create evaluator
    evaluator = FitnessEvaluator(config)

    # Generate random network
    from network_gen import RandomNetworkGenerator, GeneratorConfig
    gen_config = GeneratorConfig.from_yaml('network_gen/configs/imagenet_224.yaml')
    generator = RandomNetworkGenerator(gen_config)

    network_config = generator.generate_random()

    print("\nEvaluating random architecture...")
    scores = evaluator.evaluate(network_config)

    print(f"\nResults:")
    print(f"  ZEN Score: {scores['zen_score']:.4f}")
    print(f"  Params: {scores['params']}")
    print(f"  FLOPs: {scores['flops']:.0f}")
    print(f"  FHE Latency: {scores['fhe_latency']:.0f}")
    print(f"  FHE Boot Count: {scores['fhe_boot_count']}")


if __name__ == '__main__':
    test_evaluator()
