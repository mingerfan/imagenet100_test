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
from .fitness_function import AZNASFitnessFunction


class FitnessEvaluator:
    """Evaluates architecture fitness using modified zero_cost_proxy

    Integrates:
    - NetworkBuilder to construct models from NetworkConfig
    - Modified compute_nas_score with FHE latency
    - AZ-NAS fitness function for ranking aggregation
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

        # Initialize fitness function
        self.fitness_fn = AZNASFitnessFunction()

        # Cache for evaluated architectures (avoid re-evaluation)
        self.eval_cache = {}

        print(f"FitnessEvaluator initialized:")
        print(f"  GPU: {self.gpu}")
        print(f"  Resolution: {self.resolution}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Use dataloader: {self.use_dataloader}")

    def evaluate(self, network_config) -> Dict:
        """Evaluate single architecture

        Args:
            network_config: NetworkConfig object

        Returns:
            Dict with evaluation scores:
                - 'expressivity': float
                - 'progressivity': float
                - 'trainability': float
                - 'fhe_latency': float
                - 'fhe_boot_count': int
                - 'fhe_max_depth': int
                - 'fhe_operation_latency': float
                - 'fhe_boot_latency': float
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
                trainloader=None,  # Use random input for speed
                resolution=self.resolution,
                batch_size=self.batch_size,
                init=True,  # Initialize weights
                use_wrapper=True  # Wrap for feature extraction
            )

            # Cache result
            self.eval_cache[config_str] = scores

            return scores

        except Exception as e:
            print(f"Error evaluating architecture: {e}")
            import traceback
            traceback.print_exc()

            # Return invalid scores
            return {
                'expressivity': float('-inf'),
                'progressivity': float('-inf'),
                'trainability': float('-inf'),
                'fhe_latency': float('inf'),
                'fhe_boot_count': 0,
                'fhe_max_depth': 0,
                'fhe_operation_latency': 0.0,
                'fhe_boot_latency': 0.0
            }

    def evaluate_population(self, population):
        """Evaluate multiple architectures and compute AZ-NAS fitness

        Args:
            population: List of (network_config, existing_scores) tuples
                       If existing_scores is None, will evaluate

        Returns:
            List of (scores_dict, aznas_fitness) tuples
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

        # Compute AZ-NAS fitness for entire population
        aznas_scores = self.fitness_fn.compute_fitness(results)

        # Return pairs of (scores, aznas_fitness)
        return [(results[i], float(aznas_scores[i])) for i in range(len(results))]

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
    print(f"  Expressivity: {scores['expressivity']:.4f}")
    print(f"  Progressivity: {scores['progressivity']:.4f}")
    print(f"  Trainability: {scores['trainability']:.4f}")
    print(f"  FHE Latency: {scores['fhe_latency']:.0f}")
    print(f"  FHE Boot Count: {scores['fhe_boot_count']}")
    print(f"  FHE Max Depth: {scores['fhe_max_depth']}")


if __name__ == '__main__':
    test_evaluator()
