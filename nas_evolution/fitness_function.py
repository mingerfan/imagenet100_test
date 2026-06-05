"""ZenNAS Fitness Function with FHE Latency Constraints

Uses ZEN score as the primary evaluation metric for neural architecture search,
combined with FHE latency constraints for privacy-preserving neural networks.

Reference:
- Zen-NAS: "Zen-NAS: A Zero-Shot NAS for High-Performance Deep Image Recognition"
           (Lin et al., ICCV 2021)

FHE Latency Constraints:
- Hard constraint: Filter out architectures with latency > baseline
- Soft reward: Give multiplier bonus to useful low-latency architectures,
  while avoiding unlimited reward for models that are likely too small
"""

import numpy as np
from typing import List, Dict
import random


class ZenNASFitnessFunction:
    """ZenNAS Fitness Function with FHE Latency Constraints

    Uses ZEN score as the primary metric for architecture evaluation.
    ZEN score has been shown to have strong correlation with actual
    network performance on ImageNet.

    FHE Latency Integration:
    - Hard constraint: architectures with latency > baseline get score = -inf
    - Soft reward: moderately low latency architectures get a bonus multiplier;
      extremely tiny architectures are deliberately penalized relative to the
      best efficiency band

    Formula:
        fitness = zen_score * latency_multiplier
        (filtered to -inf if latency > baseline)
    """

    def __init__(self, latency_baseline: float = 22334905.50):
        """Initialize ZenNAS fitness function

        Args:
            latency_baseline: FHE latency baseline for constraints (default: ResNet-18)
        """
        self.latency_baseline = latency_baseline

    def compute_fitness(self, population_scores: List[Dict]) -> np.ndarray:
        """Compute ZenNAS fitness scores for population

        Args:
            population_scores: List of dicts with keys:
                - 'zen_score': float (higher is better)
                - 'fhe_latency': float (used for constraints)

        Returns:
            fitness_scores: Array of fitness values (higher is better)
                           Architectures violating latency constraint get -inf
        """
        m = len(population_scores)
        if m == 0:
            return np.array([])

        # Extract ZEN scores and latencies
        zen_scores = np.array([s.get('zen_score', -np.inf) for s in population_scores])
        fhe_latency = np.array([s.get('fhe_latency', np.inf) for s in population_scores])

        # Handle invalid ZEN scores
        zen_scores = self._handle_invalid_scores(zen_scores)

        # Compute latency multipliers (soft reward)
        latency_multipliers = self._compute_latency_multipliers(fhe_latency)

        # Final fitness = zen_score * latency_multiplier
        fitness_scores = zen_scores * latency_multipliers

        # Apply hard constraint: set filtered architectures to -inf
        fitness_scores[fhe_latency > self.latency_baseline] = -np.inf

        return fitness_scores

    def _compute_latency_multipliers(self, latencies: np.ndarray) -> np.ndarray:
        """Compute latency multipliers for fitness scores (soft rewards)

        The multiplier is intentionally non-monotonic. It rewards architectures
        that reduce FHE latency, but penalizes extremely small models because
        they often win latency while losing too much accuracy capacity.

        Reward tiers (from baseline):
        - ≤ 10% baseline: 1.10x multiplier
        - ≤ 20% baseline: 1.20x multiplier
        - ≤ 30% baseline: 1.40x multiplier
        - ≤ 40% baseline: 1.35x multiplier
        - ≤ 50% baseline: 1.30x multiplier
        - ≤ 60% baseline: 1.25x multiplier
        - ≤ 70% baseline: 1.20x multiplier
        - ≤ 80% baseline: 1.10x multiplier
        - ≤ 90% baseline: 1.05x multiplier
        - ≤ 100% baseline: 1.00x multiplier (no bonus)

        Args:
            latencies: Array of FHE latency values

        Returns:
            multipliers: Array of multiplier values (>= 1.0)
        """
        multipliers = np.ones_like(latencies, dtype=float)

        tiers = [
            (0.10, 1.10),
            (0.20, 1.20),
            (0.30, 1.40),
            (0.40, 1.35),
            (0.50, 1.30),
            (0.60, 1.25),
            (0.70, 1.20),
            (0.80, 1.10),
            (0.90, 1.05),
            (1.00, 1.00),
        ]

        for i, latency in enumerate(latencies):
            for threshold, multiplier in tiers:
                if latency <= threshold * self.latency_baseline:
                    multipliers[i] = multiplier
                    break

        return multipliers

    def _handle_invalid_scores(self, scores: np.ndarray) -> np.ndarray:
        """Replace invalid scores (NaN, inf) with worst possible value

        Args:
            scores: Array of ZEN scores

        Returns:
            scores: Array with invalid values replaced
        """
        scores = scores.copy()
        invalid_mask = ~np.isfinite(scores)

        if np.any(invalid_mask):
            # For ZEN score (higher is better), use minimum valid value - 1
            valid_min = np.min(scores[~invalid_mask]) if np.any(~invalid_mask) else 0
            scores[invalid_mask] = valid_min - 1.0

        return scores

    def get_best_indices(self, fitness_scores: np.ndarray, k: int = 10) -> np.ndarray:
        """Get indices of top k architectures by fitness score

        Args:
            fitness_scores: Array of fitness scores
            k: Number of top architectures to return

        Returns:
            indices: Indices of top k architectures (highest scores)
        """
        k = min(k, len(fitness_scores))
        return np.argsort(fitness_scores)[-k:][::-1]

    def compute_detailed_stats(self, population_scores: List[Dict], fitness_scores: np.ndarray) -> Dict:
        """Compute detailed statistics about the population

        Args:
            population_scores: List of score dicts
            fitness_scores: Array of fitness scores

        Returns:
            dict with statistics
        """
        latencies = np.array([s.get('fhe_latency', np.inf) for s in population_scores])
        zen_scores = np.array([s.get('zen_score', -np.inf) for s in population_scores])
        params_list = [s.get('params', 0) for s in population_scores]
        flops_list = [s.get('flops', 0) for s in population_scores]

        # Count latency violations and bonuses
        violation_count = np.sum(latencies > self.latency_baseline)
        bonus_count = np.sum(latencies <= 0.9 * self.latency_baseline)

        # Filter out -inf scores for valid statistics
        valid_mask = np.isfinite(fitness_scores)
        valid_scores = fitness_scores[valid_mask]
        valid_zen = zen_scores[np.isfinite(zen_scores)]

        return {
            'best_fitness': float(np.max(valid_scores)) if len(valid_scores) > 0 else -np.inf,
            'mean_fitness': float(np.mean(valid_scores)) if len(valid_scores) > 0 else -np.inf,
            'worst_fitness': float(np.min(valid_scores)) if len(valid_scores) > 0 else -np.inf,
            'std_fitness': float(np.std(valid_scores)) if len(valid_scores) > 0 else 0,
            'valid_count': int(np.sum(valid_mask)),
            'invalid_count': int(np.sum(~valid_mask)),
            'best_zen_score': float(np.max(valid_zen)) if len(valid_zen) > 0 else -np.inf,
            'mean_zen_score': float(np.mean(valid_zen)) if len(valid_zen) > 0 else -np.inf,
            'best_latency': float(np.min(latencies)),
            'mean_latency': float(np.mean(latencies)),
            'worst_latency': float(np.max(latencies)),
            'latency_baseline': self.latency_baseline,
            'latency_violation_count': int(violation_count),
            'latency_bonus_count': int(bonus_count),
            'best_params': int(np.min(params_list)) if params_list else 0,
            'mean_params': float(np.mean(params_list)) if params_list else 0,
            'best_flops': float(np.min(flops_list)) if flops_list else 0,
            'mean_flops': float(np.mean(flops_list)) if flops_list else 0,
        }


class SelectionWithRanking:
    """Selection strategies using ZenNAS fitness ranking"""

    def __init__(self, fitness_fn: ZenNASFitnessFunction):
        """Initialize selection strategy

        Args:
            fitness_fn: ZenNAS fitness function instance
        """
        self.fitness_fn = fitness_fn

    def tournament_select(self, candidates: List, sample_size: int = 25) -> int:
        """Tournament selection using ZenNAS fitness scores

        Args:
            candidates: List of (index, scores_dict) tuples
            sample_size: Number of candidates to sample

        Returns:
            index: Index of selected parent
        """
        # Sample candidates
        sampled = random.sample(candidates, min(sample_size, len(candidates)))

        # Extract scores
        scores_list = [scores for _, scores in sampled]
        indices = [idx for idx, _ in sampled]

        # Compute fitness
        fitness_scores = self.fitness_fn.compute_fitness(scores_list)

        # Return index of best
        best_idx = np.argmax(fitness_scores)
        return indices[best_idx]
