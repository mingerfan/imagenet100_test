"""AZ-NAS Fitness Function Implementation

Implements the non-linear ranking aggregation method from AZ-NAS that uses
logarithmic penalties to ensure balanced architectures across all dimensions.

Modified to use FLOPs instead of latency in ranking, with latency as:
- Hard constraint: Filter out architectures with latency > ResNet-18 baseline
- Soft reward: Give multiplier bonus for low-latency architectures
"""

import numpy as np
from scipy.stats import rankdata
from typing import List, Dict
import random


class AZNASFitnessFunction:
    """AZ-NAS Non-Linear Ranking Aggregation with Latency Constraints

    Implements the ranking aggregation method using:
    - Expressivity, Progressivity, Trainability, FLOPs (ranked)
    - FHE Latency as hard constraint + soft reward (not ranked)

    Formula:
        base_score = sum_{M in {E,P,T,F}} log(Rank(s^M(i)) / m)
        latency_multiplier = bonus based on latency vs baseline
        final_score = base_score * latency_multiplier

    Hard constraint: architectures with latency > baseline are filtered (score = -inf)
    Soft reward: architectures with low latency get multiplier bonus (1.05-1.30x)
    """

    def __init__(self, latency_baseline=22334905.50, epsilon=1e-10):
        """Initialize AZ-NAS fitness function

        Args:
            latency_baseline: ResNet-18 FHE latency baseline for constraints
            epsilon: Small value to handle numerical stability (log(0) protection)
        """
        self.epsilon = epsilon
        self.latency_baseline = latency_baseline
        self.metrics = ['expressivity', 'progressivity', 'trainability', 'flops']

    def compute_fitness(self, population_scores: List[Dict]) -> np.ndarray:
        """Compute AZ-NAS fitness scores for population

        Args:
            population_scores: List of dicts with keys:
                - 'expressivity': float (higher is better)
                - 'progressivity': float (higher is better)
                - 'trainability': float (higher is better)
                - 'flops': float (higher is better - more computation capacity)
                - 'fhe_latency': float (used for constraints, not ranking)

        Returns:
            aznas_scores: Array of fitness values (higher/less negative is better)
                          Architectures violating latency constraint get -inf
        """
        m = len(population_scores)  # Number of candidates

        if m == 0:
            return np.array([])

        # Extract metrics from population
        expressivity = np.array([s['expressivity'] for s in population_scores])
        progressivity = np.array([s['progressivity'] for s in population_scores])
        trainability = np.array([s['trainability'] for s in population_scores])
        flops = np.array([s['flops'] for s in population_scores])
        fhe_latency = np.array([s['fhe_latency'] for s in population_scores])

        # Handle invalid values (NaN, -inf, inf)
        expressivity = self._handle_invalid_scores(expressivity, higher_is_better=True)
        progressivity = self._handle_invalid_scores(progressivity, higher_is_better=True)
        trainability = self._handle_invalid_scores(trainability, higher_is_better=True)
        flops = self._handle_invalid_scores(flops, higher_is_better=True)  # Higher FLOPs = better

        # Compute ranks (ascending: worst=1, best=m)
        # method='average' handles ties by assigning average rank
        rank_expr = rankdata(expressivity, method='average')
        rank_prog = rankdata(progressivity, method='average')
        rank_train = rankdata(trainability, method='average')

        # For FLOPs: higher is better (more computation capacity)
        # Directly rank without negation
        rank_flops = rankdata(flops, method='average')

        # Compute base AZ-NAS scores using logarithmic aggregation (4 metrics)
        base_scores = (
            np.log(rank_expr / m + self.epsilon) +
            np.log(rank_prog / m + self.epsilon) +
            np.log(rank_train / m + self.epsilon) +
            np.log(rank_flops / m + self.epsilon)
        )

        # Apply latency constraints and rewards
        latency_divisors = self._compute_latency_divisors(fhe_latency)

        # Final scores: base_score / latency_divisor
        # Since base_score is negative, dividing by larger divisor makes it less negative (better)
        aznas_scores = base_scores / latency_divisors

        # Apply hard constraint: set filtered architectures to -inf
        aznas_scores[fhe_latency > self.latency_baseline] = -np.inf

        return aznas_scores

    def _compute_latency_divisors(self, latencies: np.ndarray) -> np.ndarray:
        """Compute latency divisors for fitness scores (soft rewards only)

        Soft reward: lower latency → higher divisor → less negative score (better)
        Hard constraint is applied separately by setting score to -inf

        Reward tiers (from ResNet-18 baseline):
        - ≤ 40% baseline: 1.30 divisor (score / 1.30 = more positive)
        - ≤ 50% baseline: 1.25 divisor
        - ≤ 60% baseline: 1.20 divisor
        - ≤ 70% baseline: 1.15 divisor
        - ≤ 80% baseline: 1.10 divisor
        - ≤ 90% baseline: 1.05 divisor
        - ≤ 100% baseline: 1.00 divisor (no change)

        Args:
            latencies: Array of FHE latency values

        Returns:
            divisors: Array of divisor values (all positive, >= 1.0)
        """
        divisors = np.ones_like(latencies, dtype=float)

        # Define reward tiers (check from lowest to highest)
        tiers = [
            (0.40, 1.30),
            (0.50, 1.25),
            (0.60, 1.20),
            (0.70, 1.15),
            (0.80, 1.10),
            (0.90, 1.05),
            (1.00, 1.00),
        ]

        for i, latency in enumerate(latencies):
            # Assign divisor based on tier
            for threshold, divisor in tiers:
                if latency <= threshold * self.latency_baseline:
                    divisors[i] = divisor
                    break

        return divisors

    def _handle_invalid_scores(self, scores: np.ndarray, higher_is_better: bool) -> np.ndarray:
        """Replace invalid scores (NaN, inf) with worst possible value

        Args:
            scores: Array of metric scores
            higher_is_better: Whether higher score is better

        Returns:
            scores: Array with invalid values replaced
        """
        scores = scores.copy()
        invalid_mask = ~np.isfinite(scores)

        if np.any(invalid_mask):
            # Replace invalid with worst possible value
            if higher_is_better:
                # For metrics where higher is better, use minimum valid value - 1
                valid_min = np.min(scores[~invalid_mask]) if np.any(~invalid_mask) else 0
                scores[invalid_mask] = valid_min - 1.0
            else:
                # For metrics where lower is better, use maximum valid value + 1
                valid_max = np.max(scores[~invalid_mask]) if np.any(~invalid_mask) else 1e9
                scores[invalid_mask] = valid_max + 1.0

        return scores

    def get_best_indices(self, aznas_scores: np.ndarray, k: int = 10) -> np.ndarray:
        """Get indices of top k architectures by AZ-NAS score

        Args:
            aznas_scores: Array of AZ-NAS fitness scores
            k: Number of top architectures to return

        Returns:
            indices: Indices of top k architectures (highest scores)
        """
        # Higher (less negative) scores are better
        k = min(k, len(aznas_scores))
        return np.argsort(aznas_scores)[-k:][::-1]

    def compute_detailed_stats(self, population_scores: List[Dict], aznas_scores: np.ndarray) -> Dict:
        """Compute detailed statistics about the population

        Args:
            population_scores: List of score dicts
            aznas_scores: Array of AZ-NAS fitness scores

        Returns:
            dict with statistics:
                - 'best_aznas_score': Best AZ-NAS score in population
                - 'mean_aznas_score': Mean AZ-NAS score
                - 'worst_aznas_score': Worst AZ-NAS score
                - 'best_latency': Best (lowest) FHE latency
                - 'mean_latency': Mean FHE latency
                - 'latency_violation_count': Number of architectures exceeding baseline
                - 'latency_bonus_count': Number of architectures getting bonus
                - 'best_expressivity': Best expressivity score
                - 'best_flops': Best (lowest) FLOPs
                - etc.
        """
        latencies = np.array([s['fhe_latency'] for s in population_scores])
        expressivities = [s['expressivity'] for s in population_scores]
        progressivities = [s['progressivity'] for s in population_scores]
        trainabilities = [s['trainability'] for s in population_scores]
        flops_list = [s['flops'] for s in population_scores]

        # Count latency violations and bonuses
        violation_count = np.sum(latencies > self.latency_baseline)
        bonus_count = np.sum(latencies <= 0.9 * self.latency_baseline)

        # Filter out -inf scores for valid statistics
        valid_mask = np.isfinite(aznas_scores)
        valid_scores = aznas_scores[valid_mask]

        return {
            'best_aznas_score': float(np.max(valid_scores)) if len(valid_scores) > 0 else -np.inf,
            'mean_aznas_score': float(np.mean(valid_scores)) if len(valid_scores) > 0 else -np.inf,
            'worst_aznas_score': float(np.min(valid_scores)) if len(valid_scores) > 0 else -np.inf,
            'std_aznas_score': float(np.std(valid_scores)) if len(valid_scores) > 0 else 0,
            'valid_count': int(np.sum(valid_mask)),
            'invalid_count': int(np.sum(~valid_mask)),
            'best_latency': float(np.min(latencies)),
            'mean_latency': float(np.mean(latencies)),
            'worst_latency': float(np.max(latencies)),
            'latency_baseline': self.latency_baseline,
            'latency_violation_count': int(violation_count),
            'latency_bonus_count': int(bonus_count),
            'best_expressivity': float(np.max(expressivities)),
            'mean_expressivity': float(np.mean(expressivities)),
            'best_progressivity': float(np.max(progressivities)),
            'mean_progressivity': float(np.mean(progressivities)),
            'best_trainability': float(np.max(trainabilities)),
            'mean_trainability': float(np.mean(trainabilities)),
            'best_flops': float(np.min(flops_list)),
            'mean_flops': float(np.mean(flops_list)),
            'worst_flops': float(np.max(flops_list)),
        }


class SelectionWithRanking:
    """Selection strategies using AZ-NAS ranking"""

    def __init__(self, fitness_fn: AZNASFitnessFunction):
        """Initialize selection strategy

        Args:
            fitness_fn: AZ-NAS fitness function instance
        """
        self.fitness_fn = fitness_fn

    def tournament_select(self, candidates: List, sample_size: int = 25) -> int:
        """Tournament selection using AZ-NAS scores

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

        # Compute AZ-NAS fitness
        aznas_scores = self.fitness_fn.compute_fitness(scores_list)

        # Return index of best
        best_idx = np.argmax(aznas_scores)
        return indices[best_idx]
