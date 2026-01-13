"""AZ-NAS Fitness Function Implementation

Implements the non-linear ranking aggregation method from AZ-NAS that uses
logarithmic penalties to ensure balanced architectures across all dimensions.
"""

import numpy as np
from scipy.stats import rankdata
from typing import List, Dict
import random


class AZNASFitnessFunction:
    """AZ-NAS Non-Linear Ranking Aggregation

    Implements the ranking aggregation method from AZ-NAS that penalizes
    architectures with any weak dimension using logarithmic penalties.

    Formula:
        s^{AZ}(i) = sum_{M} log(Rank(s^M(i)) / m)

    where m is the population size, Rank is ascending (worst=1, best=m),
    and the log penalty severely punishes any weak dimension ("bucket effect").
    """

    def __init__(self, epsilon=1e-10):
        """Initialize AZ-NAS fitness function

        Args:
            epsilon: Small value to handle numerical stability (log(0) protection)
        """
        self.epsilon = epsilon
        self.metrics = ['expressivity', 'progressivity', 'trainability', 'fhe_latency']

    def compute_fitness(self, population_scores: List[Dict]) -> np.ndarray:
        """Compute AZ-NAS fitness scores for population

        Args:
            population_scores: List of dicts with keys:
                - 'expressivity': float (higher is better)
                - 'progressivity': float (higher is better)
                - 'trainability': float (higher is better)
                - 'fhe_latency': float (lower is better)

        Returns:
            aznas_scores: Array of fitness values (higher/less negative is better)
        """
        m = len(population_scores)  # Number of candidates

        if m == 0:
            return np.array([])

        # Extract metrics from population
        expressivity = np.array([s['expressivity'] for s in population_scores])
        progressivity = np.array([s['progressivity'] for s in population_scores])
        trainability = np.array([s['trainability'] for s in population_scores])
        fhe_latency = np.array([s['fhe_latency'] for s in population_scores])

        # Handle invalid values (NaN, -inf, inf)
        expressivity = self._handle_invalid_scores(expressivity, higher_is_better=True)
        progressivity = self._handle_invalid_scores(progressivity, higher_is_better=True)
        trainability = self._handle_invalid_scores(trainability, higher_is_better=True)
        fhe_latency = self._handle_invalid_scores(fhe_latency, higher_is_better=False)

        # Compute ranks (ascending: worst=1, best=m)
        # method='average' handles ties by assigning average rank
        rank_expr = rankdata(expressivity, method='average')
        rank_prog = rankdata(progressivity, method='average')
        rank_train = rankdata(trainability, method='average')

        # For latency: lower is better, so we reverse the ranking
        # Negate values before ranking so lowest latency gets highest rank
        rank_latency = rankdata(-fhe_latency, method='average')

        # Compute AZ-NAS scores using logarithmic aggregation
        aznas_scores = (
            np.log(rank_expr / m + self.epsilon) +
            np.log(rank_prog / m + self.epsilon) +
            np.log(rank_train / m + self.epsilon) +
            np.log(rank_latency / m + self.epsilon)
        )

        return aznas_scores

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
                - 'best_expressivity': Best expressivity score
                - etc.
        """
        latencies = [s['fhe_latency'] for s in population_scores]
        expressivities = [s['expressivity'] for s in population_scores]
        progressivities = [s['progressivity'] for s in population_scores]
        trainabilities = [s['trainability'] for s in population_scores]

        return {
            'best_aznas_score': float(np.max(aznas_scores)),
            'mean_aznas_score': float(np.mean(aznas_scores)),
            'worst_aznas_score': float(np.min(aznas_scores)),
            'std_aznas_score': float(np.std(aznas_scores)),
            'best_latency': float(np.min(latencies)),
            'mean_latency': float(np.mean(latencies)),
            'worst_latency': float(np.max(latencies)),
            'best_expressivity': float(np.max(expressivities)),
            'mean_expressivity': float(np.mean(expressivities)),
            'best_progressivity': float(np.max(progressivities)),
            'mean_progressivity': float(np.mean(progressivities)),
            'best_trainability': float(np.max(trainabilities)),
            'mean_trainability': float(np.mean(trainabilities)),
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
