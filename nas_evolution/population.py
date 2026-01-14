"""Population Management for Regularized Evolution

Implements population management with FIFO aging regularization as described
in "Regularized Evolution for Image Classifier Architecture Search" (Real et al., 2019).
"""

import random
from typing import List, Dict, Optional
import json


class Individual:
    """Single architecture in the population

    Attributes:
        config: NetworkConfig object defining the architecture
        scores: Dict with all evaluation scores (expressivity, progressivity, trainability, fhe_latency)
        aznas_fitness: AZ-NAS fitness score (computed from scores)
        generation: Birth generation number
        age: Age of individual (increments each generation)
        id: Unique identifier
    """

    _next_id = 0  # Class variable for generating unique IDs

    def __init__(self, network_config, scores: Dict, aznas_fitness: float, generation: int):
        """Initialize individual

        Args:
            network_config: NetworkConfig object
            scores: Dict with evaluation scores
            aznas_fitness: AZ-NAS fitness score
            generation: Generation when individual was created
        """
        self.config = network_config
        self.scores = scores
        self.aznas_fitness = aznas_fitness
        self.generation = generation
        self.age = 0
        self.id = Individual._next_id
        Individual._next_id += 1

    def increment_age(self):
        """Increment individual's age"""
        self.age += 1

    def to_dict(self) -> Dict:
        """Convert individual to dictionary for serialization

        Returns:
            dict with all individual information
        """
        return {
            'id': self.id,
            'config': self.config.to_dict(),
            'scores': self.scores,
            'aznas_fitness': self.aznas_fitness,
            'generation': self.generation,
            'age': self.age
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Individual':
        """Reconstruct individual from dictionary

        Args:
            data: Dict from to_dict()

        Returns:
            Individual object
        """
        from network_gen import NetworkConfig
        config = NetworkConfig.from_dict(data['config'])
        individual = cls(
            network_config=config,
            scores=data['scores'],
            aznas_fitness=data['aznas_fitness'],
            generation=data['generation']
        )
        individual.age = data['age']
        individual.id = data['id']
        return individual


class Population:
    """Manages population with FIFO aging (regularization)

    The population maintains a fixed-size FIFO queue of individuals.
    When adding a new individual to a full population, the oldest is removed.
    This age-based regularization prevents stagnation.

    Attributes:
        max_size: Maximum population size
        individuals: Current population (FIFO queue)
        history: All evaluated individuals (for analysis)
    """

    def __init__(self, max_size: int = 200, diversity_quota: float = 0.01,
                 latency_baseline: float = 22334905.50):
        """Initialize population

        Args:
            max_size: Maximum number of individuals in population
            diversity_quota: Minimum fraction to preserve for each depth (default 1%)
            latency_baseline: FHE latency baseline for hard constraint
        """
        self.max_size = max_size
        self.diversity_quota = diversity_quota
        self.latency_baseline = latency_baseline
        self.individuals: List[Individual] = []
        self.history: List[Individual] = []

    def add(self, network_config, scores: Dict, aznas_fitness: float, generation: int):
        """Add individual to population with diversity-aware removal

        Args:
            network_config: NetworkConfig object
            scores: Evaluation scores dict
            aznas_fitness: AZ-NAS fitness score
            generation: Current generation number
        """
        individual = Individual(network_config, scores, aznas_fitness, generation)

        # Add to population
        self.individuals.append(individual)
        self.history.append(individual)

        # ✅ Diversity-aware removal: Protect minority depths
        if len(self.individuals) > self.max_size:
            self._remove_with_diversity_protection()

        # ✅ FIX: Increment age of existing individuals only (exclude newly added)
        # The newly added individual should start with age=0
        for ind in self.individuals[:-1]:
            ind.increment_age()

    def _get_depth_distribution(self) -> Dict[int, int]:
        """Get distribution of network depths in current population

        Returns:
            Dict mapping depth (num_blocks) to count
        """
        depth_counts = {}
        for ind in self.individuals:
            depth = len(ind.config.blocks)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        return depth_counts

    def _remove_with_diversity_protection(self):
        """Remove individual with diversity protection

        Strategy:
        1. Preserve minority groups (each depth gets minimum quota)
        2. Latency violations are not protected (hard constraint)
        3. Among removable candidates, remove oldest first (FIFO)

        Priority for removal (highest to lowest):
        - P1: Majority group + latency violation
        - P2: Majority group + latency ok
        - P3: Minority group + latency violation
        - P4: Minority group + latency ok (protected)
        """
        if len(self.individuals) <= self.max_size:
            return

        # Calculate minimum quota per depth
        min_quota = max(1, int(self.max_size * self.diversity_quota))

        # Get depth distribution
        depth_counts = self._get_depth_distribution()

        # Try to find removable individual (oldest first)
        for i in range(len(self.individuals)):
            ind = self.individuals[i]
            depth = len(ind.config.blocks)
            is_minority = depth_counts[depth] <= min_quota
            is_latency_violation = ind.scores.get('fhe_latency', float('inf')) > self.latency_baseline

            # Priority 1: Majority + latency violation (remove immediately)
            if not is_minority and is_latency_violation:
                removed = self.individuals.pop(i)
                print(f"Removed individual {removed.id} (depth={depth}, latency violation, age={removed.age})")
                return

        # Priority 2: Majority + latency ok
        for i in range(len(self.individuals)):
            ind = self.individuals[i]
            depth = len(ind.config.blocks)
            is_minority = depth_counts[depth] <= min_quota

            if not is_minority:
                removed = self.individuals.pop(i)
                print(f"Removed individual {removed.id} (depth={depth}, majority group, age={removed.age})")
                return

        # Priority 3: Minority + latency violation
        for i in range(len(self.individuals)):
            ind = self.individuals[i]
            is_latency_violation = ind.scores.get('fhe_latency', float('inf')) > self.latency_baseline

            if is_latency_violation:
                depth = len(ind.config.blocks)
                removed = self.individuals.pop(i)
                print(f"Removed individual {removed.id} (depth={depth}, latency violation, age={removed.age})")
                return

        # Priority 4: All protected - remove oldest anyway
        removed = self.individuals.pop(0)
        depth = len(removed.config.blocks)
        print(f"Removed individual {removed.id} (depth={depth}, forced removal, age={removed.age})")

    def sample(self, k: int) -> List[Individual]:
        """Sample k individuals uniformly at random

        Args:
            k: Number of individuals to sample

        Returns:
            List of sampled individuals
        """
        k = min(k, len(self.individuals))
        return random.sample(self.individuals, k)

    def get_best(self, k: int = 10, from_history: bool = True) -> List[Individual]:
        """Get top k individuals by AZ-NAS fitness

        Args:
            k: Number of individuals to return
            from_history: If True, search full history; if False, search current population

        Returns:
            List of top k individuals (sorted by fitness, best first)
        """
        source = self.history if from_history else self.individuals
        k = min(k, len(source))
        sorted_pop = sorted(source, key=lambda x: x.aznas_fitness, reverse=True)
        return sorted_pop[:k]

    def get_current_stats(self) -> Dict:
        """Get statistics about current population

        Returns:
            dict with population statistics
        """
        if len(self.individuals) == 0:
            return {
                'size': 0,
                'mean_fitness': 0.0,
                'best_fitness': 0.0,
                'worst_fitness': 0.0,
                'mean_age': 0.0,
                'max_age': 0
            }

        fitnesses = [ind.aznas_fitness for ind in self.individuals]
        ages = [ind.age for ind in self.individuals]

        return {
            'size': len(self.individuals),
            'mean_fitness': float(sum(fitnesses) / len(fitnesses)),
            'best_fitness': float(max(fitnesses)),
            'worst_fitness': float(min(fitnesses)),
            'mean_age': float(sum(ages) / len(ages)),
            'max_age': int(max(ages))
        }

    def __len__(self) -> int:
        """Get current population size"""
        return len(self.individuals)

    def __iter__(self):
        """Iterate over current population"""
        return iter(self.individuals)

    def save_to_file(self, filepath: str):
        """Save population to JSON file

        Args:
            filepath: Path to save file
        """
        data = {
            'max_size': self.max_size,
            'individuals': [ind.to_dict() for ind in self.individuals],
            'history': [ind.to_dict() for ind in self.history]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'Population':
        """Load population from JSON file

        Args:
            filepath: Path to load file

        Returns:
            Population object
        """
        with open(filepath) as f:
            data = json.load(f)

        population = cls(max_size=data['max_size'])
        population.individuals = [Individual.from_dict(d) for d in data['individuals']]
        population.history = [Individual.from_dict(d) for d in data['history']]

        return population
