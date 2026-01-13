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

    def __init__(self, max_size: int = 200):
        """Initialize population

        Args:
            max_size: Maximum number of individuals in population
        """
        self.max_size = max_size
        self.individuals: List[Individual] = []
        self.history: List[Individual] = []

    def add(self, network_config, scores: Dict, aznas_fitness: float, generation: int):
        """Add individual to population, removing oldest if full

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

        # Age regularization: Remove oldest if over capacity
        if len(self.individuals) > self.max_size:
            removed = self.individuals.pop(0)
            print(f"Removed individual {removed.id} (age={removed.age}, gen={removed.generation})")

        # Increment age of all individuals
        for ind in self.individuals:
            ind.increment_age()

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
