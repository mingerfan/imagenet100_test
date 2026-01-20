"""Regularized Evolution for Neural Architecture Search

Implements the regularized evolution algorithm with:
- FIFO population aging
- Tournament selection
- Single-parent mutation
- ZenNAS fitness function with FHE constraints
"""

import sys
import os
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .population import Population
from .mutations import MutationOperator
from .evaluation import FitnessEvaluator
from .fitness_function import ZenNASFitnessFunction
from .utils import EvolutionCheckpoint, EvolutionLogger, save_best_architectures, save_sampled_architectures


class RegularizedEvolution:
    """Regularized Evolution for NAS

    Based on: Real et al. "Regularized Evolution for Image Classifier
    Architecture Search" (2019)

    Key features:
    - Age-based population management (FIFO queue)
    - Tournament selection from recent samples
    - Single-parent mutation (no crossover)
    - ZenNAS fitness function with FHE latency constraints
    """

    def __init__(self, config):
        """Initialize regularized evolution

        Args:
            config: Configuration object with search parameters
        """
        self.config = config

        # Initialize components
        diversity_quota = getattr(config.search, 'diversity_quota', 0.01)
        latency_baseline = getattr(config.fitness, 'latency_baseline', 22334905.50) if hasattr(config, 'fitness') else 22334905.50

        self.population = Population(
            max_size=config.search.population_size,
            diversity_quota=diversity_quota,
            latency_baseline=latency_baseline
        )
        self.mutator = MutationOperator()
        self.evaluator = FitnessEvaluator(config)
        self.fitness_fn = ZenNASFitnessFunction(latency_baseline=latency_baseline)

        # Logging and checkpointing
        self.logger = EvolutionLogger(config.logging.output_dir)
        self.checkpointer = EvolutionCheckpoint(config.logging.output_dir)

        # Search parameters
        self.population_size = config.search.population_size
        self.num_generations = config.search.num_generations
        self.sample_size = config.search.sample_size

        # Set random seed for reproducibility
        if hasattr(config, 'seed'):
            random.seed(config.seed)
            import torch
            torch.manual_seed(config.seed)

        print(f"RegularizedEvolution initialized:")
        print(f"  Population size: {self.population_size}")
        print(f"  Generations: {self.num_generations}")
        print(f"  Tournament sample size: {self.sample_size}")

    def run(self, resume_from: str = None):
        """Main evolution loop

        Args:
            resume_from: Optional checkpoint path to resume from

        Returns:
            List of best individuals
        """
        start_generation = 0

        if resume_from:
            # Resume from checkpoint
            start_generation, self.population, eval_cache = self.checkpointer.load(resume_from)
            self.evaluator.eval_cache = eval_cache
            self.logger.log_message(f"Resumed from generation {start_generation}")
        else:
            # Initialize population with random architectures
            self.logger.log_message("Initializing population...")
            self._initialize_population()

        # Evolution loop
        for generation in range(start_generation, self.num_generations):
            self.logger.log_message(f"\n{'='*60}")
            self.logger.log_message(f"Starting generation {generation + 1}/{self.num_generations}")

            # Tournament selection
            parent_config = self._tournament_select()

            # Mutation
            offspring_config = self.mutator.mutate(parent_config)

            # Evaluation
            self.logger.log_message("Evaluating offspring...")
            scores = self.evaluator.evaluate(offspring_config)

            # Compute fitness for this single offspring
            # (fitness is relative, but we compute it for logging)
            # We'll recompute proper fitness when needed
            dummy_fitness = scores['zen_score'] - scores['fhe_latency'] / 1e7

            # Add to population (oldest removed automatically if full)
            self.population.add(offspring_config, scores, dummy_fitness, generation)

            # Logging
            if (generation + 1) % self.config.logging.log_interval == 0:
                self._log_progress(generation + 1)

            # Checkpointing
            if (generation + 1) % self.config.logging.checkpoint_interval == 0:
                self._save_checkpoint(generation + 1)

        # Final results
        self.logger.log_message("\nEvolution complete! Computing final rankings...")

        # Get stratified sample for comprehensive evaluation
        stratified_sample = self._get_stratified_sample(
            top_k=15,
            middle_k=15,
            worst_k=15
        )

        # Log and save top architectures
        best_individuals = stratified_sample['top']
        self.logger.log_final_results(best_individuals)
        save_best_architectures(best_individuals, self.config.logging.output_dir)

        # Save middle and worst samples for evaluation
        save_sampled_architectures(stratified_sample['middle'], self.config.logging.output_dir, 'middle')
        save_sampled_architectures(stratified_sample['worst'], self.config.logging.output_dir, 'worst')

        # Log sampling statistics
        self.logger.log_message(f"\nStratified sampling complete:")
        self.logger.log_message(f"  Top architectures: {len(stratified_sample['top'])}")
        self.logger.log_message(f"  Middle architectures: {len(stratified_sample['middle'])}")
        self.logger.log_message(f"  Worst architectures: {len(stratified_sample['worst'])}")
        self.logger.log_message(f"  Total sampled: {len(stratified_sample['top']) + len(stratified_sample['middle']) + len(stratified_sample['worst'])}")

        # Generate evolution statistics plots
        self.logger.log_message("\nGenerating evolution statistics plots...")
        self.logger.plot_evolution_stats()

        return best_individuals

    def _initialize_population(self):
        """Initialize population with random architectures"""
        from network_gen import create_random_network

        self.logger.log_message(f"Generating {self.population_size} random architectures...")

        for i in range(self.population_size):
            if (i + 1) % 10 == 0:
                self.logger.log_message(f"  Progress: {i+1}/{self.population_size}")

            # Generate random architecture
            _, network_config = create_random_network()

            # Evaluate
            scores = self.evaluator.evaluate(network_config)

            # Dummy fitness for initialization (will compute proper fitness later)
            dummy_fitness = 0.0

            # Add to population
            self.population.add(network_config, scores, dummy_fitness, generation=0)

        # Now compute proper ZenNAS fitness for entire population
        self._recompute_population_fitness()

        self.logger.log_message(f"Initial population created: {len(self.population)} architectures")

    def _recompute_population_fitness(self):
        """Recompute ZenNAS fitness for entire population

        This ensures fitness scores are properly ranked relative to the
        current population.
        """
        if len(self.population) == 0:
            return

        # Collect all scores
        all_scores = [ind.scores for ind in self.population.individuals]

        # Compute ZenNAS fitness
        zen_scores = self.fitness_fn.compute_fitness(all_scores)

        # Update individual fitness values
        for i, ind in enumerate(self.population.individuals):
            ind.zen_fitness = float(zen_scores[i])

    def _tournament_select(self):
        """Tournament selection from recent population

        Returns:
            NetworkConfig of selected parent
        """
        # Sample candidates
        candidates = self.population.sample(self.sample_size)

        # Recompute fitness for candidates (in case population has changed)
        candidate_scores = [ind.scores for ind in candidates]
        zen_scores = self.fitness_fn.compute_fitness(candidate_scores)

        # Select best
        best_idx = zen_scores.argmax()
        best_individual = candidates[best_idx]

        return best_individual.config

    def _log_progress(self, generation: int):
        """Log progress at current generation

        Args:
            generation: Current generation number
        """
        # Recompute fitness for accurate statistics
        self._recompute_population_fitness()

        # Get population statistics
        stats = self.population.get_current_stats()

        # Get best individual
        best = self.population.get_best(k=1, from_history=False)[0] if len(self.population) > 0 else None

        # Log
        self.logger.log_generation(generation, stats, best)

    def _save_checkpoint(self, generation: int):
        """Save checkpoint at current generation

        Args:
            generation: Current generation number
        """
        self.checkpointer.save(
            generation=generation,
            population=self.population,
            config=self.config,
            eval_cache=self.evaluator.eval_cache
        )

    def _get_final_best(self, k: int = 10):
        """Get final best architectures with proper fitness ranking

        Args:
            k: Number of best architectures to return

        Returns:
            List of best individuals
        """
        # Recompute fitness for entire history
        all_scores = [ind.scores for ind in self.population.history]
        zen_scores = self.fitness_fn.compute_fitness(all_scores)

        # Update fitness values
        for i, ind in enumerate(self.population.history):
            ind.zen_fitness = float(zen_scores[i])

        # Get best k
        return self.population.get_best(k=k, from_history=True)

    def _get_stratified_sample(self, top_k: int = 15, middle_k: int = 15, worst_k: int = 15):
        """Get stratified sample of architectures for evaluation

        Samples architectures from three strata:
        - Top k: Best architectures by fitness
        - Middle k: Random sample from middle 50%
        - Worst k: Random sample from worst architectures

        Args:
            top_k: Number of top architectures to return
            middle_k: Number of middle architectures to sample
            worst_k: Number of worst architectures to sample

        Returns:
            dict with keys 'top', 'middle', 'worst' containing lists of individuals
        """
        import random

        # Recompute fitness for entire history
        all_scores = [ind.scores for ind in self.population.history]
        zen_scores = self.fitness_fn.compute_fitness(all_scores)

        # Update fitness values
        for i, ind in enumerate(self.population.history):
            ind.zen_fitness = float(zen_scores[i])

        # Sort by fitness (descending)
        sorted_history = sorted(self.population.history, key=lambda x: x.zen_fitness, reverse=True)

        total = len(sorted_history)

        # Top k architectures
        top_archs = sorted_history[:min(top_k, total)]

        # Middle k architectures (from middle 50%)
        middle_start = total // 4
        middle_end = 3 * total // 4
        middle_pool = sorted_history[middle_start:middle_end]

        if len(middle_pool) > middle_k:
            middle_archs = random.sample(middle_pool, middle_k)
        else:
            middle_archs = middle_pool

        # Worst k architectures (from bottom 25%)
        worst_start = 3 * total // 4
        worst_pool = sorted_history[worst_start:]

        if len(worst_pool) > worst_k:
            worst_archs = random.sample(worst_pool, worst_k)
        else:
            worst_archs = worst_pool

        return {
            'top': top_archs,
            'middle': middle_archs,
            'worst': worst_archs
        }


def main():
    """Main entry point for testing"""
    from .utils import load_config

    # Load config
    config_path = 'nas_evolution/evolution_config.yaml'
    config = load_config(config_path)

    # Create evolution
    evolution = RegularizedEvolution(config)

    # Run
    best_individuals = evolution.run()

    print(f"\nFound {len(best_individuals)} best architectures")


if __name__ == '__main__':
    main()
