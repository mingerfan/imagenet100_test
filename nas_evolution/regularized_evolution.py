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
import copy
import math

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .population import Population
from .mutations import MutationOperator
from .evaluation import FitnessEvaluator
from .fitness_function import ZenNASFitnessFunction
from .utils import EvolutionCheckpoint, EvolutionLogger, save_best_architectures, save_sampled_architectures


def _namespace_to_dict(value):
    """Convert SimpleNamespace-style config sections into plain dictionaries."""
    if hasattr(value, "__dict__"):
        return {
            key: _namespace_to_dict(item)
            for key, item in vars(value).items()
        }
    if isinstance(value, dict):
        return {
            key: _namespace_to_dict(item)
            for key, item in value.items()
        }
    return value


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
        
        # Load generator config to get search space parameters
        allowed_block_ids = None
        ct_slots = 32768
        input_size = 224
        stem_downsample = 4
        initial_min_channels = 16
        initial_max_channels = 64
        allowed_stem_codes = None
        allowed_second_ds_codes = None
        allowed_ct_policies = None
        
        if hasattr(config, 'network_config') and config.network_config:
            try:
                from network_gen.generator_config import GeneratorConfig
                generator_config = GeneratorConfig.from_yaml(config.network_config)
                search_space = generator_config.search_space
                allowed_block_ids = search_space.blocks.allowed_block_ids
                allowed_stem_codes = search_space.stem.allowed_codes
                allowed_second_ds_codes = search_space.second_downsample.allowed_codes
                allowed_ct_policies = search_space.ct_policies.allowed
                ct_slots = getattr(search_space, 'ct_slots', ct_slots)
                initial_min_channels = getattr(search_space, 'initial_min_channels', initial_min_channels)
                initial_max_channels = getattr(search_space, 'initial_max_channels', initial_max_channels)
                # Get input size from dataset config
                if hasattr(generator_config, 'dataset'):
                    input_size = getattr(generator_config.dataset, 'input_size', input_size)
            except Exception:
                pass  # Use defaults
        
        mutation_probs = _namespace_to_dict(getattr(config, 'mutation_probs', None))
        self.mutator = MutationOperator(
            mutation_probs=mutation_probs,
            allowed_block_ids=allowed_block_ids,
            ct_slots=ct_slots,
            input_size=input_size,
            stem_downsample=stem_downsample,
            initial_min_channels=initial_min_channels,
            initial_max_channels=initial_max_channels,
            allowed_stem_codes=allowed_stem_codes,
            allowed_second_ds_codes=allowed_second_ds_codes,
            allowed_ct_policies=allowed_ct_policies,
        )
        self.evaluator = FitnessEvaluator(config)
        self.fitness_fn = ZenNASFitnessFunction(latency_baseline=latency_baseline)

        # Logging and checkpointing
        self.logger = EvolutionLogger(config.logging.output_dir)
        self.checkpointer = EvolutionCheckpoint(config.logging.output_dir)

        # Search parameters
        self.population_size = config.search.population_size
        self.num_generations = config.search.num_generations
        self.sample_size = config.search.sample_size
        self.mutation_rate = float(getattr(config.search, 'mutation_rate', 1.0))

        # Set random seed for reproducibility
        if hasattr(config, 'seed'):
            random.seed(config.seed)
            import torch
            torch.manual_seed(config.seed)

        print(f"RegularizedEvolution initialized:")
        print(f"  Population size: {self.population_size}")
        print(f"  Generations: {self.num_generations}")
        print(f"  Tournament sample size: {self.sample_size}")
        print(f"  Mutation rate: {self.mutation_rate}")
        print(f"  Mutation probabilities: {self.mutator.probs}")

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
            self.fitness_fn.latency_baseline = self.population.latency_baseline
            self.evaluator.latency_baseline = self.population.latency_baseline
            self.evaluator.fitness_fn.latency_baseline = self.population.latency_baseline
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
            if random.random() < self.mutation_rate:
                offspring_config = self.mutator.mutate(parent_config)
            else:
                offspring_config = copy.deepcopy(parent_config)

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
        from network_gen.network_generator import RandomNetworkGenerator
        from network_gen.generator_config import GeneratorConfig

        # Load generator config if specified in evolution config
        generator_config = None
        if hasattr(self.config, 'network_config') and self.config.network_config:
            try:
                generator_config = GeneratorConfig.from_yaml(self.config.network_config)
                self.logger.log_message(f"Loaded network config: {self.config.network_config}")
                if generator_config.search_space.stride.allowed_block_counts:
                    self.logger.log_message(f"  Allowed block counts: {generator_config.search_space.stride.allowed_block_counts}")
            except Exception as e:
                self.logger.log_message(f"Warning: Could not load network config: {e}")

        # Create generator with config
        generator = RandomNetworkGenerator(config=generator_config)

        self.logger.log_message(f"Generating {self.population_size} random architectures...")

        initial_configs = []
        for i in range(self.population_size):
            if (i + 1) % 10 == 0:
                self.logger.log_message(f"  Generated: {i+1}/{self.population_size}")

            # Generate random architecture using configured generator
            network_config = generator.generate_random_config()
            initial_configs.append(network_config)

        self.logger.log_message("Evaluating initial population...")
        evaluated = self.evaluator.evaluate_population(initial_configs)

        for network_config, (scores, zen_fitness) in zip(initial_configs, evaluated):
            # Add to population
            self.population.add(network_config, scores, zen_fitness, generation=0)

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
        if len(candidates) == 0:
            raise RuntimeError("Tournament selection requested from an empty population")

        if not any(math.isfinite(float(score)) for score in zen_scores):
            best_individual = min(
                candidates,
                key=lambda ind: (
                    ind.scores.get('fhe_latency', float('inf')),
                    -ind.scores.get('zen_score', float('-inf')),
                ),
            )
            self.logger.log_message(
                "Tournament candidates all had invalid fitness; "
                f"falling back to lowest-latency parent id={best_individual.id}."
            )
            return best_individual.config

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
        top_archs = self._unique_top_architectures(sorted_history, top_k)

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

    def _unique_top_architectures(self, individuals, k: int):
        """Select top-k unique architectures based on full config signature."""
        unique = []
        seen = set()
        for ind in individuals:
            sig = self._config_signature(ind.config)
            if sig in seen:
                continue
            seen.add(sig)
            unique.append(ind)
            if len(unique) >= k:
                break
        unique_count = len(unique)
        if unique_count < k:
            for ind in individuals:
                if ind in unique:
                    continue
                unique.append(ind)
                if len(unique) >= k:
                    break
            if getattr(self, "logger", None):
                self.logger.log_message(
                    f"Top-{k} unique only {unique_count}; filled {k - unique_count} duplicates to reach {k}."
                )
        return unique

    def _config_signature(self, config):
        """Build a stable signature for architecture uniqueness."""
        block_sig = []
        if hasattr(config, 'blocks') and config.blocks:
            for block in config.blocks:
                block_sig.append((block.block_id, block.stride))
        else:
            block_sig = list(getattr(config, 'block_choices', []))

        return (
            getattr(config, 'stem_code', None),
            getattr(config, 'second_ds_code', None),
            getattr(config, 'stride_code', None),
            tuple(getattr(config, 'ct_policies', [])),
            getattr(config, 'initial_ct_count', None),
            tuple(block_sig),
        )


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
