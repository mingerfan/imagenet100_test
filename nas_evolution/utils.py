"""Utilities for Evolution

Includes checkpointing, configuration loading, and logging utilities.
"""

import json
import yaml
import os
from datetime import datetime
from typing import Dict, Any, List
from types import SimpleNamespace


def load_config(config_path: str) -> SimpleNamespace:
    """Load configuration from YAML file

    Args:
        config_path: Path to YAML configuration file

    Returns:
        SimpleNamespace with nested configuration
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Convert nested dicts to nested SimpleNamespace
    return dict_to_namespace(config_dict)


def dict_to_namespace(d: Dict) -> SimpleNamespace:
    """Recursively convert dict to SimpleNamespace

    Args:
        d: Dictionary to convert

    Returns:
        SimpleNamespace with nested SimpleNamespace for nested dicts
    """
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


class EvolutionCheckpoint:
    """Save/load evolution state for checkpointing"""

    def __init__(self, output_dir: str):
        """Initialize checkpoint manager

        Args:
            output_dir: Directory to save checkpoints
        """
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, generation: int, population, config: Any, eval_cache: Dict = None):
        """Save evolution checkpoint

        Args:
            generation: Current generation number
            population: Population object
            config: Configuration object
            eval_cache: Optional evaluation cache dict
        """
        checkpoint = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'population_size': len(population),
            'history_size': len(population.history),
            'config': self._serialize_config(config)
        }

        # Save population separately (can be large)
        population_file = os.path.join(
            self.checkpoint_dir,
            f'population_gen{generation}.json'
        )
        population.save_to_file(population_file)

        # Save checkpoint metadata
        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_gen{generation}.json'
        )
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        # Save evaluation cache if provided
        if eval_cache is not None:
            cache_file = os.path.join(
                self.checkpoint_dir,
                f'eval_cache_gen{generation}.json'
            )
            with open(cache_file, 'w') as f:
                json.dump(eval_cache, f)

        print(f"Checkpoint saved: generation {generation}")

    def load(self, checkpoint_path: str):
        """Load evolution checkpoint

        Args:
            checkpoint_path: Path to checkpoint JSON file

        Returns:
            (generation, population, eval_cache) tuple
        """
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        generation = checkpoint['generation']

        # Load population
        population_file = os.path.join(
            os.path.dirname(checkpoint_path),
            f'population_gen{generation}.json'
        )

        from .population import Population
        population = Population.load_from_file(population_file)

        # Load evaluation cache if exists
        cache_file = os.path.join(
            os.path.dirname(checkpoint_path),
            f'eval_cache_gen{generation}.json'
        )
        eval_cache = {}
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                eval_cache = json.load(f)

        print(f"Checkpoint loaded: generation {generation}")
        print(f"  Population size: {len(population)}")
        print(f"  History size: {len(population.history)}")
        print(f"  Cache size: {len(eval_cache)}")

        return generation, population, eval_cache

    def _serialize_config(self, config: Any) -> Dict:
        """Serialize config to dict

        Args:
            config: Configuration object

        Returns:
            Dict representation
        """
        if isinstance(config, SimpleNamespace):
            return {k: self._serialize_config(v) for k, v in vars(config).items()}
        else:
            return config


class EvolutionLogger:
    """Logger for evolution progress"""

    def __init__(self, output_dir: str):
        """Initialize logger

        Args:
            output_dir: Directory to save logs
        """
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, 'evolution.log')
        self.stats_file = os.path.join(output_dir, 'evolution_stats.json')

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"Evolution started at {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

        self.stats_history = []

    def log_generation(self, generation: int, stats: Dict, best_individual):
        """Log generation statistics

        Args:
            generation: Generation number
            stats: Population statistics dict
            best_individual: Best individual in current generation
        """
        message = f"Generation {generation}:\n"
        message += f"  Population size: {stats['size']}\n"
        message += f"  Best fitness: {stats['best_fitness']:.6f}\n"
        message += f"  Mean fitness: {stats['mean_fitness']:.6f}\n"
        message += f"  Worst fitness: {stats['worst_fitness']:.6f}\n"
        message += f"  Mean age: {stats['mean_age']:.2f}\n"

        if best_individual:
            message += f"  Best individual:\n"
            message += f"    ZEN Score: {best_individual.scores.get('zen_score', 0.0):.4f}\n"
            message += f"    Params: {best_individual.scores.get('params', 0):,}\n"
            message += f"    FLOPs: {best_individual.scores.get('flops', 0.0):.2e}\n"
            message += f"    FHE Latency: {best_individual.scores.get('fhe_latency', 0.0):.0f}\n"
            message += f"    FHE Boot Count: {best_individual.scores.get('fhe_boot_count', 0)}\n"

        message += "\n"

        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(message)

        # Print to console
        print(message)

        # Save to stats history
        stats_entry = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            **stats
        }
        if best_individual:
            stats_entry['best_scores'] = best_individual.scores

        self.stats_history.append(stats_entry)

        # Save stats to JSON
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats_history, f, indent=2)

    def log_message(self, message: str):
        """Log a custom message

        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"[{timestamp}] {message}\n"

        with open(self.log_file, 'a') as f:
            f.write(full_message)

        print(full_message.strip())

    def log_final_results(self, best_individuals):
        """Log final evolution results

        Args:
            best_individuals: List of best individuals
        """
        message = "\n" + "=" * 80 + "\n"
        message += "EVOLUTION COMPLETED\n"
        message += "=" * 80 + "\n\n"
        message += f"Top {len(best_individuals)} architectures:\n\n"

        for i, ind in enumerate(best_individuals):
            message += f"Rank {i+1}:\n"
            message += f"  Fitness: {ind.zen_fitness:.6f}\n"
            message += f"  SynFlow: {ind.scores.get('synflow_score', 0.0):.4f}\n"
            message += f"  ZEN Score: {ind.scores.get('zen_score', 0.0):.4f}\n"
            message += f"  Params: {ind.scores.get('params', 0):,}\n"
            message += f"  FLOPs: {ind.scores.get('flops', 0.0):.2e}\n"
            message += f"  FHE Latency: {ind.scores.get('fhe_latency', 0.0):.0f}\n"
            message += f"  FHE Boot Count: {ind.scores.get('fhe_boot_count', 0)}\n"
            message += f"  Generation: {ind.generation}\n"
            message += "\n"

        with open(self.log_file, 'a') as f:
            f.write(message)

        print(message)

    def plot_evolution_stats(self):
        """Plot evolution statistics and save to output directory
        
        Generates plots for:
        - Fitness over generations (best, mean, worst)
        - ZEN score progression
        - FHE latency progression
        - Parameter count distribution
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for saving
        except ImportError:
            print("Warning: matplotlib not available, skipping plot generation")
            return
        
        if not self.stats_history:
            print("No stats history available for plotting")
            return
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Extract data from stats history
        generations = [s['generation'] for s in self.stats_history]
        best_fitness = [s['best_fitness'] for s in self.stats_history]
        mean_fitness = [s['mean_fitness'] for s in self.stats_history]
        worst_fitness = [s['worst_fitness'] for s in self.stats_history]
        
        # Extract best scores if available
        zen_scores = []
        fhe_latencies = []
        params_list = []
        fhe_boot_counts = []
        
        for s in self.stats_history:
            if 'best_scores' in s:
                zen_scores.append(s['best_scores'].get('zen_score', 0))
                fhe_latencies.append(s['best_scores'].get('fhe_latency', 0))
                params_list.append(s['best_scores'].get('params', 0))
                fhe_boot_counts.append(s['best_scores'].get('fhe_boot_count', 0))
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else None
        
        # 1. Fitness progression plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(generations, best_fitness, 'g-', linewidth=2, label='Best Fitness', marker='o', markersize=3)
        ax.plot(generations, mean_fitness, 'b-', linewidth=1.5, label='Mean Fitness', alpha=0.7)
        ax.plot(generations, worst_fitness, 'r-', linewidth=1, label='Worst Fitness', alpha=0.5)
        ax.fill_between(generations, worst_fitness, best_fitness, alpha=0.2, color='blue')
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness Score', fontsize=12)
        ax.set_title('Evolution Fitness Progression', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'fitness_progression.png'), dpi=150)
        plt.close()
        
        # 2. ZEN Score progression (if available)
        if zen_scores:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(generations[:len(zen_scores)], zen_scores, 'purple', linewidth=2, marker='o', markersize=3)
            ax.set_xlabel('Generation', fontsize=12)
            ax.set_ylabel('ZEN Score', fontsize=12)
            ax.set_title('Best Architecture ZEN Score Progression', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'zen_score_progression.png'), dpi=150)
            plt.close()
        
        # 3. FHE Latency progression (if available)
        if fhe_latencies:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(generations[:len(fhe_latencies)], [l/1e6 for l in fhe_latencies], 'orange', linewidth=2, marker='o', markersize=3)
            ax.set_xlabel('Generation', fontsize=12)
            ax.set_ylabel('FHE Latency (×10⁶)', fontsize=12)
            ax.set_title('Best Architecture FHE Latency Progression', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'fhe_latency_progression.png'), dpi=150)
            plt.close()
        
        # 4. FHE Boot Count progression (if available)
        if fhe_boot_counts:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(generations[:len(fhe_boot_counts)], fhe_boot_counts, 'teal', linewidth=2, marker='o', markersize=3)
            ax.set_xlabel('Generation', fontsize=12)
            ax.set_ylabel('FHE Boot Count', fontsize=12)
            ax.set_title('Best Architecture FHE Boot Count Progression', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'fhe_boot_count_progression.png'), dpi=150)
            plt.close()
        
        # 5. Combined metrics plot (2x2 subplots)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Fitness
        axes[0, 0].plot(generations, best_fitness, 'g-', linewidth=2, label='Best')
        axes[0, 0].plot(generations, mean_fitness, 'b-', linewidth=1.5, label='Mean', alpha=0.7)
        axes[0, 0].fill_between(generations, worst_fitness, best_fitness, alpha=0.2, color='blue')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].set_title('Fitness Progression')
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # ZEN Score
        if zen_scores:
            axes[0, 1].plot(generations[:len(zen_scores)], zen_scores, 'purple', linewidth=2)
            axes[0, 1].set_xlabel('Generation')
            axes[0, 1].set_ylabel('ZEN Score')
            axes[0, 1].set_title('ZEN Score (Best)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # FHE Latency
        if fhe_latencies:
            axes[1, 0].plot(generations[:len(fhe_latencies)], [l/1e6 for l in fhe_latencies], 'orange', linewidth=2)
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('FHE Latency (×10⁶)')
            axes[1, 0].set_title('FHE Latency (Best)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Parameters
        if params_list:
            axes[1, 1].plot(generations[:len(params_list)], [p/1e6 for p in params_list], 'brown', linewidth=2)
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Parameters (M)')
            axes[1, 1].set_title('Parameters (Best)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Evolution Statistics Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'evolution_summary.png'), dpi=150)
        plt.close()
        
        print(f"Evolution plots saved to {plots_dir}")


def save_best_architectures(best_individuals, output_dir: str):
    """Save best architectures to separate files

    Args:
        best_individuals: List of best Individual objects
        output_dir: Directory to save architecture files
    """
    best_dir = os.path.join(output_dir, 'best_models')
    os.makedirs(best_dir, exist_ok=True)

    for i, ind in enumerate(best_individuals):
        filename = f'rank{i+1}_fitness{ind.zen_fitness:.4f}.json'
        filepath = os.path.join(best_dir, filename)

        data = {
            'rank': i + 1,
            'zen_fitness': ind.zen_fitness,
            'scores': ind.scores,
            'generation': ind.generation,
            'config': ind.config.to_dict()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    print(f"Saved {len(best_individuals)} best architectures to {best_dir}")


def save_sampled_architectures(individuals, output_dir: str, category: str):
    """Save sampled architectures to separate files

    Args:
        individuals: List of Individual objects
        output_dir: Directory to save architecture files
        category: Category name ('top', 'middle', 'worst')
    """
    sample_dir = os.path.join(output_dir, f'{category}_models')
    os.makedirs(sample_dir, exist_ok=True)

    for i, ind in enumerate(individuals):
        filename = f'{category}_rank{i+1}_fitness{ind.zen_fitness:.4f}.json'
        filepath = os.path.join(sample_dir, filename)

        data = {
            'category': category,
            'rank_in_category': i + 1,
            'zen_fitness': ind.zen_fitness,
            'scores': ind.scores,
            'generation': ind.generation,
            'config': ind.config.to_dict()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    print(f"Saved {len(individuals)} {category} architectures to {sample_dir}")
