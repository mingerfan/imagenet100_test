"""Fitness Evaluation for NAS

Wraps the zero_cost_proxy evaluation and integrates with network_gen to build
and evaluate architectures.
"""

import sys
import os
import torch
import multiprocessing as mp
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional, Sequence, Tuple
from .fitness_function import ZenNASFitnessFunction
from utils import (
    format_gpu_ids_with_physical,
    format_visible_gpu_mapping,
    parse_gpu_id_list,
    resolve_gpu_selection,
)


def _config_cache_key(network_config) -> str:
    return str(sorted(network_config.to_dict().items()))


def _invalid_scores(synflow_check: bool = False) -> Dict:
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
    if synflow_check:
        invalid_scores.update({
            'synflow_score': float('nan'),
            'synflow_grad_norm': float('nan'),
            'synflow_params': 0,
            'synflow_issue': 'evaluation_error',
            'synflow_ok': False
        })
    return invalid_scores


def _evaluate_network_config(network_config, settings: Dict, gpu_id: Optional[int]) -> Dict:
    try:
        seed = settings.get('seed')
        task_index = settings.get('task_index')
        if seed is not None and task_index is not None:
            task_seed = int(seed) + int(task_index)
            random.seed(task_seed)
            try:
                import numpy as np
                np.random.seed(task_seed % (2**32 - 1))
            except Exception:
                pass
            torch.manual_seed(task_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(task_seed)

        if gpu_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)

        # Build model from config
        from network_gen import NetworkBuilder
        builder = NetworkBuilder()
        model = builder.build(network_config)

        # Import zero_cost_proxy functions
        from network_evaluate.zero_cost_proxy import compute_nas_score

        # Compute zero-cost proxy scores with FHE latency
        scores = compute_nas_score(
            model=model,
            gpu=gpu_id,
            trainloader=None,
            resolution=settings['resolution'],
            batch_size=settings['batch_size'],
            include_synflow=settings.get('synflow_check', False),
        )

        if settings.get('synflow_check', False):
            synflow_issue = scores.get('synflow_issue')
            if synflow_issue:
                print(f"  ⚠ SynFlow issue detected: {synflow_issue}")

        return scores

    except Exception as e:
        print(f"Error evaluating architecture: {e}")
        import traceback
        traceback.print_exc()
        return _invalid_scores(settings.get('synflow_check', False))

    finally:
        if gpu_id is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _evaluation_worker_loop(task_queue, result_queue, settings: Dict, gpu_id: Optional[int]):
    from network_gen import NetworkConfig

    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    while True:
        task = task_queue.get()
        if task is None:
            break

        index, config_dict = task
        task_settings = dict(settings)
        task_settings['task_index'] = index
        try:
            network_config = NetworkConfig.from_dict(config_dict)
            scores = _evaluate_network_config(network_config, task_settings, gpu_id)
            result_queue.put((index, scores))
        except Exception:
            import traceback
            traceback.print_exc()
            result_queue.put((index, _invalid_scores(settings.get('synflow_check', False))))


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
        self.resolution = config.evaluation.resolution
        self.batch_size = config.evaluation.batch_size
        self.use_dataloader = config.evaluation.use_dataloader
        self.synflow_check = getattr(config.evaluation, "synflow_check", False)
        self.seed = getattr(config, "seed", None)
        self.latency_baseline = getattr(config.fitness, 'latency_baseline', 22334905.50) if hasattr(config, 'fitness') else 22334905.50
        self.parallel_evaluations = bool(
            getattr(config.evaluation, "parallel_evaluations", True)
        )
        self.max_workers = getattr(config.evaluation, "max_workers", None)
        self.gpus = self._resolve_gpus(config)
        self.gpu = self.gpus[0] if self.gpus else None
        self._next_gpu_index = 0

        # Initialize fitness function
        self.fitness_fn = ZenNASFitnessFunction(latency_baseline=self.latency_baseline)

        # Cache for evaluated architectures (avoid re-evaluation)
        self.eval_cache = {}

        print(f"FitnessEvaluator initialized:")
        print(f"  GPUs: {format_gpu_ids_with_physical(self.gpus, self.visible_to_physical_gpus)}")
        print(f"  CUDA_VISIBLE_DEVICES: {self.cuda_visible_devices}")
        print(f"  Visible GPU mapping: {format_visible_gpu_mapping(self.visible_to_physical_gpus)}")
        print(f"  Resolution: {self.resolution}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Latency baseline: {self.latency_baseline}")
        print(f"  Use dataloader: {self.use_dataloader}")
        print(f"  SynFlow check: {self.synflow_check}")
        print(f"  Parallel evaluations: {self.parallel_evaluations}")

    def _resolve_gpus(self, config) -> List[int]:
        evaluation = config.evaluation
        gpu_spec = getattr(evaluation, "gpus", None)
        if gpu_spec is None and hasattr(evaluation, "gpu"):
            gpu = getattr(evaluation, "gpu")
            gpu_spec = None if gpu is None else [gpu]

        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        excluded = parse_gpu_id_list(
            getattr(evaluation, "exclude_gpus", None),
            device_count=device_count,
        )
        selection = resolve_gpu_selection(
            gpu_spec,
            excluded_physical_gpus=excluded,
            device_count=device_count,
        )
        self.visible_to_physical_gpus = selection.visible_to_physical
        self.cuda_visible_devices = selection.cuda_visible_devices
        self.skipped_gpus = selection.skipped
        if selection.skipped:
            print(
                "  Excluded GPUs: "
                f"{format_gpu_ids_with_physical(selection.skipped, selection.visible_to_physical)}"
            )
        return selection.selected

    def _next_gpu(self) -> Optional[int]:
        if not self.gpus:
            return None
        gpu = self.gpus[self._next_gpu_index % len(self.gpus)]
        self._next_gpu_index += 1
        return gpu

    def _evaluation_settings(self, task_index: Optional[int] = None) -> Dict:
        settings = {
            'resolution': self.resolution,
            'batch_size': self.batch_size,
            'synflow_check': self.synflow_check,
            'seed': self.seed,
        }
        if task_index is not None:
            settings['task_index'] = task_index
        return settings

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
        config_str = _config_cache_key(network_config)
        if config_str in self.eval_cache:
            print(f"  Using cached evaluation")
            return self.eval_cache[config_str]

        gpu_id = self._next_gpu()
        scores = _evaluate_network_config(
            network_config,
            self._evaluation_settings(),
            gpu_id,
        )

        # Cache result
        self.eval_cache[config_str] = scores

        return scores

    def _parallel_evaluate_pending(
        self,
        pending: Sequence[Tuple[int, object]],
        results: List[Optional[Dict]],
    ) -> None:
        worker_gpus = self.gpus
        if self.max_workers:
            worker_gpus = worker_gpus[:max(1, int(self.max_workers))]
        worker_count = min(len(worker_gpus), len(pending))
        if worker_count <= 1:
            for index, network_config in pending:
                scores = self.evaluate(network_config)
                results[index] = scores
            return

        print(f"Parallel NAS evaluation: {len(pending)} architectures on {worker_count} GPUs")
        try:
            ctx = mp.get_context('spawn')
        except RuntimeError:
            ctx = mp.get_context()

        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        settings = self._evaluation_settings()

        for index, network_config in pending:
            task_queue.put((index, network_config.to_dict()))
        for _ in range(worker_count):
            task_queue.put(None)

        processes = []
        for gpu_id in worker_gpus[:worker_count]:
            process = ctx.Process(
                target=_evaluation_worker_loop,
                args=(task_queue, result_queue, settings, gpu_id),
                name=f"nas-eval-gpu-{gpu_id}",
            )
            process.start()
            processes.append(process)

        received = 0
        while received < len(pending):
            index, scores = result_queue.get()
            results[index] = scores
            received += 1

        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                print(f"⚠ worker {process.name} did not exit cleanly; terminating")
                process.terminate()
                process.join(timeout=10)

    def evaluate_population(self, population):
        """Evaluate multiple architectures and compute ZenNAS fitness

        Args:
            population: List of (network_config, existing_scores) tuples
                       If existing_scores is None, will evaluate

        Returns:
            List of (scores_dict, fitness) tuples
        """
        results: List[Optional[Dict]] = [None] * len(population)
        pending: List[Tuple[int, object]] = []

        for i, item in enumerate(population):
            if isinstance(item, tuple):
                network_config, existing_scores = item
            else:
                network_config = item
                existing_scores = None

            print(f"Evaluating architecture {i+1}/{len(population)}...")

            # Evaluate if needed
            if existing_scores is None:
                config_str = _config_cache_key(network_config)
                if config_str in self.eval_cache:
                    print(f"  Using cached evaluation")
                    results[i] = self.eval_cache[config_str]
                else:
                    pending.append((i, network_config))
            else:
                results[i] = existing_scores

        if pending:
            if self.parallel_evaluations and len(self.gpus) > 1:
                self._parallel_evaluate_pending(pending, results)
            else:
                for index, network_config in pending:
                    results[index] = self.evaluate(network_config)

            for index, network_config in pending:
                self.eval_cache[_config_cache_key(network_config)] = results[index]

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
