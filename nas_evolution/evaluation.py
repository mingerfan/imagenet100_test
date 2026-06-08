"""Fitness Evaluation for NAS

Wraps the zero_cost_proxy evaluation and integrates with network_gen to build
and evaluate architectures.
"""

import sys
import os
import torch
import multiprocessing as mp
import queue
import random
import time

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


def _invalid_scores(
    synflow_check: bool = False,
    reason: str = "evaluation_error",
) -> Dict:
    invalid_scores = {
        'evaluation_status': reason,
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


def _log_eval_task(
    gpu_id: Optional[int],
    task_index: Optional[int],
    message: str,
    *,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    gpu_label = "CPU" if gpu_id is None else str(gpu_id)
    if task_index is None:
        task_label = "single"
    else:
        task_label = str(int(task_index) + 1)
    print(f"[GPU {gpu_label}] Task {task_label}: {message}", flush=True)


def _seed_eval_task(seed: Optional[int], task_index: Optional[int], gpu_id: Optional[int]) -> None:
    if seed is None or task_index is None:
        return

    task_seed = int(seed) + int(task_index)
    random.seed(task_seed)
    try:
        import numpy as np
        np.random.seed(task_seed % (2**32 - 1))
    except Exception:
        pass

    # Seed CPU and only the selected CUDA device. torch.manual_seed() and
    # torch.cuda.manual_seed_all() touch every visible GPU, which is expensive
    # and fragile when many process workers are launched.
    torch.default_generator.manual_seed(task_seed)
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.manual_seed(task_seed)


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def _evaluate_network_config(network_config, settings: Dict, gpu_id: Optional[int]) -> Dict:
    task_index = settings.get('task_index')
    verbose = bool(settings.get('verbose_eval_tasks', False))
    task_start = time.perf_counter()
    _log_eval_task(gpu_id, task_index, "start", enabled=verbose)
    try:
        if gpu_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        _seed_eval_task(settings.get('seed'), task_index, gpu_id)

        # Build model from config
        build_start = time.perf_counter()
        _log_eval_task(gpu_id, task_index, "build_model start", enabled=verbose)
        from network_gen import NetworkBuilder
        builder = NetworkBuilder()
        model = builder.build(network_config)
        _log_eval_task(
            gpu_id,
            task_index,
            f"build_model done ({time.perf_counter() - build_start:.2f}s)",
            enabled=verbose,
        )

        # Import zero_cost_proxy functions
        from network_evaluate.zero_cost_proxy import compute_nas_score

        # Compute zero-cost proxy scores with FHE latency
        score_start = time.perf_counter()
        _log_eval_task(gpu_id, task_index, "compute_nas_score start", enabled=verbose)
        scores = compute_nas_score(
            model=model,
            gpu=gpu_id,
            trainloader=None,
            resolution=settings['resolution'],
            batch_size=settings['batch_size'],
            fhe_batch_size=settings.get('fhe_batch_size', 1),
            include_synflow=settings.get('synflow_check', False),
        )
        _log_eval_task(
            gpu_id,
            task_index,
            f"compute_nas_score done ({time.perf_counter() - score_start:.2f}s)",
            enabled=verbose,
        )
        scores.setdefault('evaluation_status', 'ok')

        if settings.get('synflow_check', False):
            synflow_issue = scores.get('synflow_issue')
            if synflow_issue:
                print(f"  ⚠ SynFlow issue detected: {synflow_issue}")

        _log_eval_task(
            gpu_id,
            task_index,
            f"done total={time.perf_counter() - task_start:.2f}s",
            enabled=verbose,
        )
        return scores

    except Exception as e:
        _log_eval_task(gpu_id, task_index, f"error: {type(e).__name__}: {e}")
        print(f"Error evaluating architecture: {e}")
        import traceback
        traceback.print_exc()
        return _invalid_scores(settings.get('synflow_check', False), reason="evaluation_error")

    finally:
        if gpu_id is not None and torch.cuda.is_available():
            empty_start = time.perf_counter()
            _log_eval_task(gpu_id, task_index, "empty_cache start", enabled=verbose)
            torch.cuda.empty_cache()
            _log_eval_task(
                gpu_id,
                task_index,
                f"empty_cache done ({time.perf_counter() - empty_start:.2f}s)",
                enabled=verbose,
            )


def _evaluation_worker_loop(
    task_queue,
    result_queue,
    status_queue,
    settings: Dict,
    gpu_id: Optional[int],
):
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
        verbose = bool(task_settings.get('verbose_eval_tasks', False))
        try:
            if status_queue is not None:
                status_queue.put(("start", os.getpid(), gpu_id, index, time.time()))
            network_config = NetworkConfig.from_dict(config_dict)
            scores = _evaluate_network_config(network_config, task_settings, gpu_id)
            _log_eval_task(gpu_id, index, "result_queue.put start", enabled=verbose)
            result_queue.put((index, scores))
            _log_eval_task(gpu_id, index, "result_queue.put done", enabled=verbose)
            if status_queue is not None:
                status_queue.put(("done", os.getpid(), gpu_id, index, time.time()))
        except Exception:
            import traceback
            traceback.print_exc()
            _log_eval_task(gpu_id, index, "result_queue.put invalid_scores start")
            result_queue.put((
                index,
                _invalid_scores(settings.get('synflow_check', False), reason="worker_error"),
            ))
            _log_eval_task(gpu_id, index, "result_queue.put invalid_scores done")
            if status_queue is not None:
                status_queue.put(("done", os.getpid(), gpu_id, index, time.time()))


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
        self.fhe_batch_size = int(getattr(config.evaluation, "fhe_batch_size", 1))
        self.use_dataloader = config.evaluation.use_dataloader
        self.synflow_check = getattr(config.evaluation, "synflow_check", False)
        self.seed = getattr(config, "seed", None)
        self.latency_baseline = getattr(config.fitness, 'latency_baseline', 22334905.50) if hasattr(config, 'fitness') else 22334905.50
        self.parallel_evaluations = bool(
            getattr(config.evaluation, "parallel_evaluations", True)
        )
        self.max_workers = getattr(config.evaluation, "max_workers", None)
        self.worker_poll_interval = float(
            getattr(config.evaluation, "worker_poll_interval", 30.0)
        )
        self.task_timeout = getattr(config.evaluation, "task_timeout", 600.0)
        if self.task_timeout is not None:
            self.task_timeout = float(self.task_timeout)
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
        print(f"  FHE batch size: {self.fhe_batch_size}")
        print(f"  Latency baseline: {self.latency_baseline}")
        print(f"  Use dataloader: {self.use_dataloader}")
        print(f"  SynFlow check: {self.synflow_check}")
        print(f"  Parallel evaluations: {self.parallel_evaluations}")
        print(f"  Max workers: {self.max_workers}")
        print(f"  Task timeout: {self.task_timeout}")

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
            'fhe_batch_size': self.fhe_batch_size,
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
        worker_gpus = list(self.gpus)
        if self.max_workers is not None:
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
        status_queue = ctx.Queue()
        settings = self._evaluation_settings()

        for index, network_config in pending:
            task_queue.put((index, network_config.to_dict()))
        for _ in range(worker_count):
            task_queue.put(None)

        processes = []
        for gpu_id in worker_gpus[:worker_count]:
            process = ctx.Process(
                target=_evaluation_worker_loop,
                args=(task_queue, result_queue, status_queue, settings, gpu_id),
                name=f"nas-eval-gpu-{gpu_id}",
            )
            process.start()
            processes.append(process)

        pending_indices = {index for index, _ in pending}
        running_tasks: Dict[int, Tuple[Optional[int], int, float]] = {}
        timed_out_indices = set()
        timeout_terminated_pids = set()
        progress_start = time.perf_counter()
        last_progress_line_len = 0

        def _missing_task_labels() -> str:
            missing = [
                str(index + 1)
                for index in sorted(pending_indices)
                if results[index] is None
            ]
            if len(missing) > 20:
                return ", ".join(missing[:20]) + f", ... ({len(missing)} total)"
            return ", ".join(missing)

        def _failed_workers() -> List[Tuple[str, int]]:
            return [
                (process.name, int(process.exitcode))
                for process in processes
                if process.pid not in timeout_terminated_pids
                and process.exitcode not in (None, 0)
            ]

        def _print_progress(done: int, *, final: bool = False) -> None:
            nonlocal last_progress_line_len
            total = len(pending)
            elapsed = time.perf_counter() - progress_start
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = max(0, total - done)
            eta = remaining / rate if rate > 0 else 0.0
            avg = elapsed / done if done > 0 else 0.0
            line = (
                f"[{done}/{total}] done | "
                f"elapsed: {_format_duration(elapsed)} | "
                f"ETA: {_format_duration(eta)} | "
                f"{remaining} remaining | "
                f"{rate:.2f} arch/s | "
                f"avg: {avg:.1f}s/arch"
            )
            padding = " " * max(0, last_progress_line_len - len(line))
            print("\r" + line + padding, end="\n" if final else "", flush=True)
            last_progress_line_len = len(line)

        def _drain_status_queue() -> None:
            while True:
                try:
                    event, pid, gpu_id, index, timestamp = status_queue.get_nowait()
                except queue.Empty:
                    break
                if event == "start":
                    running_tasks[int(pid)] = (gpu_id, int(index), float(timestamp))
                elif event == "done":
                    running_tasks.pop(int(pid), None)

        def _process_by_pid(pid: int):
            for process in processes:
                if process.pid == pid:
                    return process
            return None

        def _mark_timed_out_tasks(now: float) -> None:
            if self.task_timeout is None:
                return
            for pid, (gpu_id, index, started_at) in list(running_tasks.items()):
                if results[index] is not None:
                    running_tasks.pop(pid, None)
                    continue
                elapsed = now - started_at
                if elapsed < self.task_timeout:
                    continue

                print()
                print(
                    f"Parallel NAS evaluation task timeout: Task {index + 1} "
                    f"on GPU {gpu_id} exceeded {self.task_timeout:.0f}s; "
                    "marking invalid and terminating worker",
                    flush=True,
                )
                results[index] = _invalid_scores(self.synflow_check, reason="timeout")
                timed_out_indices.add(index)
                running_tasks.pop(pid, None)
                process = _process_by_pid(pid)
                if process is not None and process.is_alive():
                    timeout_terminated_pids.add(pid)
                    process.terminate()

        received = 0
        try:
            while received < len(pending):
                _drain_status_queue()
                _mark_timed_out_tasks(time.time())
                received = sum(
                    1 for index in pending_indices if results[index] is not None
                )
                if received >= len(pending):
                    break

                try:
                    index, scores = result_queue.get(timeout=self.worker_poll_interval)
                except queue.Empty as exc:
                    _drain_status_queue()
                    _mark_timed_out_tasks(time.time())
                    received = sum(
                        1 for index in pending_indices if results[index] is not None
                    )
                    _print_progress(received)
                    if received >= len(pending):
                        break

                    failed = _failed_workers()
                    if failed:
                        unresolved = [
                            index
                            for index in pending_indices
                            if results[index] is None
                        ]
                        if not unresolved:
                            break
                        raise RuntimeError(
                            "Parallel NAS evaluation worker failed before returning all "
                            f"results: {failed}. Missing tasks: {_missing_task_labels()}"
                        ) from exc

                    if not any(process.is_alive() for process in processes):
                        raise RuntimeError(
                            "All parallel NAS evaluation workers exited before returning "
                            f"all results. Missing tasks: {_missing_task_labels()}"
                        ) from exc

                    continue

                if index in pending_indices and results[index] is None:
                    results[index] = scores
                    received += 1
                elif index in timed_out_indices:
                    print(
                        f"Parallel NAS evaluation ignored late result for timed-out "
                        f"Task {index + 1}",
                        flush=True,
                    )
                else:
                    results[index] = scores
                _print_progress(received)
        except Exception:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=10)
            raise
        finally:
            final_received = sum(
                1 for index in pending_indices if results[index] is not None
            )
            _print_progress(final_received, final=True)
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

            if not (self.parallel_evaluations and len(self.gpus) > 1):
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
