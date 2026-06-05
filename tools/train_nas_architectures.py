#!/usr/bin/env python3
"""Short-train NAS JSON architectures with the shared MultiGPUManager."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import get_dataset_info, normalize_dataset_name
from trainers import MultiGPUManager
from utils import parse_gpu_id_list
from utils.nas_training import (
    build_nas_model_configs,
    build_nas_results,
    load_nas_architectures,
    save_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nas-results", default=None, help="Evolution result directory.")
    parser.add_argument(
        "--json",
        nargs="*",
        default=None,
        help="Explicit architecture JSON files. Can be mixed with --nas-results.",
    )
    parser.add_argument(
        "--selection",
        default="top50_middle20_worst20",
        help=(
            "Selection preset: all, topN, middleN, worstN, promotedN, or "
            "top50_middle20_worst20."
        ),
    )
    parser.add_argument(
        "--training-results",
        default=None,
        help="CSV used by promotedN selection; defaults to <nas-results>/training_results.csv.",
    )
    parser.add_argument("--dataset", default="cifar100", help="cifar100/cifar10/imagenet100.")
    parser.add_argument("--train-dir", default=None, help="Training root or CIFAR root.")
    parser.add_argument("--val-dir", default=None, help="Validation root or CIFAR root.")
    parser.add_argument("--download", action="store_true", help="Allow CIFAR download.")
    parser.add_argument("--input-size", type=int, default=224, help="Proxy input size.")
    parser.add_argument("--result-dir", default="results/nas_proxy_cifar100")
    parser.add_argument("--gpus", nargs="+", default=["all"])
    parser.add_argument("--exclude-gpus", nargs="*", default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-freq", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true", default=False)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    return parser.parse_args()


def _fitness(arch: Dict) -> float:
    return float(arch.get("zen_fitness", arch.get("aznas_fitness", 0.0)))


def _load_explicit_jsons(paths: Iterable[str]) -> List[Dict]:
    architectures = []
    for raw_path in paths:
        path = Path(raw_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        architectures.append(
            {
                "category": data.get("category", "json") if isinstance(data, dict) else "json",
                "arch_id": path.stem,
                "json_path": str(path),
                "scores": data.get("scores", {}) if isinstance(data, dict) else {},
                "zen_fitness": _fitness(data) if isinstance(data, dict) else 0.0,
                "generation": data.get("generation", 0) if isinstance(data, dict) else 0,
            }
        )
    return architectures


def _take_by_category(architectures: List[Dict], category: str, count: int) -> List[Dict]:
    aliases = {
        "top": {"top", "best"},
        "best": {"top", "best"},
        "middle": {"middle"},
        "worst": {"worst"},
    }
    allowed = aliases.get(category, {category})
    selected = [arch for arch in architectures if arch["category"] in allowed]
    reverse = category not in {"worst"}
    selected.sort(key=_fitness, reverse=reverse)
    return selected[:count]


def _load_promoted(selection: str, csv_path: Path) -> List[Dict]:
    match = re.fullmatch(r"promoted(\d+)", selection)
    if not match:
        return []
    limit = int(match.group(1))
    if not csv_path.exists():
        raise FileNotFoundError(f"promoted selection needs training CSV: {csv_path}")

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            json_path = row.get("json_path")
            if not json_path:
                continue
            try:
                acc = float(row.get("best_val_acc", 0.0))
            except ValueError:
                acc = 0.0
            row["_best_val_acc"] = acc
            rows.append(row)
    rows.sort(key=lambda row: row["_best_val_acc"], reverse=True)

    promoted = []
    for row in rows[:limit]:
        json_path = row["json_path"]
        promoted.extend(_load_explicit_jsons([json_path]))
        promoted[-1]["category"] = "promoted"
    return promoted


def select_architectures(
    architectures: List[Dict],
    selection: str,
    training_results_csv: Path | None,
) -> List[Dict]:
    selection = selection.strip().lower()
    if selection == "all":
        return architectures

    if selection == "top50_middle20_worst20":
        return (
            _take_by_category(architectures, "top", 50)
            + _take_by_category(architectures, "middle", 20)
            + _take_by_category(architectures, "worst", 20)
        )

    promoted_match = re.fullmatch(r"promoted\d+", selection)
    if promoted_match:
        if training_results_csv is None:
            raise ValueError("promotedN selection requires --training-results or --nas-results")
        return _load_promoted(selection, training_results_csv)

    match = re.fullmatch(r"(top|best|middle|worst)(\d+)", selection)
    if match:
        return _take_by_category(architectures, match.group(1), int(match.group(2)))

    raise ValueError(f"Unsupported selection: {selection}")


def default_data_dirs(dataset_name: str, train_dir: str | None, val_dir: str | None) -> tuple[str, str]:
    if train_dir and val_dir:
        return train_dir, val_dir
    if dataset_name in ("cifar10", "cifar100"):
        root = train_dir or val_dir or "./data"
        return root, root
    if dataset_name == "imagenet100":
        train = train_dir or "/home/xuming/Documents/dataset/imagenet_100/train"
        val = val_dir or "/home/xuming/Documents/dataset/imagenet_100/val"
        return train, val
    raise ValueError(f"Explicit --train-dir/--val-dir required for dataset {dataset_name}")


def main() -> None:
    args = parse_args()
    dataset_name = normalize_dataset_name(args.dataset)
    dataset_info = get_dataset_info(dataset_name)
    train_dir, val_dir = default_data_dirs(dataset_name, args.train_dir, args.val_dir)

    architectures: List[Dict] = []
    if args.nas_results:
        architectures.extend(load_nas_architectures(args.nas_results, categories=["best", "middle", "worst"]))
    if args.json:
        architectures.extend(_load_explicit_jsons(args.json))
    if not architectures and not str(args.selection).startswith("promoted"):
        raise ValueError("No architectures loaded. Provide --nas-results or --json.")

    training_csv = None
    if args.training_results:
        training_csv = Path(args.training_results)
    elif args.nas_results:
        training_csv = Path(args.nas_results) / "training_results.csv"

    selected = select_architectures(architectures, args.selection, training_csv)
    if not selected:
        raise ValueError(f"Selection {args.selection!r} produced no architectures")

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    selected_path = result_dir / "selected_architectures.json"
    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)
    print(f"Selected {len(selected)} architectures -> {selected_path}")

    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "save_checkpoints": args.save_freq > 0,
        "save_freq": args.save_freq,
        "use_amp": args.use_amp,
        "val_force_fp32": True,
        "optimizer_type": "adamw",
        "weight_decay": args.weight_decay,
        "scheduler": "cosine",
        "warmup_epochs": args.warmup_epochs,
        "warmup_start_factor": 0.05,
        "min_lr_ratio": 0.01,
        "grad_clip_max_norm": 1.0,
        "label_smoothing": args.label_smoothing,
        "trainer_kwargs": {
            "collapse_guard_enabled": True,
            "collapse_guard_drop": 10.0,
            "collapse_guard_patience": 1,
            "collapse_guard_action": "stop",
        },
    }
    model_configs, arch_map = build_nas_model_configs(
        selected,
        dataset_num_classes=dataset_info["num_classes"],
        training_config=training_config,
        result_root=str(result_dir / "models"),
    )

    excluded_gpus = parse_gpu_id_list(args.exclude_gpus)
    manager = MultiGPUManager(
        train_dir=train_dir,
        val_dir=val_dir,
        result_dir=str(result_dir / "models"),
        gpus=args.gpus,
        excluded_gpus=excluded_gpus,
        num_classes=dataset_info["num_classes"],
        default_epochs=args.epochs,
        default_batch_size=args.batch_size,
        default_lr=args.learning_rate,
        default_num_workers=args.num_workers,
        use_memory_fs=dataset_info["type"] == "imagefolder",
        dataset=dataset_name,
        download=args.download,
        input_size=args.input_size,
        seed=args.seed,
    )
    train_results = manager.train_models(
        model_configs=model_configs,
        force=args.force,
        parallel=not args.no_parallel,
        return_details=True,
    )
    details = train_results.get("details", {})
    results = build_nas_results(details, arch_map)
    save_results(results, str(result_dir))

    run_summary = {
        "selection": args.selection,
        "selected": len(selected),
        "success": len(train_results.get("success", {})),
        "failed": train_results.get("failed", {}),
        "skipped": train_results.get("skipped", {}),
        "dataset": dataset_name,
        "input_size": args.input_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    with open(result_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    if train_results.get("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
