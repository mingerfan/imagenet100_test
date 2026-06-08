#!/usr/bin/env python3
"""Run the recommended two-stage NAS flow end to end."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="results/nas_two_stage")
    parser.add_argument(
        "--evolution-config",
        default="nas_evolution/evolution_config_swish_mbconv.yaml",
    )
    parser.add_argument(
        "--nas-results",
        default=None,
        help="Existing phase-1 evolution directory; skips running evolution.",
    )
    parser.add_argument("--phase1-result-dir", default=None)
    parser.add_argument("--replacement-mask-dir", default=None)
    parser.add_argument("--resume-evolution", default=None)
    parser.add_argument("--skip-evolution", action="store_true")
    parser.add_argument("--skip-phase1-train", action="store_true")
    parser.add_argument("--skip-replacement-planning", action="store_true")
    parser.add_argument("--skip-replacement-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--gpus", nargs="+", default=["all"])
    parser.add_argument("--exclude-gpus", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--network-config", default=None)

    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--phase1-selection", default="top50_middle20_worst20")
    parser.add_argument("--phase1-epochs", type=int, default=12)
    parser.add_argument("--phase1-batch-size", type=int, default=128)
    parser.add_argument("--phase1-learning-rate", type=float, default=7e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--max-parallel-workers", type=int, default=None)
    parser.add_argument("--worker-result-timeout", type=float, default=60.0)

    parser.add_argument(
        "--replacement-source",
        choices=("proxy_promoted", "nas_top"),
        default="proxy_promoted",
        help="Choose source architectures for replacement masks.",
    )
    parser.add_argument("--replacement-source-csv", default=None)
    parser.add_argument("--replacement-arch-count", type=int, default=1)
    parser.add_argument(
        "--actions",
        nargs="+",
        default=None,
        help="Replacement actions; defaults to planner defaults.",
    )
    parser.add_argument("--top-site-actions", type=int, default=6)
    parser.add_argument("--max-replacements", type=int, default=3)
    parser.add_argument("--max-masks", type=int, default=30)
    parser.add_argument("--recompute-fhe", action="store_true")

    parser.add_argument("--replacement-rounds", nargs="+", type=int, default=[2, 10, 20])
    parser.add_argument("--promotion-counts", nargs="+", type=int, default=[8, 3])
    parser.add_argument("--replacement-batch-size", type=int, default=None)
    parser.add_argument("--replacement-learning-rate", type=float, default=None)
    parser.add_argument("--replacement-num-workers", type=int, default=None)
    parser.add_argument("--replacement-prefetch-factor", type=int, default=None)
    parser.add_argument(
        "--replacement-training-preset",
        choices=(
            "auto",
            "replacement_autofhe_degree2",
            "replacement_learned_slow_scale",
            "replacement_smartpaf",
            "replacement_smartpaf_at",
            "replacement_autofhe_pat",
        ),
        default="auto",
    )
    parser.add_argument("--smartpaf-calibration-batches", type=int, default=8)
    parser.add_argument("--smartpaf-ct-steps", type=int, default=300)
    return parser.parse_args()


def run_command(cmd: Sequence[str], *, dry_run: bool) -> None:
    print("\n$ " + shlex.join(str(part) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def add_common_train_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd.extend(["--dataset", args.dataset])
    cmd.extend(["--input-size", str(args.input_size)])
    cmd.extend(["--gpus", *args.gpus])
    cmd.extend(["--seed", str(args.seed)])
    if args.exclude_gpus:
        cmd.extend(["--exclude-gpus", *args.exclude_gpus])
    if args.train_dir:
        cmd.extend(["--train-dir", args.train_dir])
    if args.val_dir:
        cmd.extend(["--val-dir", args.val_dir])
    if args.download:
        cmd.append("--download")
    if args.force:
        cmd.append("--force")
    if args.no_parallel:
        cmd.append("--no-parallel")


def run_evolution(args: argparse.Namespace, evolution_dir: Path) -> None:
    cmd = [
        PYTHON,
        "nas_evolution/run_evolution.py",
        "--config",
        args.evolution_config,
        "--output_dir",
        str(evolution_dir),
        "--gpus",
        *args.gpus,
    ]
    if args.exclude_gpus:
        cmd.extend(["--exclude_gpus", *args.exclude_gpus])
    if args.population_size is not None:
        cmd.extend(["--population_size", str(args.population_size)])
    if args.num_generations is not None:
        cmd.extend(["--num_generations", str(args.num_generations)])
    if args.network_config:
        cmd.extend(["--network_config", args.network_config])
    if args.resume_evolution:
        cmd.extend(["--resume", args.resume_evolution])
    run_command(cmd, dry_run=args.dry_run)


def run_phase1_proxy_train(args: argparse.Namespace, evolution_dir: Path, result_dir: Path) -> Path:
    cmd = [
        PYTHON,
        "tools/train_nas_architectures.py",
        "--nas-results",
        str(evolution_dir),
        "--selection",
        args.phase1_selection,
        "--epochs",
        str(args.phase1_epochs),
        "--batch-size",
        str(args.phase1_batch_size),
        "--learning-rate",
        str(args.phase1_learning_rate),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--worker-result-timeout",
        str(args.worker_result_timeout),
        "--result-dir",
        str(result_dir),
        "--training-preset",
        "swish_proxy",
    ]
    add_common_train_args(cmd, args)
    if args.max_parallel_workers is not None:
        cmd.extend(["--max-parallel-workers", str(args.max_parallel_workers)])
    run_command(cmd, dry_run=args.dry_run)
    return result_dir / "training_results.csv"


def _float_from_row(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def select_proxy_promoted(csv_path: Path, limit: int) -> List[Path]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Phase-1 training CSV not found: {csv_path}")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            json_path = row.get("json_path")
            if json_path:
                rows.append(row)
    rows.sort(key=lambda row: _float_from_row(row, "best_val_acc"), reverse=True)

    selected = []
    seen = set()
    for row in rows:
        path = Path(row["json_path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path in seen or not path.exists():
            continue
        seen.add(path)
        selected.append(path)
        if len(selected) >= limit:
            break
    return selected


def select_nas_top(evolution_dir: Path, limit: int) -> List[Path]:
    best_dir = evolution_dir / "best_models"
    if not best_dir.exists():
        raise FileNotFoundError(f"Best model directory not found: {best_dir}")
    paths = sorted(best_dir.glob("rank*_fitness*.json"), key=lambda p: p.name)
    if not paths:
        paths = sorted(best_dir.glob("*.json"), key=lambda p: p.name)
    return paths[:limit]


def select_replacement_sources(
    args: argparse.Namespace,
    evolution_dir: Path,
    phase1_csv: Path,
) -> List[Path]:
    if args.replacement_source == "nas_top":
        selected = select_nas_top(evolution_dir, args.replacement_arch_count)
    else:
        csv_path = Path(args.replacement_source_csv) if args.replacement_source_csv else phase1_csv
        if not csv_path.is_absolute():
            csv_path = REPO_ROOT / csv_path
        selected = select_proxy_promoted(csv_path, args.replacement_arch_count)
    if not selected:
        raise RuntimeError("No replacement source architectures were selected")
    print("\nReplacement source architectures:")
    for path in selected:
        print(f"  - {path}")
    return selected


def collect_mask_jsons(mask_root: Path) -> List[Path]:
    if not mask_root.exists():
        return []
    excluded_names = {"manifest.json", "replacement_scores.json"}
    paths = []
    for path in sorted(mask_root.rglob("*.json")):
        if path.name in excluded_names:
            continue
        if path.name.endswith("_replacement_scores.json"):
            continue
        paths.append(path)
    return paths


def run_replacement_planning(
    args: argparse.Namespace,
    source_architectures: Iterable[Path],
    mask_root: Path,
) -> List[Path]:
    generated_paths: List[Path] = []
    for arch_path in source_architectures:
        arch_dir = mask_root / arch_path.stem
        score_path = arch_dir / "replacement_scores.json"
        score_csv_path = arch_dir / "replacement_scores.csv"

        score_cmd = [
            PYTHON,
            "tools/nas_replacement_planner.py",
            "score-sites",
            "--arch",
            str(arch_path),
            "--output",
            str(score_path),
            "--csv-output",
            str(score_csv_path),
            "--top-k",
            str(args.top_site_actions),
            "--input-size",
            str(args.input_size),
        ]
        if args.actions:
            score_cmd.extend(["--actions", *args.actions])
        if args.recompute_fhe:
            score_cmd.append("--recompute-fhe")
        run_command(score_cmd, dry_run=args.dry_run)

        mask_cmd = [
            PYTHON,
            "tools/nas_replacement_planner.py",
            "generate-masks",
            "--arch",
            str(arch_path),
            "--scores",
            str(score_path),
            "--output-dir",
            str(arch_dir),
            "--top-site-actions",
            str(args.top_site_actions),
            "--max-replacements",
            str(args.max_replacements),
            "--max-masks",
            str(args.max_masks),
            "--input-size",
            str(args.input_size),
        ]
        if args.actions:
            mask_cmd.extend(["--actions", *args.actions])
        if args.recompute_fhe:
            mask_cmd.append("--recompute-fhe")
        run_command(mask_cmd, dry_run=args.dry_run)

        manifest_path = arch_dir / "manifest.json"
        if not args.dry_run and manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            for item in manifest.get("generated", []):
                path = Path(item["path"])
                if not path.is_absolute():
                    path = REPO_ROOT / path
                generated_paths.append(path)

    return generated_paths


def replacement_preset_for_round(args: argparse.Namespace, epochs: int) -> str:
    if args.replacement_training_preset != "auto":
        return args.replacement_training_preset
    return "replacement_autofhe_degree2" if epochs <= 4 else "replacement_learned_slow_scale"


def run_replacement_training(
    args: argparse.Namespace,
    mask_paths: List[Path],
    run_root: Path,
) -> List[Path]:
    if not mask_paths:
        raise RuntimeError("No replacement mask JSON files found")
    if len(args.promotion_counts) != max(0, len(args.replacement_rounds) - 1):
        raise ValueError("--promotion-counts must have one fewer value than --replacement-rounds")

    csv_paths: List[Path] = []
    previous_csv: Path | None = None
    for round_idx, epochs in enumerate(args.replacement_rounds):
        result_dir = run_root / f"replacement_train_e{epochs}"
        preset = replacement_preset_for_round(args, epochs)
        cmd = [
            PYTHON,
            "tools/train_nas_architectures.py",
            "--epochs",
            str(epochs),
            "--batch-size",
            str(args.replacement_batch_size or args.phase1_batch_size),
            "--learning-rate",
            str(args.replacement_learning_rate or args.phase1_learning_rate),
            "--num-workers",
            str(args.replacement_num_workers or args.num_workers),
            "--prefetch-factor",
            str(args.replacement_prefetch_factor or args.prefetch_factor),
            "--worker-result-timeout",
            str(args.worker_result_timeout),
            "--result-dir",
            str(result_dir),
            "--training-preset",
            preset,
            "--smartpaf-calibration-batches",
            str(args.smartpaf_calibration_batches),
            "--smartpaf-ct-steps",
            str(args.smartpaf_ct_steps),
        ]
        if round_idx == 0:
            cmd.extend(["--json", *[str(path) for path in mask_paths], "--selection", "all"])
        else:
            if previous_csv is None:
                raise RuntimeError("Internal error: missing previous replacement CSV")
            promote_n = args.promotion_counts[round_idx - 1]
            cmd.extend(["--selection", f"promoted{promote_n}", "--training-results", str(previous_csv)])
        add_common_train_args(cmd, args)
        if args.max_parallel_workers is not None:
            cmd.extend(["--max-parallel-workers", str(args.max_parallel_workers)])
        run_command(cmd, dry_run=args.dry_run)
        previous_csv = result_dir / "training_results.csv"
        csv_paths.append(previous_csv)
    return csv_paths


def write_pipeline_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    evolution_dir: Path,
    phase1_csv: Path,
    source_architectures: Sequence[Path],
    mask_paths: Sequence[Path],
    replacement_csvs: Sequence[Path],
) -> None:
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "run_root": str(path.parent),
        "evolution_dir": str(evolution_dir),
        "phase1_training_csv": str(phase1_csv),
        "replacement_source_architectures": [str(path) for path in source_architectures],
        "replacement_masks": [str(path) for path in mask_paths],
        "replacement_training_csvs": [str(path) for path in replacement_csvs],
        "replacement_rounds": args.replacement_rounds,
        "promotion_counts": args.promotion_counts,
        "phase1_training_preset": "swish_proxy",
        "replacement_training_preset": args.replacement_training_preset,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote pipeline manifest: {path}")


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root
    evolution_dir = Path(args.nas_results) if args.nas_results else run_root / "phase1_evolution"
    if not evolution_dir.is_absolute():
        evolution_dir = REPO_ROOT / evolution_dir
    phase1_result_dir = Path(args.phase1_result_dir) if args.phase1_result_dir else run_root / "phase1_proxy"
    if not phase1_result_dir.is_absolute():
        phase1_result_dir = REPO_ROOT / phase1_result_dir
    mask_root = Path(args.replacement_mask_dir) if args.replacement_mask_dir else run_root / "replacement_masks"
    if not mask_root.is_absolute():
        mask_root = REPO_ROOT / mask_root

    skip_evolution = args.skip_evolution or bool(args.nas_results)
    if skip_evolution:
        print(f"Using existing evolution results: {evolution_dir}")
    else:
        run_evolution(args, evolution_dir)

    phase1_csv = phase1_result_dir / "training_results.csv"
    if args.skip_phase1_train:
        print(f"Skipping phase-1 proxy training; using CSV: {phase1_csv}")
    else:
        phase1_csv = run_phase1_proxy_train(args, evolution_dir, phase1_result_dir)

    try:
        source_architectures = select_replacement_sources(args, evolution_dir, phase1_csv)
    except (FileNotFoundError, RuntimeError) as exc:
        if args.dry_run:
            print(f"\nDry run stopped before replacement source selection: {exc}")
            return
        raise

    if args.skip_replacement_planning:
        mask_paths = collect_mask_jsons(mask_root)
        print(f"Skipping replacement planning; found {len(mask_paths)} masks under {mask_root}")
    else:
        mask_paths = run_replacement_planning(args, source_architectures, mask_root)
        if args.dry_run:
            print("\nDry run: replacement mask paths are not available until planning runs.")
            mask_paths = []

    replacement_csvs: List[Path] = []
    if args.skip_replacement_train:
        print("Skipping replacement training.")
    elif not args.dry_run:
        replacement_csvs = run_replacement_training(args, mask_paths, run_root)
    else:
        print("Dry run: skipping replacement training command expansion because mask paths are generated at runtime.")

    if not args.dry_run:
        write_pipeline_manifest(
            run_root / "two_stage_manifest.json",
            args=args,
            evolution_dir=evolution_dir,
            phase1_csv=phase1_csv,
            source_architectures=source_architectures,
            mask_paths=mask_paths,
            replacement_csvs=replacement_csvs,
        )


if __name__ == "__main__":
    main()
