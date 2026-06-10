#!/usr/bin/env python3
"""Train and select Phase-2 replacement-mask candidates.

This is the replacement for the old ablation-JSON workflow. It assumes the
current Phase-2 defaults from "Update phase 2 replacement mask defaults": masks
replace activations only by default and do not add gated/self-gated modules.

The script runs staged short training over existing mask JSONs, promotes a
balanced portfolio between rounds, then reports the best low-latency and best
accuracy/latency-efficiency candidates from the final round.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

EXCLUDED_JSON_NAMES = {"manifest.json", "replacement_scores.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="results/phase2_replacement_selection")
    parser.add_argument(
        "--mask-root",
        default="results/nas_two_stage/replacement_masks",
        help="Root containing generated replacement mask JSONs.",
    )
    parser.add_argument(
        "--json",
        nargs="*",
        default=None,
        help="Explicit mask JSON files. If omitted, --mask-root is scanned.",
    )
    parser.add_argument(
        "--max-initial-masks",
        type=int,
        default=0,
        help="Optional cap for initial masks; 0 means all collected masks.",
    )
    parser.add_argument("--rounds", nargs="+", type=int, default=[2, 10, 20])
    parser.add_argument("--promotion-counts", nargs="+", type=int, default=[8, 3])
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--gpus", nargs="+", default=["all"])
    parser.add_argument("--exclude-gpus", nargs="*", default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--max-parallel-workers", type=int, default=None)
    parser.add_argument("--worker-result-timeout", type=float, default=60.0)
    parser.add_argument(
        "--training-preset",
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
    parser.add_argument("--smartpaf-transition-epochs", type=float, default=None)

    parser.add_argument(
        "--promotion-accuracy-share",
        type=float,
        default=0.375,
        help="Share of each promotion quota reserved for accuracy-biased ranking.",
    )
    parser.add_argument(
        "--promotion-efficiency-share",
        type=float,
        default=0.375,
        help="Share of each promotion quota reserved for efficiency-biased ranking.",
    )
    parser.add_argument(
        "--latency-tradeoff-weight",
        type=float,
        default=0.1,
        help="Accuracy points credited per 1%% latency reduction for accuracy ranking.",
    )
    parser.add_argument(
        "--efficiency-latency-tradeoff-weight",
        type=float,
        default=0.3,
        help="Accuracy points credited per 1%% latency reduction for efficiency ranking.",
    )
    parser.add_argument(
        "--promotion-accuracy-drop",
        type=float,
        default=1.0,
        help="Max accuracy drop in percentage points for latency/efficiency promotion branches.",
    )
    parser.add_argument(
        "--final-accuracy-drop",
        type=float,
        default=1.0,
        help="Max accuracy drop in percentage points for final latency/efficiency recommendations.",
    )
    return parser.parse_args()


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def collect_mask_jsons(args: argparse.Namespace) -> List[Path]:
    if args.json:
        paths = [_resolve_path(path) for path in args.json]
    else:
        mask_root = _resolve_path(args.mask_root)
        paths = []
        if mask_root.exists():
            for path in sorted(mask_root.rglob("*.json")):
                if path.name in EXCLUDED_JSON_NAMES:
                    continue
                if path.name.endswith("_replacement_scores.json"):
                    continue
                paths.append(path)

    deduped: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)

    if args.max_initial_masks > 0:
        deduped = deduped[: args.max_initial_masks]
    return deduped


def _float_from_row(row: Dict, key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0


def _latency_reference(rows: Sequence[Dict]) -> float:
    source_latencies = [
        _float_from_row(row, "source_fhe_latency")
        for row in rows
        if _finite_positive(_float_from_row(row, "source_fhe_latency"))
    ]
    if source_latencies:
        source_latencies.sort()
        return source_latencies[len(source_latencies) // 2]

    latencies = [
        _float_from_row(row, "fhe_latency")
        for row in rows
        if _finite_positive(_float_from_row(row, "fhe_latency"))
    ]
    if not latencies:
        return math.nan
    latencies.sort()
    return latencies[len(latencies) // 2]


def _latency_reduction_pct(row: Dict, fallback_reference: float = math.nan) -> float:
    reduction = _float_from_row(row, "fhe_latency_reduction_pct")
    if math.isfinite(reduction):
        return reduction

    source_latency = _float_from_row(row, "source_fhe_latency")
    latency = _float_from_row(row, "fhe_latency")
    if _finite_positive(source_latency) and math.isfinite(latency):
        return (source_latency - latency) / source_latency * 100.0
    if _finite_positive(fallback_reference) and math.isfinite(latency):
        return (fallback_reference - latency) / fallback_reference * 100.0
    return 0.0


def _accuracy(row: Dict) -> float:
    return _float_from_row(row, "best_val_acc", 0.0)


def _latency(row: Dict) -> float:
    return _float_from_row(row, "fhe_latency", math.inf)


def _score(row: Dict, weight: float, latency_reference: float) -> float:
    return _accuracy(row) + weight * _latency_reduction_pct(row, latency_reference)


def _valid_rows(rows: Sequence[Dict]) -> List[Dict]:
    valid = []
    for row in rows:
        if not row.get("json_path"):
            continue
        if not math.isfinite(_accuracy(row)):
            continue
        if not _finite_positive(_latency(row)):
            continue
        valid.append(row)
    return valid


def load_training_rows(csv_path: Path) -> List[Dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = []
        for idx, row in enumerate(csv.DictReader(f)):
            if not row.get("json_path"):
                continue
            row["_source_order"] = idx
            rows.append(row)
    return rows


def _branch_counts(limit: int, accuracy_share: float, efficiency_share: float) -> Tuple[int, int, int]:
    if limit <= 0:
        return 0, 0, 0
    if accuracy_share < 0 or efficiency_share < 0 or accuracy_share + efficiency_share > 1.0:
        raise ValueError("Promotion shares must be non-negative and sum to <= 1")

    accuracy_count = int(round(limit * accuracy_share))
    efficiency_count = int(round(limit * efficiency_share))
    if accuracy_share > 0 and accuracy_count == 0:
        accuracy_count = 1
    if efficiency_share > 0 and efficiency_count == 0 and limit > 1:
        efficiency_count = 1

    while accuracy_count + efficiency_count > limit:
        if efficiency_count >= accuracy_count and efficiency_count > 0:
            efficiency_count -= 1
        elif accuracy_count > 0:
            accuracy_count -= 1
        else:
            break

    latency_count = limit - accuracy_count - efficiency_count
    if limit >= 3 and latency_count == 0 and (1.0 - accuracy_share - efficiency_share) > 0:
        if efficiency_count >= accuracy_count and efficiency_count > 1:
            efficiency_count -= 1
            latency_count += 1
        elif accuracy_count > 1:
            accuracy_count -= 1
            latency_count += 1
    return accuracy_count, efficiency_count, latency_count


def _pareto_front(rows: Sequence[Dict]) -> List[Dict]:
    front = []
    for row in rows:
        acc = _accuracy(row)
        lat = _latency(row)
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_acc = _accuracy(other)
            other_lat = _latency(other)
            if (
                other_acc >= acc
                and other_lat <= lat
                and (other_acc > acc or other_lat < lat)
            ):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front


def select_portfolio(
    rows: Sequence[Dict],
    limit: int,
    *,
    accuracy_share: float,
    efficiency_share: float,
    latency_tradeoff_weight: float,
    efficiency_latency_tradeoff_weight: float,
    accuracy_drop: float,
) -> List[Dict]:
    valid = _valid_rows(rows)
    if not valid:
        return []
    limit = min(limit, len(valid))
    latency_ref = _latency_reference(valid)
    best_acc = max(_accuracy(row) for row in valid)
    accuracy_floor = best_acc - accuracy_drop
    guarded = [row for row in valid if _accuracy(row) >= accuracy_floor]
    if not guarded:
        guarded = valid

    accuracy_count, efficiency_count, latency_count = _branch_counts(
        limit,
        accuracy_share,
        efficiency_share,
    )
    selected: List[Dict] = []
    seen_paths = set()

    def append_ranked(ranked: Iterable[Dict], count: int, branch: str) -> None:
        for row in ranked:
            if count <= 0:
                return
            path = row.get("json_path")
            if not path or path in seen_paths:
                continue
            copy_row = dict(row)
            copy_row["_selection_branch"] = branch
            selected.append(copy_row)
            seen_paths.add(path)
            count -= 1

    accuracy_ranked = sorted(
        valid,
        key=lambda row: (
            _accuracy(row),
            _score(row, latency_tradeoff_weight, latency_ref),
            -_latency(row),
            -int(row.get("_source_order", 0)),
        ),
        reverse=True,
    )
    efficiency_ranked = sorted(
        guarded,
        key=lambda row: (
            _score(row, efficiency_latency_tradeoff_weight, latency_ref),
            _latency_reduction_pct(row, latency_ref),
            _accuracy(row),
            -_latency(row),
        ),
        reverse=True,
    )
    latency_ranked = sorted(
        guarded,
        key=lambda row: (
            _latency(row),
            -_accuracy(row),
            -_latency_reduction_pct(row, latency_ref),
        ),
    )
    pareto_ranked = sorted(
        _pareto_front(valid),
        key=lambda row: (
            _score(row, efficiency_latency_tradeoff_weight, latency_ref),
            _accuracy(row),
            -_latency(row),
        ),
        reverse=True,
    )

    append_ranked(accuracy_ranked, accuracy_count, "accuracy")
    append_ranked(efficiency_ranked, efficiency_count, "efficiency")
    append_ranked(latency_ranked, latency_count, "latency")
    append_ranked(pareto_ranked, limit - len(selected), "pareto")
    append_ranked(accuracy_ranked, limit - len(selected), "backfill")
    return selected[:limit]


def _record_from_row(row: Dict, label: str | None = None) -> Dict:
    record = {
        "json_path": row.get("json_path", ""),
        "arch_id": row.get("arch_id", ""),
        "category": row.get("category", ""),
        "best_val_acc": _accuracy(row),
        "fhe_latency": _latency(row),
        "source_fhe_latency": _float_from_row(row, "source_fhe_latency"),
        "fhe_latency_reduction_pct": _latency_reduction_pct(row),
        "selection_branch": row.get("_selection_branch", ""),
    }
    if label:
        record["label"] = label
    return record


def select_final_recommendations(
    rows: Sequence[Dict],
    *,
    final_accuracy_drop: float,
    latency_tradeoff_weight: float,
    efficiency_latency_tradeoff_weight: float,
) -> Dict:
    valid = _valid_rows(rows)
    if not valid:
        raise RuntimeError("Final training CSV contains no valid rows with accuracy and latency")

    latency_ref = _latency_reference(valid)
    best_accuracy = max(
        valid,
        key=lambda row: (
            _accuracy(row),
            _score(row, latency_tradeoff_weight, latency_ref),
            -_latency(row),
        ),
    )
    accuracy_floor = _accuracy(best_accuracy) - final_accuracy_drop
    guarded = [row for row in valid if _accuracy(row) >= accuracy_floor]
    if not guarded:
        guarded = valid

    best_latency = min(
        guarded,
        key=lambda row: (
            _latency(row),
            -_accuracy(row),
            -_score(row, efficiency_latency_tradeoff_weight, latency_ref),
        ),
    )
    best_efficiency = max(
        guarded,
        key=lambda row: (
            _score(row, efficiency_latency_tradeoff_weight, latency_ref),
            _latency_reduction_pct(row, latency_ref),
            _accuracy(row),
            -_latency(row),
        ),
    )

    pareto = sorted(
        _pareto_front(valid),
        key=lambda row: (
            _score(row, efficiency_latency_tradeoff_weight, latency_ref),
            _accuracy(row),
            -_latency(row),
        ),
        reverse=True,
    )
    return {
        "accuracy_floor": accuracy_floor,
        "latency_reference": latency_ref,
        "best_accuracy": _record_from_row(best_accuracy, "best_accuracy"),
        "best_latency": _record_from_row(best_latency, "best_latency"),
        "best_efficiency": _record_from_row(best_efficiency, "best_efficiency"),
        "pareto_front": [_record_from_row(row) for row in pareto],
    }


def training_preset_for_round(args: argparse.Namespace, epochs: int) -> str:
    if args.training_preset != "auto":
        return args.training_preset
    return "replacement_autofhe_degree2" if epochs <= 4 else "replacement_learned_slow_scale"


def run_command(cmd: Sequence[str], *, dry_run: bool) -> None:
    print("\n$ " + shlex.join(str(part) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _add_common_training_args(cmd: List[str], args: argparse.Namespace) -> None:
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
    if args.max_parallel_workers is not None:
        cmd.extend(["--max-parallel-workers", str(args.max_parallel_workers)])


def run_training_round(
    args: argparse.Namespace,
    json_paths: Sequence[Path],
    epochs: int,
    result_dir: Path,
) -> Path:
    result_csv = result_dir / "training_results.csv"
    if args.reuse_existing and result_csv.exists():
        print(f"Reusing existing training CSV: {result_csv}")
        return result_csv

    if not json_paths:
        raise RuntimeError(f"No JSON candidates for {epochs}-epoch round")

    cmd = [
        PYTHON,
        "tools/train_nas_architectures.py",
        "--json",
        *[str(path) for path in json_paths],
        "--selection",
        "all",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--worker-result-timeout",
        str(args.worker_result_timeout),
        "--result-dir",
        str(result_dir),
        "--training-preset",
        training_preset_for_round(args, epochs),
        "--smartpaf-calibration-batches",
        str(args.smartpaf_calibration_batches),
        "--smartpaf-ct-steps",
        str(args.smartpaf_ct_steps),
    ]
    if args.smartpaf_transition_epochs is not None:
        cmd.extend(["--smartpaf-transition-epochs", str(args.smartpaf_transition_epochs)])
    _add_common_training_args(cmd, args)
    run_command(cmd, dry_run=args.dry_run)
    return result_csv


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_round_selection(
    path: Path,
    *,
    source_csv: Path,
    selected_rows: Sequence[Dict],
    next_epochs: int,
) -> None:
    payload = {
        "source_csv": str(source_csv),
        "next_epochs": next_epochs,
        "selected": [_record_from_row(row) for row in selected_rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_final_outputs(run_root: Path, final_csv: Path, recommendations: Dict) -> None:
    json_path = run_root / "final_selection.json"
    csv_path = run_root / "final_selection.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "final_training_csv": str(final_csv),
                **recommendations,
            },
            f,
            indent=2,
        )

    rows = [
        recommendations["best_accuracy"],
        recommendations["best_latency"],
        recommendations["best_efficiency"],
    ]
    write_csv(
        csv_path,
        rows,
        [
            "label",
            "json_path",
            "arch_id",
            "category",
            "best_val_acc",
            "fhe_latency",
            "source_fhe_latency",
            "fhe_latency_reduction_pct",
            "selection_branch",
        ],
    )
    paths_path = run_root / "recommended_json_paths.txt"
    seen = set()
    with open(paths_path, "w", encoding="utf-8") as f:
        for row in rows:
            path = row.get("json_path", "")
            if path and path not in seen:
                f.write(path + "\n")
                seen.add(path)

    print(f"\nFinal selection JSON: {json_path}")
    print(f"Final selection CSV: {csv_path}")
    print(f"Recommended JSON paths: {paths_path}")


def main() -> None:
    args = parse_args()
    if len(args.promotion_counts) != max(0, len(args.rounds) - 1):
        raise ValueError("--promotion-counts must have one fewer value than --rounds")

    run_root = _resolve_path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    current_jsons = collect_mask_jsons(args)
    missing = [path for path in current_jsons if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Candidate JSON not found: {missing[0]}")
    if not current_jsons:
        raise RuntimeError("No replacement mask JSONs found")

    manifest = {
        "mask_root": args.mask_root,
        "initial_candidates": [str(path) for path in current_jsons],
        "rounds": args.rounds,
        "promotion_counts": args.promotion_counts,
        "promotion_accuracy_share": args.promotion_accuracy_share,
        "promotion_efficiency_share": args.promotion_efficiency_share,
        "latency_tradeoff_weight": args.latency_tradeoff_weight,
        "efficiency_latency_tradeoff_weight": args.efficiency_latency_tradeoff_weight,
        "promotion_accuracy_drop": args.promotion_accuracy_drop,
        "final_accuracy_drop": args.final_accuracy_drop,
        "round_outputs": [],
    }

    final_csv: Path | None = None
    for round_idx, epochs in enumerate(args.rounds):
        result_dir = run_root / f"train_e{epochs}"
        round_csv = run_training_round(args, current_jsons, epochs, result_dir)
        final_csv = round_csv
        manifest["round_outputs"].append(
            {
                "epochs": epochs,
                "result_dir": str(result_dir),
                "training_csv": str(round_csv),
                "candidate_count": len(current_jsons),
            }
        )
        if args.dry_run:
            break

        rows = load_training_rows(round_csv)
        if round_idx < len(args.rounds) - 1:
            promote_n = args.promotion_counts[round_idx]
            next_epochs = args.rounds[round_idx + 1]
            selected_rows = select_portfolio(
                rows,
                promote_n,
                accuracy_share=args.promotion_accuracy_share,
                efficiency_share=args.promotion_efficiency_share,
                latency_tradeoff_weight=args.latency_tradeoff_weight,
                efficiency_latency_tradeoff_weight=args.efficiency_latency_tradeoff_weight,
                accuracy_drop=args.promotion_accuracy_drop,
            )
            if not selected_rows:
                raise RuntimeError(f"No candidates selected from {round_csv}")
            selection_path = run_root / f"selected_for_e{next_epochs}.json"
            write_round_selection(
                selection_path,
                source_csv=round_csv,
                selected_rows=selected_rows,
                next_epochs=next_epochs,
            )
            manifest["round_outputs"][-1]["next_selection"] = str(selection_path)
            current_jsons = [_resolve_path(row["json_path"]) for row in selected_rows]

    manifest_path = run_root / "phase2_selection_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote run manifest: {manifest_path}")

    if args.dry_run:
        print("Dry run finished before final recommendation scoring.")
        return
    if final_csv is None:
        raise RuntimeError("No training rounds were executed")

    final_rows = load_training_rows(final_csv)
    recommendations = select_final_recommendations(
        final_rows,
        final_accuracy_drop=args.final_accuracy_drop,
        latency_tradeoff_weight=args.latency_tradeoff_weight,
        efficiency_latency_tradeoff_weight=args.efficiency_latency_tradeoff_weight,
    )
    write_final_outputs(run_root, final_csv, recommendations)
    print("\nRecommended candidates:")
    for key in ("best_latency", "best_efficiency"):
        item = recommendations[key]
        print(
            f"  {key}: acc={item['best_val_acc']:.4f}, "
            f"latency={item['fhe_latency']:.2f}, json={item['json_path']}"
        )


if __name__ == "__main__":
    main()
