#!/usr/bin/env python3
"""Create NAS phase-2 ablation variants and a full-training YAML."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

POLY_TO_NON_POLY_BLOCK = {
    0: 1,
    2: 3,
    4: 5,
    6: 7,
    8: 9,
    10: 11,
    12: 13,
    14: 15,
    16: 17,
    18: 19,
    20: 21,
}

SELF_GATED_TO_PLAIN_BLOCK = {
    8: 0,
    9: 1,
    10: 2,
    11: 3,
    12: 4,
    13: 5,
    14: 6,
    15: 7,
    18: 16,
    19: 17,
    20: 16,
    21: 17,
}

POLY_OVERRIDES = {"poly4", "poly4_herpn", "hermitepoly4", "stablepoly4"}

VARIANT_KINDS = (
    "source",
    "mask",
    "no_poly",
    "no_selfgated",
    "no_poly_no_selfgated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-results",
        required=True,
        help="Final phase-2 replacement training CSV, usually replacement_train_e20/training_results.csv.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for generated ablation JSONs.")
    parser.add_argument("--config-output", required=True, help="Full-training YAML to write.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of final candidates to ablate.")
    parser.add_argument(
        "--sort-metric",
        choices=("accuracy", "accuracy_latency"),
        default="accuracy_latency",
        help="How to rank candidates from --training-results.",
    )
    parser.add_argument(
        "--latency-tradeoff-weight",
        type=float,
        default=0.1,
        help="Accuracy points credited per 1%% latency reduction for accuracy_latency sorting.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_KINDS,
        default=list(VARIANT_KINDS),
        help="Ablation variants to generate.",
    )

    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--smartpaf-transition-epochs", type=float, default=8.0)
    parser.add_argument("--smartpaf-calibration-batches", type=int, default=8)
    parser.add_argument("--smartpaf-ct-steps", type=int, default=300)
    parser.add_argument(
        "--include-duplicate-models",
        action="store_true",
        help="Include structurally duplicate variants in the training YAML.",
    )
    return parser.parse_args()


def _float_from_row(row: Dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _latency_reduction_pct(row: Dict) -> float:
    reduction = _float_from_row(row, "fhe_latency_reduction_pct", math.nan)
    if math.isfinite(reduction):
        return reduction
    source_latency = _float_from_row(row, "source_fhe_latency", math.nan)
    latency = _float_from_row(row, "fhe_latency", math.nan)
    if math.isfinite(source_latency) and source_latency > 0 and math.isfinite(latency):
        return (source_latency - latency) / source_latency * 100.0
    return 0.0


def _candidate_score(row: Dict, sort_metric: str, latency_tradeoff_weight: float) -> float:
    acc = _float_from_row(row, "best_val_acc")
    if sort_metric == "accuracy_latency":
        return acc + latency_tradeoff_weight * _latency_reduction_pct(row)
    return acc


def _load_ranked_rows(args: argparse.Namespace) -> List[Dict]:
    csv_path = Path(args.training_results)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("json_path")]
    rows.sort(
        key=lambda row: _candidate_score(
            row,
            args.sort_metric,
            args.latency_tradeoff_weight,
        ),
        reverse=True,
    )
    return rows[: args.top_k]


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path
    candidates = [Path.cwd() / path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _config(wrapper: Dict) -> Dict:
    config = wrapper.get("config")
    if not isinstance(config, dict):
        raise ValueError("Architecture JSON must contain a config object")
    return config


def _replacement_indices(mask_wrapper: Dict) -> List[int]:
    plan = mask_wrapper.get("replacement_plan", {})
    replacements = plan.get("replacements", []) if isinstance(plan, dict) else []
    indices = []
    for replacement in replacements:
        try:
            indices.append(int(replacement["block_index"]))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(set(indices))


def _sync_block_choices(config: Dict) -> None:
    blocks = config.get("blocks", [])
    config["block_choices"] = [int(block["block_id"]) for block in blocks]


def _strip_poly_at_indices(config: Dict, indices: Iterable[int]) -> None:
    blocks = config.get("blocks", [])
    for index in indices:
        if index < 0 or index >= len(blocks):
            continue
        block = blocks[index]
        block_id = int(block["block_id"])
        if block_id in POLY_TO_NON_POLY_BLOCK:
            block["block_id"] = POLY_TO_NON_POLY_BLOCK[block_id]
        override = str(block.get("activation_override", "")).strip().lower()
        if override in POLY_OVERRIDES or override.startswith("poly"):
            block.pop("activation_override", None)
    _sync_block_choices(config)


def _strip_selfgated_at_indices(config: Dict, indices: Iterable[int]) -> None:
    blocks = config.get("blocks", [])
    for index in indices:
        if index < 0 or index >= len(blocks):
            continue
        block = blocks[index]
        block_id = int(block["block_id"])
        if block_id in SELF_GATED_TO_PLAIN_BLOCK:
            block["block_id"] = SELF_GATED_TO_PLAIN_BLOCK[block_id]
    _sync_block_choices(config)


def _source_from_mask(mask_wrapper: Dict) -> Dict:
    source_path = mask_wrapper.get("source_architecture")
    if source_path:
        resolved = _resolve_path(str(source_path))
        if resolved.exists():
            return _load_json(resolved)

    source = copy.deepcopy(mask_wrapper)
    config = _config(source)
    plan = mask_wrapper.get("replacement_plan", {})
    replacements = plan.get("replacements", []) if isinstance(plan, dict) else []
    for replacement in replacements:
        try:
            index = int(replacement["block_index"])
            from_block_id = int(replacement["from_block_id"])
        except (KeyError, TypeError, ValueError):
            continue
        blocks = config.get("blocks", [])
        if 0 <= index < len(blocks):
            blocks[index]["block_id"] = from_block_id
            blocks[index].pop("activation_override", None)
    _sync_block_choices(config)
    source.pop("scores", None)
    return source


def _sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    value = value.strip("-_.")
    return value or "candidate"


def _config_hash(wrapper: Dict) -> str:
    config = copy.deepcopy(_config(wrapper))
    for key in ("name", "description", "created_at"):
        config.pop(key, None)
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _has_poly_activation(wrapper: Dict) -> bool:
    for block in _config(wrapper).get("blocks", []):
        block_id = int(block.get("block_id", -1))
        if block_id in POLY_TO_NON_POLY_BLOCK:
            return True
        override = str(block.get("activation_override", "")).strip().lower()
        if override in POLY_OVERRIDES or override.startswith("poly"):
            return True
    return False


def _prepare_wrapper(
    wrapper: Dict,
    *,
    candidate_label: str,
    variant_kind: str,
    source_json_path: str,
    mask_json_path: str,
) -> Dict:
    output = copy.deepcopy(wrapper)
    config = _config(output)
    name = f"{candidate_label}_{variant_kind}"
    config["name"] = name
    config["description"] = (
        f"NAS ablation variant '{variant_kind}' generated from {mask_json_path}."
    )
    output["category"] = "nas_ablation"
    output["ablation_variant"] = variant_kind
    output["source_architecture"] = source_json_path
    output["mask_architecture"] = mask_json_path
    output.pop("zen_fitness", None)
    output.pop("aznas_fitness", None)
    if variant_kind not in {"source", "mask"}:
        output.pop("scores", None)
    return output


def _make_variants(mask_path: Path, candidate_label: str) -> Dict[str, Dict]:
    mask_wrapper = _load_json(mask_path)
    source_wrapper = _source_from_mask(mask_wrapper)
    indices = _replacement_indices(mask_wrapper)
    source_path = str(mask_wrapper.get("source_architecture", ""))

    variants: Dict[str, Dict] = {}
    variants["source"] = _prepare_wrapper(
        source_wrapper,
        candidate_label=candidate_label,
        variant_kind="source",
        source_json_path=source_path,
        mask_json_path=str(mask_path),
    )
    variants["mask"] = _prepare_wrapper(
        mask_wrapper,
        candidate_label=candidate_label,
        variant_kind="mask",
        source_json_path=source_path,
        mask_json_path=str(mask_path),
    )

    no_poly = _prepare_wrapper(
        mask_wrapper,
        candidate_label=candidate_label,
        variant_kind="no_poly",
        source_json_path=source_path,
        mask_json_path=str(mask_path),
    )
    _strip_poly_at_indices(_config(no_poly), indices)
    variants["no_poly"] = no_poly

    no_selfgated = _prepare_wrapper(
        mask_wrapper,
        candidate_label=candidate_label,
        variant_kind="no_selfgated",
        source_json_path=source_path,
        mask_json_path=str(mask_path),
    )
    _strip_selfgated_at_indices(_config(no_selfgated), indices)
    variants["no_selfgated"] = no_selfgated

    no_poly_no_selfgated = _prepare_wrapper(
        mask_wrapper,
        candidate_label=candidate_label,
        variant_kind="no_poly_no_selfgated",
        source_json_path=source_path,
        mask_json_path=str(mask_path),
    )
    config = _config(no_poly_no_selfgated)
    _strip_poly_at_indices(config, indices)
    _strip_selfgated_at_indices(config, indices)
    variants["no_poly_no_selfgated"] = no_poly_no_selfgated

    return variants


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _model_entry(
    name: str,
    json_path: Path,
    wrapper: Dict,
    args: argparse.Namespace,
) -> Dict:
    entry = {
        "name": name,
        "class": "nas-json",
        "json_path": _relative_path(json_path),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "use_amp": False,
        "val_force_fp32": True,
        "save_freq": 0,
        "optimizer_type": "adamw",
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "warmup_epochs": args.warmup_epochs,
        "warmup_start_factor": 0.05,
        "min_lr_ratio": 0.01,
        "grad_clip_max_norm": 1.0,
        "label_smoothing": 0.05,
        "params": {
            "num_classes": args.num_classes,
            "pretrained": False,
        },
    }
    if _has_poly_activation(wrapper):
        entry.update(
            {
                "poly_weight_decay": 0.0,
                "poly_scale_lr_mult": 0.1,
                "trainer_kwargs": {
                    "gate_reg_lambda": 0.0,
                    "poly4_warmup_ratio": 0.35,
                    "poly4_scale_mode": "learned",
                    "poly4_degree": 2,
                    "poly4_output_scale": 0.2,
                    "smartpaf_ct_init": True,
                    "smartpaf_ct_batches": args.smartpaf_calibration_batches,
                    "smartpaf_ct_max_samples": 20000,
                    "smartpaf_ct_steps": args.smartpaf_ct_steps,
                    "smartpaf_ct_lr": 0.01,
                    "smartpaf_progressive": True,
                    "smartpaf_group_epochs": "auto",
                    "smartpaf_transition_epochs": args.smartpaf_transition_epochs,
                    "smartpaf_alternate_training": False,
                    "collapse_guard_enabled": True,
                    "collapse_guard_drop": 10.0,
                    "collapse_guard_patience": 1,
                    "collapse_guard_action": "stop",
                },
            }
        )
    else:
        entry["trainer_kwargs"] = {
            "collapse_guard_enabled": True,
            "collapse_guard_drop": 10.0,
            "collapse_guard_patience": 1,
            "collapse_guard_action": "stop",
        }
    return entry


def _write_yaml(path: Path, model_entries: Sequence[Dict], args: argparse.Namespace) -> None:
    config = {
        "global": {
            "num_classes": args.num_classes,
            "default_epochs": args.epochs,
            "default_batch_size": args.batch_size,
            "default_learning_rate": args.learning_rate,
            "default_num_workers": args.num_workers,
            "default_save_freq": 0,
        },
        "json_models": list(model_entries),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    config_output = Path(args.config_output)
    if not config_output.is_absolute():
        config_output = REPO_ROOT / config_output
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = _load_ranked_rows(args)
    if not selected_rows:
        raise RuntimeError("No candidate rows found in training results CSV")

    manifest = {
        "training_results": args.training_results,
        "top_k": args.top_k,
        "sort_metric": args.sort_metric,
        "latency_tradeoff_weight": args.latency_tradeoff_weight,
        "variants": [],
        "training_config": str(config_output),
    }
    model_entries: List[Dict] = []
    seen_hashes: Dict[str, str] = {}

    for candidate_idx, row in enumerate(selected_rows, start=1):
        mask_path = _resolve_path(row["json_path"])
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask JSON not found: {mask_path}")
        candidate_label = f"cand{candidate_idx:02d}_{_sanitize_name(mask_path.stem)}"
        variants = _make_variants(mask_path, candidate_label)

        for variant_kind in args.variants:
            wrapper = variants[variant_kind]
            variant_name = f"{candidate_label}_{variant_kind}"
            out_path = output_dir / f"{variant_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(wrapper, f, indent=2)

            cfg_hash = _config_hash(wrapper)
            duplicate_of = seen_hashes.get(cfg_hash)
            train_included = args.include_duplicate_models or duplicate_of is None
            if train_included:
                seen_hashes.setdefault(cfg_hash, variant_name)
                model_entries.append(_model_entry(variant_name, out_path, wrapper, args))

            manifest["variants"].append(
                {
                    "candidate_rank": candidate_idx,
                    "variant": variant_kind,
                    "name": variant_name,
                    "path": str(out_path),
                    "mask_json": str(mask_path),
                    "best_val_acc": row.get("best_val_acc", ""),
                    "fhe_latency": row.get("fhe_latency", ""),
                    "fhe_latency_reduction_pct": row.get("fhe_latency_reduction_pct", ""),
                    "config_hash": cfg_hash,
                    "train_included": train_included,
                    "duplicate_of": duplicate_of,
                }
            )

    manifest_path = output_dir / "ablation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _write_yaml(config_output, model_entries, args)

    print(f"Wrote {len(manifest['variants'])} ablation JSONs under {output_dir}")
    print(f"Wrote training YAML with {len(model_entries)} unique models: {config_output}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
