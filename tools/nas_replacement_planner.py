#!/usr/bin/env python3
"""Plan bounded activation/gating replacement masks for NAS JSON architectures."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


SWISH_TO_STABLEPOLY4 = {
    1: 0,
    3: 2,
    5: 4,
    7: 6,
}
SWISH_TO_GATED_LSWISH = {
    1: 9,
    3: 11,
    5: 13,
    7: 15,
}
BODY_SWISH_MB_CONV_IDS = set(SWISH_TO_STABLEPOLY4)
ACTION_DEFAULTS = ("stablepoly4", "hermitepoly4", "swish_herpn", "gated_lswish")
ACCEPTANCE_RULE = (
    "Keep a mask if best_acc >= baseline_best_acc - 0.5pp, or if "
    "fhe_latency <= 0.9 * baseline_latency and "
    "best_acc >= baseline_best_acc - 1.0pp."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score-sites", help="Score candidate site/action replacements.")
    add_common_arch_args(score)
    score.add_argument("--output", default=None, help="Score JSON path.")
    score.add_argument("--csv-output", default=None, help="Optional score CSV path.")
    score.add_argument("--top-k", type=int, default=6)
    score.add_argument("--actions", nargs="+", default=list(ACTION_DEFAULTS))
    score.add_argument("--include-stem", action="store_true", help="Reserved; body-only by default.")
    score.add_argument("--include-second-downsample", action="store_true", help="Reserved; body-only by default.")
    score.add_argument("--recompute-fhe", action="store_true")
    score.add_argument("--input-size", type=int, default=224)
    score.add_argument("--batch-size", type=int, default=64)

    masks = subparsers.add_parser("generate-masks", help="Generate bounded replacement JSON masks.")
    add_common_arch_args(masks)
    masks.add_argument("--scores", default=None, help="Score JSON from score-sites.")
    masks.add_argument("--output-dir", default="configs/nas_replacement_masks")
    masks.add_argument("--top-site-actions", type=int, default=6)
    masks.add_argument("--max-replacements", type=int, default=3)
    masks.add_argument("--max-masks", type=int, default=30)
    masks.add_argument("--actions", nargs="+", default=list(ACTION_DEFAULTS))
    masks.add_argument("--recompute-fhe", action="store_true")
    masks.add_argument("--input-size", type=int, default=224)
    masks.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def add_common_arch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arch", required=True, help="Source architecture JSON.")


def load_architecture(path: str | Path) -> Tuple[Dict, Dict]:
    arch_path = Path(path)
    with open(arch_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Architecture JSON root must be an object")
    if isinstance(data.get("config"), dict):
        wrapper = data
        config = data["config"]
    else:
        wrapper = {"config": data}
        config = data
    if "blocks" not in config:
        raise ValueError("Architecture config must contain blocks")
    return wrapper, config


def sync_block_choices(config: Dict) -> None:
    config["block_choices"] = [int(block["block_id"]) for block in config["blocks"]]


def apply_action(config: Dict, block_index: int, action: str) -> Dict:
    variant = copy.deepcopy(config)
    block = variant["blocks"][block_index]
    old_id = int(block["block_id"])
    action = action.lower()

    if action == "stablepoly4":
        if old_id not in SWISH_TO_STABLEPOLY4:
            raise ValueError(f"Block {block_index} id={old_id} is not eligible for stablepoly4")
        block["block_id"] = SWISH_TO_STABLEPOLY4[old_id]
        block.pop("activation_override", None)
    elif action == "hermitepoly4":
        if old_id not in BODY_SWISH_MB_CONV_IDS:
            raise ValueError(f"Block {block_index} id={old_id} is not eligible for hermitepoly4")
        block["activation_override"] = "poly4_herpn"
    elif action == "swish_herpn":
        if old_id not in BODY_SWISH_MB_CONV_IDS:
            raise ValueError(f"Block {block_index} id={old_id} is not eligible for swish_herpn")
        block["activation_override"] = "swish_herpn"
    elif action == "gated_lswish":
        if old_id not in SWISH_TO_GATED_LSWISH:
            raise ValueError(f"Block {block_index} id={old_id} is not eligible for gated_lswish")
        block["block_id"] = SWISH_TO_GATED_LSWISH[old_id]
        block.pop("activation_override", None)
    else:
        raise ValueError(f"Unsupported action: {action}")

    sync_block_choices(variant)
    return variant


def apply_plan(config: Dict, replacements: Iterable[Dict]) -> Dict:
    variant = copy.deepcopy(config)
    for repl in replacements:
        variant = apply_action(variant, int(repl["block_index"]), repl["action"])
    return variant


def body_site_features(config: Dict, input_size: int) -> List[Dict]:
    blocks = config["blocks"]
    feature = max(1, int(input_size) // 4)
    if int(config.get("second_ds_code", 5)) != 5:
        feature = max(1, feature // 2)

    features = []
    for idx, block in enumerate(blocks):
        if int(block.get("stride", 1)) == 2:
            feature = max(1, feature // 2)
        out_channels = int(block.get("out_channels", 0))
        cost = float(out_channels * feature * feature)
        position = idx / max(1, len(blocks) - 1)
        sensitivity = (1.0 - position) * 0.35
        if int(block.get("stride", 1)) == 2:
            sensitivity += 0.10
        features.append(
            {
                "block_index": idx,
                "block_id": int(block["block_id"]),
                "out_channels": out_channels,
                "feature_size": feature,
                "site_cost": cost,
                "sensitivity": sensitivity,
                "stride": int(block.get("stride", 1)),
            }
        )
    max_cost = max([item["site_cost"] for item in features] or [1.0])
    for item in features:
        item["site_cost_norm"] = item["site_cost"] / max_cost if max_cost > 0 else 0.0
    return features


def action_prior(action: str) -> Dict:
    table = {
        "stablepoly4": {"fhe_factor": 1.00, "accuracy_prior": -0.15, "risk": 0.35},
        "hermitepoly4": {"fhe_factor": 0.95, "accuracy_prior": -0.10, "risk": 0.25},
        "swish_herpn": {"fhe_factor": 0.20, "accuracy_prior": 0.10, "risk": 0.10},
        "gated_lswish": {"fhe_factor": -0.15, "accuracy_prior": 0.20, "risk": 0.30},
    }
    if action not in table:
        raise ValueError(f"Unsupported action: {action}")
    return table[action]


def heuristic_score(site: Dict, action: str) -> Dict:
    prior = action_prior(action)
    estimated_latency_gain = site["site_cost_norm"] * prior["fhe_factor"]
    risk_penalty = site["sensitivity"] * prior["risk"]
    score = estimated_latency_gain + prior["accuracy_prior"] - risk_penalty
    return {
        "score": float(score),
        "estimated_latency_gain": float(estimated_latency_gain),
        "sensitivity_penalty": float(risk_penalty),
        "score_source": "heuristic",
    }


def build_model_from_config(config: Dict):
    from network_gen.network_config import NetworkConfig
    from network_gen.network_generator import create_network

    return create_network(NetworkConfig.from_dict(config))


def compute_latency(config: Dict, input_size: int, batch_size: int) -> float:
    from network_evaluate.zero_cost_proxy import compute_fhe_latency

    model = build_model_from_config(config)
    metrics = compute_fhe_latency(model, (batch_size, 3, input_size, input_size))
    return float(metrics.get("fhe_latency", float("inf")))


def recomputed_score(
    config: Dict,
    site: Dict,
    action: str,
    baseline_latency: float,
    input_size: int,
    batch_size: int,
) -> Dict:
    variant_config = apply_action(config, site["block_index"], action)
    latency = compute_latency(variant_config, input_size, batch_size)
    if math.isfinite(baseline_latency) and baseline_latency > 0 and math.isfinite(latency):
        actual_gain = (baseline_latency - latency) / baseline_latency
    else:
        actual_gain = 0.0
    fallback = heuristic_score(site, action)
    prior = action_prior(action)
    score = actual_gain + prior["accuracy_prior"] - fallback["sensitivity_penalty"]
    return {
        "score": float(score),
        "estimated_latency_gain": float(actual_gain),
        "sensitivity_penalty": fallback["sensitivity_penalty"],
        "score_source": "recomputed_fhe",
        "estimated_fhe_latency": latency,
    }


def eligible_actions(block_id: int, actions: Iterable[str]) -> List[str]:
    result = []
    for action in actions:
        action = action.lower()
        if action in {"stablepoly4", "gated_lswish"} and block_id in SWISH_TO_STABLEPOLY4:
            result.append(action)
        elif action in {"hermitepoly4", "swish_herpn"} and block_id in BODY_SWISH_MB_CONV_IDS:
            result.append(action)
    return result


def score_sites(args: argparse.Namespace) -> Dict:
    wrapper, config = load_architecture(args.arch)
    baseline_scores = wrapper.get("scores", {}) if isinstance(wrapper.get("scores"), dict) else {}
    baseline_latency = float(baseline_scores.get("fhe_latency", float("nan")))
    if args.recompute_fhe or not math.isfinite(baseline_latency):
        baseline_latency = compute_latency(config, args.input_size, args.batch_size)

    rows = []
    for site in body_site_features(config, args.input_size):
        for action in eligible_actions(site["block_id"], args.actions):
            if args.recompute_fhe:
                score_info = recomputed_score(
                    config,
                    site,
                    action,
                    baseline_latency,
                    args.input_size,
                    args.batch_size,
                )
            else:
                score_info = heuristic_score(site, action)
            row = {
                **site,
                "action": action,
                **score_info,
            }
            rows.append(row)

    rows.sort(key=lambda row: row["score"], reverse=True)
    output = {
        "source_architecture": str(args.arch),
        "baseline_latency": baseline_latency,
        "baseline_scores": baseline_scores,
        "top_k": args.top_k,
        "site_actions": rows,
        "top_site_actions": rows[: args.top_k],
    }
    return output


def write_score_outputs(scores: Dict, output_path: str | None, csv_path: str | None) -> None:
    if output_path:
        path = Path(output_path)
    else:
        src = Path(scores["source_architecture"])
        path = src.with_name(f"{src.stem}_replacement_scores.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(f"Wrote scores: {path}")

    if csv_path:
        rows = scores["site_actions"]
        csv_out = Path(csv_path)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "block_index",
                "block_id",
                "action",
                "score",
                "estimated_latency_gain",
                "sensitivity_penalty",
                "score_source",
                "estimated_fhe_latency",
                "out_channels",
                "feature_size",
                "stride",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote score CSV: {csv_out}")


def load_or_compute_scores(args: argparse.Namespace) -> Dict:
    if args.scores:
        with open(args.scores, "r", encoding="utf-8") as f:
            return json.load(f)
    score_args = argparse.Namespace(
        arch=args.arch,
        actions=args.actions,
        recompute_fhe=args.recompute_fhe,
        input_size=args.input_size,
        batch_size=args.batch_size,
        top_k=args.top_site_actions,
    )
    return score_sites(score_args)


def combo_key(combo: Tuple[Dict, ...]) -> float:
    return sum(float(item["score"]) for item in combo)


def make_variant_name(source_stem: str, mask_idx: int, replacements: List[Dict]) -> str:
    parts = [f"b{r['block_index']}_{r['action']}" for r in replacements]
    suffix = "_".join(parts)
    return f"{source_stem}_mask{mask_idx:02d}_{suffix}"


def generate_masks(args: argparse.Namespace) -> Dict:
    wrapper, config = load_architecture(args.arch)
    scores = load_or_compute_scores(args)
    candidates = scores["site_actions"][: args.top_site_actions]

    combos: List[Tuple[Dict, ...]] = []
    for size in range(1, args.max_replacements + 1):
        for combo in itertools.combinations(candidates, size):
            sites = [item["block_index"] for item in combo]
            if len(set(sites)) != len(sites):
                continue
            combos.append(combo)
    combos.sort(key=combo_key, reverse=True)
    combos = combos[: args.max_masks]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_stem = Path(args.arch).stem
    generated = []

    for mask_idx, combo in enumerate(combos, start=1):
        replacements = [
            {
                "block_index": int(item["block_index"]),
                "from_block_id": int(item["block_id"]),
                "action": item["action"],
                "score": float(item["score"]),
                "score_source": item["score_source"],
            }
            for item in combo
        ]
        variant_config = apply_plan(config, replacements)
        variant_name = make_variant_name(source_stem, mask_idx, replacements)
        variant_config["name"] = variant_name
        variant_config["description"] = (
            "Phase-2 replacement mask generated from a Swish MBConv NAS architecture."
        )

        variant = copy.deepcopy(wrapper)
        variant["config"] = variant_config
        if "scores" in variant:
            variant["source_scores"] = variant.pop("scores")
        variant.pop("zen_fitness", None)
        variant.pop("aznas_fitness", None)
        variant["category"] = "replacement_mask"
        variant["source_architecture"] = str(args.arch)
        variant["variant"] = variant_name
        variant["replacement_plan"] = {
            "replacements": replacements,
            "combined_score": combo_key(combo),
            "acceptance_rule": ACCEPTANCE_RULE,
            "baseline_latency": scores.get("baseline_latency"),
        }

        out_path = output_dir / f"{variant_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(variant, f, indent=2)
        generated.append(
            {
                "path": str(out_path),
                "variant": variant_name,
                "num_replacements": len(replacements),
                "combined_score": combo_key(combo),
                "replacements": replacements,
            }
        )

    manifest = {
        "source_architecture": str(args.arch),
        "scores_source": args.scores or "computed",
        "top_site_actions": args.top_site_actions,
        "max_replacements": args.max_replacements,
        "max_masks": args.max_masks,
        "generated": generated,
        "acceptance_rule": ACCEPTANCE_RULE,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated {len(generated)} masks in {output_dir}")
    print(f"Wrote manifest: {manifest_path}")
    return manifest


def main() -> None:
    args = parse_args()
    if args.command == "score-sites":
        scores = score_sites(args)
        write_score_outputs(scores, args.output, args.csv_output)
        for row in scores["top_site_actions"]:
            print(
                f"b{row['block_index']:02d} {row['action']:<14} "
                f"score={row['score']:+.4f} "
                f"gain={row['estimated_latency_gain']:+.4f} "
                f"source={row['score_source']}"
            )
    elif args.command == "generate-masks":
        generate_masks(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
