#!/usr/bin/env python3
"""Select AutoFHE-style polynomial precision candidates from local runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


DEFAULT_CANDIDATES = [
    "configs/proxy_imagenet100_96_pa_ct_fast.yaml",
    "configs/proxy_imagenet100_96_pa_ct_degree2_outscale02_fast.yaml",
    "configs/proxy_imagenet100_96_pa_ct_ss_fast.yaml",
    "configs/proxy_imagenet100_96_pa_ct_ss_degree2_fast.yaml",
    "configs/proxy_imagenet100_96_pa_ct_ss_degree3_fast.yaml",
    "configs/proxy_imagenet100_96_pa_ct_ss_outscale02_fast.yaml",
    "configs/proxy_imagenet100_96_pa_ct_ss_degree2_outscale02_fast.yaml",
    "configs/proxy_imagenet100_96_pa_ct_ds_to_ss_fast.yaml",
    "configs/proxy_imagenet100_96_autofhe_adaptive_degree_fast.yaml",
]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _read_history(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _model_configs(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for item in config.get("models", []) or []:
        if isinstance(item, dict):
            yield item


def _result_history(repo_root: Path, config_path: Path, model_name: str) -> Path:
    result_dir = repo_root / "results" / config_path.stem / model_name
    return result_dir / "train_history.csv"


def _normalize_degrees(raw: Any, default_degree: int, poly_modules: int) -> List[int]:
    if raw is None:
        return [int(default_degree)] * poly_modules
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, (list, tuple)):
        degrees = [int(value) for value in raw]
        if len(degrees) < poly_modules:
            degrees.extend([int(default_degree)] * (poly_modules - len(degrees)))
        return degrees[:poly_modules]
    if isinstance(raw, dict):
        degrees = []
        for idx in range(poly_modules):
            value = (
                raw.get(str(idx))
                or raw.get(str(idx + 1))
                or raw.get(f"module_{idx}")
                or raw.get(f"module_{idx + 1}")
                or default_degree
            )
            degrees.append(int(value))
        return degrees
    raise ValueError(f"Unsupported degree spec: {raw!r}")


def _poly_depth(degree: int) -> int:
    return int(math.ceil(math.log2(max(1, int(degree)))))


def _summarize_rows(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {"epochs": 0, "best_acc": 0.0, "final_acc": 0.0, "status": "EMPTY"}
    val_acc = [_to_float(row.get("val_acc")) for row in rows]
    max_drop = max((prev - cur for prev, cur in zip(val_acc, val_acc[1:])), default=0.0)
    nonfinite = sum(
        int(_to_float(row.get("nonfinite_train_batches"), 0))
        + int(_to_float(row.get("nonfinite_val_batches"), 0))
        for row in rows
    )
    guard = sum(int(_to_float(row.get("collapse_guard_triggered"), 0)) for row in rows)
    status = "PASS"
    if nonfinite:
        status = "NONFINITE"
    if max_drop >= 10.0 or guard:
        status = "COLLAPSE"
    return {
        "epochs": len(rows),
        "best_acc": max(val_acc),
        "final_acc": val_acc[-1],
        "max_drop": max_drop,
        "nonfinite": nonfinite,
        "guard": guard,
        "status": status,
    }


def collect_candidates(repo_root: Path, config_paths: Iterable[Path], poly_modules: int) -> List[Dict[str, Any]]:
    rows = []
    for config_path in config_paths:
        absolute_config = config_path if config_path.is_absolute() else repo_root / config_path
        if not absolute_config.exists():
            continue
        config = _read_yaml(absolute_config)
        rel_config = absolute_config.relative_to(repo_root)
        for model in _model_configs(config):
            model_name = str(model["name"])
            trainer_kwargs = dict(model.get("trainer_kwargs", {}) or {})
            degree = int(trainer_kwargs.get("poly4_degree", 4) or 4)
            degrees = _normalize_degrees(trainer_kwargs.get("poly4_degrees"), degree, poly_modules)
            scale_mode = str(trainer_kwargs.get("poly4_scale_mode", "learned"))
            output_scale = trainer_kwargs.get("poly4_output_scale", 0.1)
            history = _result_history(repo_root, rel_config, model_name)
            summary = _summarize_rows(_read_history(history)) if history.exists() else {
                "epochs": 0,
                "best_acc": 0.0,
                "final_acc": 0.0,
                "max_drop": 0.0,
                "nonfinite": 0,
                "guard": 0,
                "status": "MISSING",
            }
            depth_sum = sum(_poly_depth(item) for item in degrees)
            rows.append({
                "model": model_name,
                "config": str(rel_config),
                "history": str(history.relative_to(repo_root)) if history.exists() else "",
                "scale_mode": scale_mode,
                "degrees": degrees,
                "output_scale": float(output_scale),
                "depth_sum": depth_sum,
                "deployable_static": scale_mode in {"static", "dynamic"},
                **summary,
            })
    return rows


def pareto_front(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = [row for row in candidates if row["status"] == "PASS"]
    front = []
    for row in valid:
        dominated = False
        for other in valid:
            if other is row:
                continue
            if (
                other["best_acc"] >= row["best_acc"]
                and other["depth_sum"] <= row["depth_sum"]
                and (
                    other["best_acc"] > row["best_acc"]
                    or other["depth_sum"] < row["depth_sum"]
                )
            ):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda item: (-item["best_acc"], item["depth_sum"]))


def choose_recommendation(candidates: List[Dict[str, Any]], tolerance: float) -> Optional[Dict[str, Any]]:
    valid = [row for row in candidates if row["status"] == "PASS"]
    if not valid:
        return None
    best_acc = max(row["best_acc"] for row in valid)
    near_best = [row for row in valid if row["best_acc"] >= best_acc - tolerance]
    return sorted(near_best, key=lambda item: (item["depth_sum"], -item["best_acc"], item["model"]))[0]


def print_table(rows: List[Dict[str, Any]]) -> None:
    header = ("model", "best", "final", "depth", "degrees", "scale", "status")
    print("{:<54s} {:>7s} {:>7s} {:>5s} {:>10s} {:>8s} {:>9s}".format(*header))
    print("-" * 112)
    for row in sorted(rows, key=lambda item: (-item["best_acc"], item["depth_sum"], item["model"])):
        print(
            "{:<54s} {:>7.2f} {:>7.2f} {:>5d} {:>10s} {:>8s} {:>9s}".format(
                row["model"][:54],
                row["best_acc"],
                row["final_acc"],
                row["depth_sum"],
                ",".join(str(item) for item in row["degrees"]),
                row["scale_mode"][:8],
                row["status"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root")
    parser.add_argument("--poly-modules", type=int, default=2, help="StablePoly module count in this proxy")
    parser.add_argument(
        "--accuracy-tolerance",
        type=float,
        default=0.25,
        help="Accuracy gap in percentage points allowed when preferring lower depth",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON")
    parser.add_argument("configs", nargs="*", help="Config paths. Defaults to known precision proxies.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    configs = [Path(item) for item in (args.configs or DEFAULT_CANDIDATES)]
    candidates = collect_candidates(repo_root, configs, args.poly_modules)
    front = pareto_front(candidates)
    recommendation = choose_recommendation(candidates, args.accuracy_tolerance)
    payload = {
        "candidates": candidates,
        "pareto_front": front,
        "recommendation": recommendation,
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print_table(candidates)
    print()
    if recommendation:
        print("Recommended AutoFHE-style precision candidate:")
        print(
            f"  {recommendation['model']} "
            f"(best={recommendation['best_acc']:.2f}, depth_sum={recommendation['depth_sum']}, "
            f"degrees={recommendation['degrees']}, scale={recommendation['scale_mode']}, "
            f"tolerance={args.accuracy_tolerance:.2f}pp)"
        )
    else:
        print("No passing candidate found.")


if __name__ == "__main__":
    main()
