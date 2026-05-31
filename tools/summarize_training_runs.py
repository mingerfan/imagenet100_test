#!/usr/bin/env python3
"""Summarize training histories for stability/accuracy ablations."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _to_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_history(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _summarize_history(path: Path, baseline: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    rows = _read_history(path)
    if not rows:
        return {
            "model": path.parent.name,
            "history": str(path),
            "epochs": 0,
            "status": "EMPTY",
        }

    val_acc = [_to_float(row.get("val_acc")) for row in rows]
    train_acc = [_to_float(row.get("train_acc")) for row in rows]
    best_acc = max(val_acc) if val_acc else 0.0
    final_acc = val_acc[-1] if val_acc else 0.0
    final_train_acc = train_acc[-1] if train_acc else 0.0
    max_drop = 0.0
    for prev, cur in zip(val_acc, val_acc[1:]):
        max_drop = max(max_drop, prev - cur)

    nonfinite_train = sum(int(_to_float(row.get("nonfinite_train_batches"), 0)) for row in rows)
    nonfinite_val = sum(int(_to_float(row.get("nonfinite_val_batches"), 0)) for row in rows)
    skipped_train = sum(int(_to_float(row.get("train_skipped_batches"), 0)) for row in rows)
    skipped_val = sum(int(_to_float(row.get("val_skipped_batches"), 0)) for row in rows)
    collapse_hits = sum(int(_to_float(row.get("collapse_guard_triggered"), 0)) for row in rows)
    collapse_checkpoints = list(path.parent.glob("collapse_epoch_*.pth"))
    collapse_hits = max(collapse_hits, len(collapse_checkpoints))

    status = "PASS"
    if nonfinite_train or nonfinite_val:
        status = "NONFINITE"
    if max_drop >= 10.0 or collapse_checkpoints:
        status = "COLLAPSE"

    baseline_delta = None
    model = path.parent.name
    if baseline and model in baseline:
        baseline_delta = best_acc - baseline[model]

    return {
        "model": model,
        "history": str(path),
        "epochs": len(rows),
        "best_acc": best_acc,
        "final_acc": final_acc,
        "final_train_acc": final_train_acc,
        "best_final_gap": best_acc - final_acc,
        "max_epoch_drop": max_drop,
        "nonfinite_train_batches": nonfinite_train,
        "nonfinite_val_batches": nonfinite_val,
        "train_skipped_batches": skipped_train,
        "val_skipped_batches": skipped_val,
        "collapse_guard_hits": collapse_hits,
        "collapse_checkpoints": len(collapse_checkpoints),
        "baseline_delta": baseline_delta,
        "status": status,
    }


def _load_baseline(path: Optional[str]) -> Optional[Dict[str, float]]:
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    baseline = {}
    for name, item in data.items():
        if isinstance(item, dict):
            acc = item.get("best_val_acc") or item.get("best_acc") or item.get("accuracy")
            if acc is not None:
                baseline[name] = float(acc) * (100.0 if float(acc) <= 1.0 else 1.0)
    return baseline


def _iter_history_files(paths: Iterable[str]) -> Iterable[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from path.rglob("train_history.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Result directories or train_history.csv files")
    parser.add_argument("--baseline-json", default=None, help="Optional JSON with historical best accuracies")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of table")
    args = parser.parse_args()

    baseline = _load_baseline(args.baseline_json)
    summaries = [_summarize_history(path, baseline) for path in _iter_history_files(args.paths)]
    summaries.sort(key=lambda row: (row.get("status") != "PASS", -float(row.get("best_acc", 0.0))))

    if args.json:
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
        return

    header = (
        "model", "epochs", "best", "final", "gap", "max_drop",
        "nonfinite", "skipped", "guard", "status"
    )
    print("{:<45s} {:>6s} {:>7s} {:>7s} {:>7s} {:>9s} {:>10s} {:>8s} {:>5s} {:>10s}".format(*header))
    print("-" * 130)
    for row in summaries:
        nonfinite = int(row.get("nonfinite_train_batches", 0)) + int(row.get("nonfinite_val_batches", 0))
        skipped = int(row.get("train_skipped_batches", 0)) + int(row.get("val_skipped_batches", 0))
        print(
            "{:<45s} {:>6d} {:>7.2f} {:>7.2f} {:>7.2f} {:>9.2f} {:>10d} {:>8d} {:>5d} {:>10s}".format(
                str(row.get("model", ""))[:45],
                int(row.get("epochs", 0)),
                float(row.get("best_acc", 0.0)),
                float(row.get("final_acc", 0.0)),
                float(row.get("best_final_gap", 0.0)),
                float(row.get("max_epoch_drop", 0.0)),
                nonfinite,
                skipped,
                int(row.get("collapse_guard_hits", 0)),
                str(row.get("status", "")),
            )
        )


if __name__ == "__main__":
    main()
