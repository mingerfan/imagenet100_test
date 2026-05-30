#!/usr/bin/env python3
"""Analyze FHE latency for best_models and compare with common baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torchvision

# Ensure repo root is on sys.path when executed as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_gen.network_config import NetworkConfig
from network_gen.network_generator import create_network
from fhe_statistics import FheInfo


def _summarize_info(info: FheInfo) -> Dict[str, float]:
    total_with_boot = info.total_latency + info.total_boot_latency
    boot_fraction = (info.total_boot_latency / total_with_boot) if total_with_boot > 0 else 0.0
    return {
        "total_latency": float(total_with_boot),
        "op_latency": float(info.total_latency),
        "boot_latency": float(info.total_boot_latency),
        "boot_count": float(info.total_boot_count),
        "boot_fraction": float(boot_fraction),
        "max_depth": float(info.get_max_depth()),
        "params": float(info.get_parameter_count()),
        "flops": float(info.get_flops_count()),
    }


def _op_breakdown(info: FheInfo) -> Dict[str, float]:
    return {
        op: float(stats["latency"] + stats["boot_latency"])
        for op, stats in info.op_stats.items()
    }


def _top_ops(breakdown: Dict[str, float], top_n: int) -> List[Tuple[str, float]]:
    return sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)[: max(0, top_n)]


def _analyze_model(name: str, model) -> Tuple[FheInfo, Dict[str, float], Dict[str, float]]:
    info = FheInfo(model, model_name=name)
    info.run_statistics()
    return info, _summarize_info(info), _op_breakdown(info)


def _load_best_models(best_models_dir: Path) -> List[Tuple[str, str, object]]:
    results: List[Tuple[str, str, object]] = []
    for path in sorted(best_models_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = NetworkConfig.from_dict(data["config"])
        model = create_network(cfg)
        results.append((path.stem, cfg.name or path.stem, model))
    return results


def _baseline_models() -> List[Tuple[str, object]]:
    return [
        ("ResNet18", torchvision.models.resnet18()),
        ("ResNet34", torchvision.models.resnet34()),
        ("EfficientNet_B0", torchvision.models.efficientnet_b0()),
    ]


def _pick_first_depths(info: FheInfo, k: int = 2) -> List[int]:
    depths = sorted({meta.out_depth for meta in info.node_meta_list.values() if not meta.is_fused})
    return depths[: max(0, k)]


def _apply_plot_style():
    try:
        import scienceplots  # noqa: F401
        import matplotlib.pyplot as plt
        plt.style.use(["science", "ieee"])
    except Exception:
        import matplotlib.pyplot as plt
        plt.style.use("default")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "text.usetex": False,
        "font.serif": ["Times New Roman"],
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "lines.linewidth": 0.8,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })


def _plot_boot_fraction(summary_rows, save_path: str):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skipping boot fraction plot")
        return

    _apply_plot_style()

    models = [row["model"] for row in summary_rows]
    boot_pct = [float(row["boot_fraction"]) * 100 for row in summary_rows]

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)

    bars = ax.bar(
        x,
        boot_pct,
        width=0.6,
        color="#2F7FB8",
        edgecolor="black",
        linewidth=0.6,
        label="Boot Latency (%)",
    )

    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{boot_pct[i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#1f4e79",
        )

    ax.set_ylabel("Boot Fraction (%)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    ax.set_ylim(0, max(boot_pct + [1.0]) * 1.2)
    ax.legend(frameon=False, loc="upper left", fontsize=11, prop={"weight": "bold"})

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Boot fraction plot saved to {save_path}")
    plt.close(fig)


def _plot_combined(front_metrics, summary_rows, save_path: str):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skipping combined plot")
        return

    _apply_plot_style()

    model_names = list(front_metrics.keys())
    gmacs = [front_metrics[m].get("front_flops_pct", 0.0) for m in model_names]
    latency = [front_metrics[m].get("front_latency_pct", 0.0) for m in model_names]

    boot_pct = [float(row["boot_fraction"]) * 100 for row in summary_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    x = np.arange(len(model_names))
    width = 0.36

    # Left: front20 GMACs vs latency
    bars1 = ax1.bar(x - width / 2, gmacs, width, label="20% Depth GMACs", color="#A6CDE2",
                    edgecolor="black", linewidth=0.6)
    bars2 = ax1.bar(x + width / 2, latency, width, label="20% Depth Latency", color="#2F7FB8",
                    edgecolor="black", linewidth=0.6)
    for idx, bar in enumerate(bars2):
        g = gmacs[idx]
        l = latency[idx]
        ratio = (l / g) if g > 0 else 0.0
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{ratio:.1f}x",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#1f4e79",
        )
    ax1.set_ylabel("Percentage (%)", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=20, ha="right", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelsize=11)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.set_ylim(0, max(gmacs + latency + [1.0]) * 1.15)
    ax1.legend(frameon=False, loc="upper left", fontsize=11, prop={"weight": "bold"})

    # Right: boot fraction
    x2 = np.arange(len(model_names))
    bars = ax2.bar(
        x2,
        boot_pct,
        width=0.6,
        color="#2F7FB8",
        edgecolor="black",
        linewidth=0.6,
        label="Boot Latency (%)",
    )
    for i, bar in enumerate(bars):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{boot_pct[i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#1f4e79",
        )
    ax2.set_ylabel("Boot Fraction (%)", fontsize=14, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(model_names, rotation=20, ha="right", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=11)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, max(boot_pct + [1.0]) * 1.2)
    ax2.legend(frameon=False, loc="upper left", fontsize=11, prop={"weight": "bold"})

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Combined plot saved to {save_path}")
    plt.close(fig)


def _plot_latency_comparison(summary_rows, save_path: str):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skipping latency comparison plot")
        return

    _apply_plot_style()

    models = [row["model"] for row in summary_rows]
    total_ms = [float(row["total_latency"]) for row in summary_rows]
    total_m = [v / 1e6 for v in total_ms]

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)

    bars = ax.bar(
        x,
        total_m,
        width=0.6,
        color="#2F7FB8",
        edgecolor="black",
        linewidth=0.6,
        label="Total FHE Latency (M)",
    )

    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{total_m[i]:.2f}M",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#1f4e79",
        )

    ax.set_ylabel("Latency (M)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(total_m + [1.0]) * 1.2)
    ax.legend(frameon=False, loc="upper left", fontsize=11, prop={"weight": "bold"})

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Latency comparison plot saved to {save_path}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze best_models FHE latency and compare with baselines.")
    parser.add_argument("--best-models-dir", default="best_models", help="Directory containing best_models JSON files.")
    parser.add_argument("--output-dir", default="fhe_statistics/results", help="Directory to save CSV/JSON outputs.")
    parser.add_argument("--top-ops", type=int, default=8, help="Top ops to include in the JSON summary.")
    parser.add_argument("--front-depth-ratio", type=float, default=0.2, help="Front depth ratio for metrics.")
    parser.add_argument("--ct-depths", default="auto", help="Comma-separated depths for ciphertext stats, or 'auto'.")
    args = parser.parse_args()

    best_dir = Path(args.best_models_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    detail = {}
    front_metrics = {}
    ct_rows = []
    ct_depths_raw = args.ct_depths.strip().lower()
    ct_depths = [int(x) for x in ct_depths_raw.split(",") if x.strip() and x.strip().isdigit()]
    use_auto_ct_depths = ct_depths_raw == "auto"

    # Best models
    for short_name, cfg_name, model in _load_best_models(best_dir):
        info, summary, breakdown = _analyze_model(short_name, model)
        front = info.get_front_depth_metrics(args.front_depth_ratio)
        front_metrics[short_name] = front
        ct_depths_use = _pick_first_depths(info) if use_auto_ct_depths else ct_depths
        ct_stats = info.get_depth_ct_stats(ct_depths_use)
        summary_rows.append({
            "group": "best_models",
            "model": short_name,
            "config_name": cfg_name,
            **summary,
        })
        detail[short_name] = {
            "group": "best_models",
            "config_name": cfg_name,
            "summary": summary,
            "front_depth_metrics": front,
            "top_ops": _top_ops(breakdown, args.top_ops),
            "op_breakdown": breakdown,
        }
        for depth, stats in ct_stats.items():
            ct_rows.append({
                "group": "best_models",
                "model": short_name,
                "depth": depth,
                **stats,
            })

    # Baselines
    for name, model in _baseline_models():
        info, summary, breakdown = _analyze_model(name, model)
        front = info.get_front_depth_metrics(args.front_depth_ratio)
        front_metrics[name] = front
        ct_depths_use = _pick_first_depths(info) if use_auto_ct_depths else ct_depths
        ct_stats = info.get_depth_ct_stats(ct_depths_use)
        summary_rows.append({
            "group": "baseline",
            "model": name,
            "config_name": name,
            **summary,
        })
        detail[name] = {
            "group": "baseline",
            "config_name": name,
            "summary": summary,
            "front_depth_metrics": front,
            "top_ops": _top_ops(breakdown, args.top_ops),
            "op_breakdown": breakdown,
        }
        for depth, stats in ct_stats.items():
            ct_rows.append({
                "group": "baseline",
                "model": name,
                "depth": depth,
                **stats,
            })

    # Save CSV summary
    csv_path = out_dir / "best_models_latency_summary.csv"
    fieldnames = [
        "group",
        "model",
        "config_name",
        "total_latency",
        "op_latency",
        "boot_latency",
        "boot_count",
        "boot_fraction",
        "max_depth",
        "params",
        "flops",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    # Save JSON details
    json_path = out_dir / "best_models_latency_details.json"
    json_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")

    # Save front-depth metrics CSV
    ratio_pct = int(round(args.front_depth_ratio * 100))
    front_csv_path = out_dir / f"front{ratio_pct}_depth_metrics.csv"
    front_fieldnames = [
        "model",
        "front_depth_ratio",
        "front_depth_threshold",
        "front_latency_pct",
        "front_param_pct",
        "front_flops_pct",
        "latency_flops_ratio",
    ]
    with front_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=front_fieldnames)
        writer.writeheader()
        for model_name, metrics in front_metrics.items():
            g = metrics.get("front_flops_pct", 0.0)
            l = metrics.get("front_latency_pct", 0.0)
            row = {
                "model": model_name,
                **metrics,
                "latency_flops_ratio": (l / g) if g > 0 else 0.0,
            }
            writer.writerow(row)

    # Save shallow ciphertext stats CSV
    ct_csv_path = out_dir / "shallow_ct_stats.csv"
    ct_fieldnames = [
        "group",
        "model",
        "depth",
        "nodes",
        "sum_in_ct",
        "sum_out_ct",
        "avg_in_ct",
        "avg_out_ct",
        "min_in_ct",
        "max_in_ct",
        "min_out_ct",
        "max_out_ct",
        "sum_boot_count",
    ]
    with ct_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ct_fieldnames)
        writer.writeheader()
        for row in ct_rows:
            writer.writerow(row)

    # Save front-depth comparison plot
    plot_path = out_dir / f"front{ratio_pct}_depth_comparison.pdf"
    FheInfo.plot_front_depth_comparison(front_metrics, save_path=str(plot_path), show=False)

    # Save boot fraction plot
    boot_plot_path = out_dir / "boot_fraction.pdf"
    _plot_boot_fraction(summary_rows, str(boot_plot_path))

    # Save combined plot
    combined_plot_path = out_dir / f"front{ratio_pct}_boot_combined.pdf"
    _plot_combined(front_metrics, summary_rows, str(combined_plot_path))

    # Save latency comparison plot
    latency_plot_path = out_dir / "latency_comparison.pdf"
    _plot_latency_comparison(summary_rows, str(latency_plot_path))

    # Print a compact console table
    def fmt(x: float) -> str:
        return f"{x/1e6:.3f}M" if x >= 1e6 else f"{x:.1f}"

    print("Model comparison (total latency includes boot):")
    for row in summary_rows:
        print(
            f"- {row['model']:<28} "
            f"total={fmt(row['total_latency']):>10} "
            f"boot={fmt(row['boot_latency']):>10} "
            f"boot%={row['boot_fraction']*100:>6.2f} "
            f"depth={int(row['max_depth']):>4}"
        )
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {front_csv_path}")
    print(f"Saved: {ct_csv_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {boot_plot_path}")
    print(f"Saved: {combined_plot_path}")
    print(f"Saved: {latency_plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
