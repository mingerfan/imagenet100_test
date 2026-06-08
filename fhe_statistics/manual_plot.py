"""Manual plotting template for normalized GMACs and FHE latency.

Edit the MODELS list below, then run:

    python fhe_statistics/manual_plot.py

The default output uses a timestamped filename, for example:

    fhe_statistics/results/manual_gmacs_latency_20260602_203015.png
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable


# Data copied from Table 3.
#
# These are raw values:
#   - "gmacs" is GMACs.
#   - "latency" is FHE Latency (M).
#   - "front_gmacs_pct" is cumulative GMACs in the first 20% of depth.
#   - "front_latency_pct" is cumulative FHE latency in the first 20% of depth.
# By default, the script normalizes both metrics by the first row, ResNet-18.
MODELS = [
    {
        "name": "ResNet-18",
        "gmacs": 1.82,
        "latency": 22.3,
        "front_gmacs_pct": 12.80,
        "front_latency_pct": 33.60,
    },
    {
        "name": "ResNet-34",
        "gmacs": 3.68,
        "latency": 35.6,
        "front_gmacs_pct": 18.90,
        "front_latency_pct": 37.60,
    },
    {
        "name": "ResNet-50",
        "gmacs": 4.13,
        "latency": 70.8,
        "front_gmacs_pct": 17.80,
        "front_latency_pct": 30.30,
    },
    {
        "name": "VGG-16",
        "gmacs": 15.47,
        "latency": 128.3,
        "front_gmacs_pct": 18.50,
        "front_latency_pct": 41.30,
    },
    {
        "name": "MobileNet-V2",
        "gmacs": 0.33,
        "latency": 36.1,
        "front_gmacs_pct": 26.50,
        "front_latency_pct": 51.60,
    },
    {
        "name": "EfficientNet-B0",
        "gmacs": 0.41,
        "latency": 22.4,
        "front_gmacs_pct": 20.90,
        "front_latency_pct": 32.60,
    },
]


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_PREFIX = "manual_gmacs_latency"


def _apply_plot_style() -> None:
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "ieee"])
    except Exception:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.serif": ["Times New Roman"],
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def _normalized_rows(
    rows: Iterable[dict[str, float | str]],
    normalize_to_first: bool,
) -> list[dict[str, float | str]]:
    rows = list(rows)
    if not rows:
        raise ValueError("MODELS is empty. Add at least one model row.")

    normalized = [
        {
            "name": str(row["name"]),
            "gmacs": float(row["gmacs"]),
            "latency": float(row["latency"]),
            "front_gmacs_pct": float(row.get("front_gmacs_pct", 0.0)),
            "front_latency_pct": float(row.get("front_latency_pct", 0.0)),
        }
        for row in rows
    ]

    if not normalize_to_first:
        return normalized

    base_gmacs = normalized[0]["gmacs"]
    base_latency = normalized[0]["latency"]
    if base_gmacs <= 0 or base_latency <= 0:
        raise ValueError("The first row must have positive gmacs and latency.")

    for row in normalized:
        row["gmacs"] = row["gmacs"] / base_gmacs
        row["latency"] = row["latency"] / base_latency

    return normalized


def _format_multiplier(value: float) -> str:
    return f"{value:.2f}x" if value < 10 else f"{value:.1f}x"


def _default_output_path(timestamp_style: str) -> Path:
    if timestamp_style == "unix":
        timestamp = str(int(time.time()))
    elif timestamp_style == "none":
        timestamp = ""
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{DEFAULT_OUTPUT_PREFIX}.png"
    if timestamp:
        filename = f"{DEFAULT_OUTPUT_PREFIX}_{timestamp}.png"

    return DEFAULT_OUTPUT_DIR / filename


def plot_gmacs_latency(
    rows: Iterable[dict[str, float | str]],
    output_path: str | Path,
    *,
    normalize_to_first: bool = True,
    show_depthwise: bool = True,
    show: bool = False,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = output_path.parent / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np

    _apply_plot_style()

    rows = _normalized_rows(rows, normalize_to_first=normalize_to_first)
    names = [str(row["name"]) for row in rows]
    gmacs = [float(row["gmacs"]) for row in rows]
    latency = [float(row["latency"]) for row in rows]
    front_gmacs_pct = [float(row["front_gmacs_pct"]) for row in rows]
    front_latency_pct = [float(row["front_latency_pct"]) for row in rows]

    if any(value < 0 for value in gmacs + latency):
        raise ValueError("gmacs and latency values must be non-negative.")
    if any(value < 0 or value > 100 for value in front_gmacs_pct + front_latency_pct):
        raise ValueError("front_gmacs_pct and front_latency_pct must be in [0, 100].")

    group_spacing = 0.58
    x = np.arange(len(names)) * group_spacing
    width = 0.20

    fig_width = max(5.8, 0.95 * len(names) + 1.3)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0), dpi=300)

    bars_gmacs = ax.bar(
        x - width / 2,
        gmacs,
        width,
        label="GMACs (Normalized)",
        color="#A7CFE3",
        edgecolor="black",
        linewidth=0.7,
    )
    bars_latency = ax.bar(
        x + width / 2,
        latency,
        width,
        label="FHE Latency (Normalized)",
        color="#E41A1C",
        edgecolor="black",
        linewidth=0.7,
    )

    max_value = max(gmacs + latency + [1.0])

    if show_depthwise:
        front_gmacs = [
            value * pct / 100.0 for value, pct in zip(gmacs, front_gmacs_pct)
        ]
        front_latency = [
            value * pct / 100.0 for value, pct in zip(latency, front_latency_pct)
        ]
        ax.bar(
            x - width / 2,
            front_gmacs,
            width,
            color="none",
            edgecolor="black",
            linewidth=0.5,
            hatch="////",
            zorder=4,
        )
        ax.bar(
            x + width / 2,
            front_latency,
            width,
            color="none",
            edgecolor="black",
            linewidth=0.5,
            hatch="////",
            zorder=4,
        )
        depth_label_offset = max(max_value * 0.018, 0.08)
        depth_label_box = {
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.5,
            "alpha": 0.92,
        }

        for idx, (xpos, height, pct) in enumerate(zip(x - width / 2, front_gmacs, front_gmacs_pct)):
            ax.text(
                xpos,
                height + depth_label_offset,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="black",
                bbox=depth_label_box,
                zorder=6,
            )
        for idx, (xpos, height, pct) in enumerate(zip(x + width / 2, front_latency, front_latency_pct)):
            ax.text(
                xpos,
                height + depth_label_offset,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="black",
                bbox=depth_label_box,
                zorder=6,
            )

    value_offset = max(max_value * 0.025, 0.03)
    for bars, values in (
        (bars_gmacs, gmacs),
        (bars_latency, latency),
    ):
        for bar, value in zip(bars, values):
            extra_offset = max_value * 0.045 if value < 0.65 else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + value_offset + extra_offset,
                _format_multiplier(value),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="black",
            )

    ax.set_ylabel("Normalized Value", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    if show_depthwise:
        from matplotlib.patches import Patch

        handles.append(
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="////",
                label="First 20% Depth Share",
            )
        )
    ax.legend(handles=handles, frameon=False, loc="upper left", prop={"weight": "bold"})

    y_limit = max(max_value * 1.22, 1.35)
    ax.set_ylim(0, y_limit)
    ax.margins(x=0.13)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot normalized GMACs and FHE latency comparison from manual data."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the saved plot. If omitted, a timestamped filename is used.",
    )
    parser.add_argument(
        "--timestamp-style",
        choices=["datetime", "unix", "none"],
        default="datetime",
        help="Timestamp style for the default output filename.",
    )
    parser.add_argument(
        "--normalize-to-first",
        dest="normalize_to_first",
        action="store_true",
        default=True,
        help="Normalize both metrics by the first row. This is the default.",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize_to_first",
        action="store_false",
        help="Plot MODELS values exactly as written.",
    )
    parser.add_argument(
        "--no-depthwise",
        dest="show_depthwise",
        action="store_false",
        default=True,
        help="Hide the hatched first-20%%-depth contribution inside each bar.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving.",
    )
    args = parser.parse_args()

    output = args.output or _default_output_path(args.timestamp_style)
    output_path = plot_gmacs_latency(
        MODELS,
        output,
        normalize_to_first=args.normalize_to_first,
        show_depthwise=args.show_depthwise,
        show=args.show,
    )
    print(f"Saved plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
