#!/usr/bin/env python3
"""Create partial activation-override variants of the evolution rank-10 NAS architecture."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        default="configs/nas_variants/evolution_rank10.json",
        help="Source NAS JSON variant.",
    )
    parser.add_argument(
        "--dst",
        default="configs/nas_variants/evolution_rank10_swish_herpn_idx6.json",
        help="Destination JSON variant.",
    )
    parser.add_argument(
        "--indices",
        default="6",
        help="Comma-separated body block indices to override.",
    )
    parser.add_argument(
        "--activation",
        default="swish_herpn",
        help="Activation override registered in network_gen.search_space.ACTIVATION_TYPES.",
    )
    parser.add_argument(
        "--variant-name",
        default=None,
        help="Optional suffix used in config name and metadata.",
    )
    return parser.parse_args()


def parse_indices(raw: str) -> list[int]:
    indices = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            indices.append(int(item))
    return sorted(set(indices))


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    indices = parse_indices(args.indices)
    activation = str(args.activation).strip().lower()
    index_label = "_".join(str(i) for i in indices) or "none"
    variant_name = args.variant_name or f"{activation}_idx{index_label}"

    data = json.loads(src.read_text(encoding="utf-8"))
    variant = copy.deepcopy(data)
    config = variant["config"]

    replacements = []
    for idx, block in enumerate(config["blocks"]):
        if idx not in indices:
            block.pop("activation_override", None)
            continue
        old_override = block.get("activation_override")
        block["activation_override"] = activation
        replacements.append(
            {
                "index": idx,
                "block_id": int(block["block_id"]),
                "from_activation_override": old_override,
                "to_activation_override": activation,
            }
        )

    config["block_choices"] = [int(block["block_id"]) for block in config["blocks"]]
    config["name"] = f"net_7845934d_{variant_name}_proxy"
    config["description"] = (
        "Activation-override proxy derived from evolution rank-10 by replacing "
        f"body block indices {indices} with {activation} while keeping block ids "
        "and all convolutional structure unchanged."
    )

    variant["source_architecture"] = str(src)
    variant["variant"] = f"{variant_name}_activation_override"
    variant["replacements"] = replacements

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(variant, indent=2), encoding="utf-8")
    print(f"Wrote {dst} with {len(replacements)} activation overrides")


if __name__ == "__main__":
    main()
