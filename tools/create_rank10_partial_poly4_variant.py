#!/usr/bin/env python3
"""Create partial StablePoly4 variants of the evolution rank-10 NAS architecture."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SWISH_TO_POLY4_BLOCK_ID = {
    1: 0,
    3: 2,
    5: 4,
    7: 6,
    9: 8,
    11: 10,
    13: 12,
    15: 14,
    17: 16,
    19: 18,
    21: 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        default="configs/nas_variants/evolution_rank10.json",
        help="Source NAS JSON variant.",
    )
    parser.add_argument(
        "--dst",
        default="configs/nas_variants/evolution_rank10_poly4_head5.json",
        help="Destination JSON variant.",
    )
    parser.add_argument(
        "--head-blocks",
        type=int,
        default=5,
        help="Number of leading body blocks to map from Swish/LearnableSwish to StablePoly4.",
    )
    parser.add_argument(
        "--indices",
        default=None,
        help=(
            "Comma-separated body block indices to map to StablePoly4. "
            "When provided, this takes precedence over --head-blocks."
        ),
    )
    return parser.parse_args()


def parse_indices(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    indices = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        indices.add(int(item))
    return indices


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    head_blocks = max(0, int(args.head_blocks))
    selected_indices = parse_indices(args.indices)

    data = json.loads(src.read_text(encoding="utf-8"))
    variant = copy.deepcopy(data)
    config = variant["config"]

    replacements = []
    for idx, block in enumerate(config["blocks"]):
        old_id = int(block["block_id"])
        should_replace = idx in selected_indices if selected_indices is not None else idx < head_blocks
        new_id = SWISH_TO_POLY4_BLOCK_ID.get(old_id, old_id) if should_replace else old_id
        if new_id != old_id:
            replacements.append({"index": idx, "from": old_id, "to": new_id})
            block["block_id"] = new_id

    config["block_choices"] = [int(block["block_id"]) for block in config["blocks"]]
    if selected_indices is None:
        variant_name = f"poly4_head{head_blocks}"
        description = (
            "AutoFHE candidate derived from evolution rank-10 by mapping the first "
            f"{head_blocks} Swish/LearnableSwish body blocks to StablePoly4 while "
            "keeping the remaining tail blocks unchanged."
        )
    else:
        index_label = "_".join(str(i) for i in sorted(selected_indices)) or "none"
        variant_name = f"poly4_idx{index_label}"
        description = (
            "AutoFHE candidate derived from evolution rank-10 by mapping body block "
            f"indices {sorted(selected_indices)} from Swish/LearnableSwish to "
            "StablePoly4 while keeping all other body blocks unchanged."
        )
    config["name"] = f"net_7845934d_{variant_name}_proxy"
    config["description"] = description

    variant["source_architecture"] = str(src)
    variant["variant"] = f"{variant_name}_block_activation_mapping"
    variant["replacements"] = replacements

    dst.write_text(json.dumps(variant, indent=2), encoding="utf-8")
    print(f"Wrote {dst} with {len(replacements)} block replacements")


if __name__ == "__main__":
    main()
