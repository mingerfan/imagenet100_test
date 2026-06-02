#!/usr/bin/env python3
"""Create a StablePoly4 variant of the evolution rank-10 NAS architecture."""

from __future__ import annotations

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


def main() -> None:
    src = Path("configs/nas_variants/evolution_rank10.json")
    dst = Path("configs/nas_variants/evolution_rank10_poly4.json")

    data = json.loads(src.read_text(encoding="utf-8"))
    variant = copy.deepcopy(data)
    config = variant["config"]

    replacements = []
    for idx, block in enumerate(config["blocks"]):
        old_id = int(block["block_id"])
        new_id = SWISH_TO_POLY4_BLOCK_ID.get(old_id, old_id)
        if new_id != old_id:
            replacements.append({"index": idx, "from": old_id, "to": new_id})
            block["block_id"] = new_id

    config["block_choices"] = [int(block["block_id"]) for block in config["blocks"]]
    config["name"] = "net_7845934d_poly4_proxy"
    config["description"] = (
        "AutoFHE candidate derived from evolution rank-10 by mapping Swish/"
        "LearnableSwish block variants to their StablePoly4 counterparts."
    )

    variant["source_architecture"] = str(src)
    variant["variant"] = "poly4_block_activation_mapping"
    variant["replacements"] = replacements

    dst.write_text(json.dumps(variant, indent=2), encoding="utf-8")
    print(f"Wrote {dst} with {len(replacements)} block replacements")


if __name__ == "__main__":
    main()
