"""
GPU selection helpers.
"""

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union


GpuSpec = Union[int, str]


@dataclass(frozen=True)
class GPUSelection:
    requested: List[GpuSpec]
    selected: List[int]
    skipped: List[int]
    visible_to_physical: Dict[int, int]
    cuda_visible_devices: str


def _cuda_device_count() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 0


def _visible_to_physical_map() -> Dict[int, int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = _cuda_device_count()
    if not visible:
        return {idx: idx for idx in range(count)}

    mapping: Dict[int, int] = {}
    for visible_idx, raw in enumerate(part.strip() for part in visible.split(",")):
        if not raw:
            continue
        try:
            physical_idx = int(raw)
        except ValueError:
            physical_idx = visible_idx
        mapping[visible_idx] = physical_idx

    if count > 0:
        mapping = {idx: mapping.get(idx, idx) for idx in range(count)}
    return mapping


def _expand_gpu_token(token: GpuSpec, all_visible: Sequence[int]) -> List[int]:
    if isinstance(token, int):
        return [token]

    text = str(token).strip().lower()
    if text in {"all", "*"}:
        return list(all_visible)
    if "," in text:
        expanded: List[int] = []
        for part in text.split(","):
            expanded.extend(_expand_gpu_token(part, all_visible))
        return expanded
    if "-" in text:
        start_text, end_text = text.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        step = 1 if end >= start else -1
        return list(range(start, end + step, step))
    return [int(text)]


def _dedupe_preserve_order(values: Iterable[int]) -> List[int]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def resolve_gpu_selection(
    requested: Optional[Sequence[GpuSpec]],
    *,
    allow_gpu0: bool = False,
    excluded_physical_gpus: Optional[Sequence[int]] = None,
) -> GPUSelection:
    visible_to_physical = _visible_to_physical_map()
    all_visible = sorted(visible_to_physical)

    requested_list = list(requested) if requested else ["all"]
    selected_raw: List[int] = []
    for token in requested_list:
        selected_raw.extend(_expand_gpu_token(token, all_visible))
    selected_raw = _dedupe_preserve_order(selected_raw)

    invalid = [gpu for gpu in selected_raw if gpu not in visible_to_physical]
    if invalid:
        raise ValueError(
            f"GPU ids not visible: {invalid}; visible ids are {all_visible or 'none'}"
        )

    excluded = set(excluded_physical_gpus or [])
    if not allow_gpu0:
        excluded.add(0)

    selected: List[int] = []
    skipped: List[int] = []
    for gpu in selected_raw:
        physical = visible_to_physical.get(gpu, gpu)
        if physical in excluded:
            skipped.append(gpu)
        else:
            selected.append(gpu)

    return GPUSelection(
        requested=requested_list,
        selected=selected,
        skipped=skipped,
        visible_to_physical=visible_to_physical,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "<未设置>"),
    )


def format_gpu_ids_with_physical(
    gpu_ids: Sequence[int],
    visible_to_physical: Optional[Dict[int, int]] = None,
) -> str:
    visible_to_physical = visible_to_physical or {}
    parts = []
    for gpu in gpu_ids:
        physical = visible_to_physical.get(gpu)
        if physical is None or physical == gpu:
            parts.append(str(gpu))
        else:
            parts.append(f"{gpu}(physical {physical})")
    return "[" + ", ".join(parts) + "]"


def format_visible_gpu_mapping(visible_to_physical: Dict[int, int]) -> str:
    if not visible_to_physical:
        return "{}"
    parts = [
        f"{visible}->{physical}"
        for visible, physical in sorted(visible_to_physical.items())
    ]
    return "{" + ", ".join(parts) + "}"
