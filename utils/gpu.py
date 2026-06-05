"""
GPU selection helpers shared by training entry points.
"""

from dataclasses import dataclass
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Union


GPUInput = Union[int, str]
RANGE_RE = re.compile(r"^(\d+)-(\d+)$")


@dataclass(frozen=True)
class ResolvedGPUSelection:
    requested: str
    requested_ids: List[int]
    selected: List[int]
    skipped: List[int]
    device_count: int
    cuda_visible_devices: str
    visible_to_physical: Dict[int, Optional[int]]


def get_cuda_device_count() -> int:
    """Return the number of CUDA devices visible to PyTorch."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def get_visible_to_physical_map(device_count: Optional[int] = None) -> Dict[int, Optional[int]]:
    """
    Map PyTorch-visible GPU ids to physical ids when CUDA_VISIBLE_DEVICES is numeric.

    If CUDA_VISIBLE_DEVICES is unset, logical ids are physical ids. If it uses UUIDs,
    MIG ids, or another non-numeric form, physical ids cannot be inferred and map to
    None.
    """
    if device_count is None:
        device_count = get_cuda_device_count()

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_visible:
        return {idx: idx for idx in range(device_count)}

    raw_tokens = [token.strip() for token in cuda_visible.split(",") if token.strip()]
    mapping: Dict[int, Optional[int]] = {}
    for logical_id in range(device_count):
        if logical_id >= len(raw_tokens):
            mapping[logical_id] = None
            continue
        token = raw_tokens[logical_id]
        mapping[logical_id] = int(token) if token.isdigit() else None
    return mapping


def _flatten_gpu_tokens(gpu_spec: Optional[Union[GPUInput, Sequence[GPUInput]]]) -> List[str]:
    if gpu_spec is None:
        return []
    if isinstance(gpu_spec, (int, str)):
        raw_items: Iterable[GPUInput] = [gpu_spec]
    else:
        raw_items = gpu_spec

    tokens: List[str] = []
    for item in raw_items:
        for token in str(item).replace(",", " ").split():
            token = token.strip()
            if token:
                tokens.append(token)
    return tokens


def _dedupe_preserve_order(values: Iterable[int]) -> List[int]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _parse_gpu_tokens(tokens: Sequence[str], device_count: int) -> List[int]:
    if not tokens:
        return list(range(device_count))

    ids: List[int] = []
    for token in tokens:
        normalized = token.lower()
        if normalized in {"auto", "all"}:
            ids.extend(range(device_count))
            continue

        range_match = RANGE_RE.match(normalized)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if end < start:
                raise ValueError(f"Invalid GPU range '{token}': end must be >= start")
            ids.extend(range(start, end + 1))
            continue

        try:
            gpu_id = int(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Invalid GPU token '{token}'. Use ids like '1 2 3', comma lists "
                f"like '1,2,3', ranges like '0-7', or 'all'."
            ) from exc
        if gpu_id < 0:
            raise ValueError(f"Invalid GPU id {gpu_id}: ids must be non-negative")
        ids.append(gpu_id)

    return _dedupe_preserve_order(ids)


def parse_gpu_id_list(
    gpu_spec: Optional[Union[GPUInput, Sequence[GPUInput]]],
    *,
    device_count: int = 0,
) -> List[int]:
    """Parse an explicit GPU id/range/list spec without applying exclusions."""
    tokens = _flatten_gpu_tokens(gpu_spec)
    if not tokens:
        return []
    return _parse_gpu_tokens(tokens, device_count)


def _validate_visible_ids(gpu_ids: Sequence[int], device_count: int) -> None:
    if device_count <= 0:
        return
    invalid = [gpu_id for gpu_id in gpu_ids if gpu_id >= device_count]
    if invalid:
        visible = f"0-{device_count - 1}" if device_count > 1 else "0"
        raise ValueError(
            f"GPU id(s) {invalid} are not visible to PyTorch. Visible GPU ids: {visible}."
        )


def _should_exclude_logical_gpu(
    logical_id: int,
    mapping: Dict[int, Optional[int]],
    excluded_physical_gpus: Sequence[int],
) -> bool:
    physical_id = mapping.get(logical_id)
    if physical_id is None:
        # Preserve the old safety behavior when physical ids cannot be inferred.
        return logical_id in excluded_physical_gpus
    return physical_id in excluded_physical_gpus


def resolve_gpu_selection(
    gpu_spec: Optional[Union[GPUInput, Sequence[GPUInput]]] = None,
    *,
    allow_gpu0: bool = True,
    excluded_physical_gpus: Sequence[int] = (),
    device_count: Optional[int] = None,
) -> ResolvedGPUSelection:
    """
    Resolve a user GPU specification to PyTorch-visible device ids.

    Supported specs:
    - None, "auto", or "all": all visible CUDA devices
    - "1 2 3" / ["1", "2", "3"]: explicit ids
    - "1,2,3": comma-separated ids
    - "0-7": inclusive ranges

    Exclusions are physical GPU ids. The allow_gpu0 flag is retained for older
    callers; setting allow_gpu0=False adds physical GPU 0 to the exclusion list.
    """
    if device_count is None:
        device_count = get_cuda_device_count()

    tokens = _flatten_gpu_tokens(gpu_spec)
    requested = "auto" if not tokens else " ".join(tokens)
    requested_ids = _parse_gpu_tokens(tokens, device_count)
    _validate_visible_ids(requested_ids, device_count)

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    mapping = get_visible_to_physical_map(device_count)
    if device_count <= 0:
        return ResolvedGPUSelection(
            requested=requested,
            requested_ids=requested_ids,
            selected=[],
            skipped=[],
            device_count=0,
            cuda_visible_devices=cuda_visible,
            visible_to_physical=mapping,
        )

    exclusions = list(excluded_physical_gpus or [])
    if not allow_gpu0 and 0 not in exclusions:
        exclusions.append(0)

    selected = list(requested_ids)
    skipped: List[int] = []
    if exclusions:
        selected = []
        for gpu_id in requested_ids:
            if _should_exclude_logical_gpu(gpu_id, mapping, exclusions):
                skipped.append(gpu_id)
            else:
                selected.append(gpu_id)

    return ResolvedGPUSelection(
        requested=requested,
        requested_ids=list(requested_ids),
        selected=selected,
        skipped=skipped,
        device_count=device_count,
        cuda_visible_devices=cuda_visible,
        visible_to_physical=mapping,
    )


def format_gpu_ids_with_physical(
    gpu_ids: Sequence[int],
    mapping: Dict[int, Optional[int]],
) -> str:
    if not gpu_ids:
        return "[]"

    parts = []
    for gpu_id in gpu_ids:
        physical_id = mapping.get(gpu_id)
        if physical_id is None or physical_id == gpu_id:
            parts.append(str(gpu_id))
        else:
            parts.append(f"{gpu_id}(physical {physical_id})")
    return "[" + ", ".join(parts) + "]"


def format_visible_gpu_mapping(mapping: Dict[int, Optional[int]]) -> str:
    if not mapping:
        return "[]"
    parts = []
    for logical_id, physical_id in sorted(mapping.items()):
        if physical_id is None:
            parts.append(f"{logical_id}->unknown")
        else:
            parts.append(f"{logical_id}->physical {physical_id}")
    return "[" + ", ".join(parts) + "]"
