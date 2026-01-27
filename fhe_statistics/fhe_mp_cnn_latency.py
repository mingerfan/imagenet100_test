from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_FILENAME_RE = re.compile(
    r"^(?P<model>resnet\d+)_cifar(?P<dataset>\d+)_image(?P<image>\d+)\.txt$",
    re.IGNORECASE,
)

_LAYER_RE = re.compile(r"^\s*layer\s+(?P<layer>\d+)\s*$", re.IGNORECASE)
_OP_RE = re.compile(r"^\s*(?P<op>.+?)\.\.\.\s*$")
_TIME_RE = re.compile(r"^\s*time\s*:\s*(?P<ms>\d+(?:\.\d+)?)\s*ms\s*$", re.IGNORECASE)
_LEVEL_RE = re.compile(r"^\s*remaining\s+level\s*:\s*(?P<lvl>\d+)\s*$", re.IGNORECASE)
_SCALE_RE = re.compile(r"^\s*scale\s*:\s*(?P<scale>[-+0-9.eE]+)\s*$")
_TOTAL_RE = re.compile(r"^\s*total\s+time\s*:\s*(?P<ms>\d+(?:\.\d+)?)\s*ms\s*$", re.IGNORECASE)
_IMAGE_LABEL_RE = re.compile(r"^\s*image\s+label\s*:\s*(?P<label>\d+)\s*$", re.IGNORECASE)
_INFERRED_LABEL_RE = re.compile(r"^\s*inferred\s+label\s*:\s*(?P<label>\d+)\s*$", re.IGNORECASE)


def _op_kind(op_name: str) -> str:
    t = op_name.strip().lower()
    if "bootstrapping" in t or t.startswith("boot"):
        return "boot"
    if "convolution" in t:
        return "conv"
    if "batch normalization" in t or "batchnorm" in t:
        return "batch_norm"
    if "relu" in t or "activation" in t:
        return "activation"
    if "pool" in t:
        return "pool"
    if "fully connected" in t or "fc" == t:
        return "fc"
    if "cipher add" in t or t == "add":
        return "add"
    return "other"


@dataclass
class FheMpCnnOpEvent:
    layer: Optional[int]
    op_name: str
    op_kind: str
    time_ms: Optional[float] = None
    remaining_level: Optional[int] = None
    scale: Optional[float] = None


@dataclass
class FheMpCnnLatencyReport:
    log_path: Path
    model: Optional[str]
    dataset: Optional[str]
    image_index: Optional[int]
    total_time_ms: Optional[float]
    image_label: Optional[int]
    inferred_label: Optional[int]
    events: List[FheMpCnnOpEvent]

    def totals_by_kind_ms(self) -> Dict[str, float]:
        totals: Dict[str, float] = defaultdict(float)
        for e in self.events:
            if e.time_ms is None:
                continue
            totals[e.op_kind] += float(e.time_ms)
        return dict(totals)

    def totals_by_op_ms(self) -> Dict[str, float]:
        totals: Dict[str, float] = defaultdict(float)
        for e in self.events:
            if e.time_ms is None:
                continue
            totals[e.op_name] += float(e.time_ms)
        return dict(totals)

    def boot_time_ms(self) -> float:
        return float(self.totals_by_kind_ms().get("boot", 0.0))

    def nonboot_time_ms(self) -> float:
        totals = self.totals_by_kind_ms()
        return float(sum(v for k, v in totals.items() if k != "boot"))


def parse_fhe_mp_cnn_log(log_path: str | Path) -> FheMpCnnLatencyReport:
    path = Path(log_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    model = dataset = None
    image_index = None
    m = _FILENAME_RE.match(path.name)
    if m:
        model = m.group("model").lower()
        dataset = f"cifar{m.group('dataset')}"
        image_index = int(m.group("image"))

    current_layer: Optional[int] = None
    current_event: Optional[FheMpCnnOpEvent] = None
    events: List[FheMpCnnOpEvent] = []
    total_time_ms: Optional[float] = None
    image_label: Optional[int] = None
    inferred_label: Optional[int] = None

    def flush_event() -> None:
        nonlocal current_event
        if current_event is not None:
            events.append(current_event)
            current_event = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        m = _LAYER_RE.match(line)
        if m:
            flush_event()
            current_layer = int(m.group("layer"))
            continue

        m = _OP_RE.match(line)
        if m:
            flush_event()
            op_name = m.group("op").strip()
            current_event = FheMpCnnOpEvent(
                layer=current_layer,
                op_name=op_name,
                op_kind=_op_kind(op_name),
            )
            continue

        m = _TIME_RE.match(line)
        if m and current_event is not None:
            current_event.time_ms = float(m.group("ms"))
            continue

        m = _LEVEL_RE.match(line)
        if m and current_event is not None:
            current_event.remaining_level = int(m.group("lvl"))
            continue

        m = _SCALE_RE.match(line)
        if m and current_event is not None:
            try:
                current_event.scale = float(m.group("scale"))
            except ValueError:
                current_event.scale = None
            continue

        m = _TOTAL_RE.match(line)
        if m:
            total_time_ms = float(m.group("ms"))
            continue

        m = _IMAGE_LABEL_RE.match(line)
        if m:
            image_label = int(m.group("label"))
            continue

        m = _INFERRED_LABEL_RE.match(line)
        if m:
            inferred_label = int(m.group("label"))
            continue

    flush_event()

    return FheMpCnnLatencyReport(
        log_path=path,
        model=model,
        dataset=dataset,
        image_index=image_index,
        total_time_ms=total_time_ms,
        image_label=image_label,
        inferred_label=inferred_label,
        events=events,
    )


def iter_fhe_mp_cnn_logs(
    result_dir: str | Path,
    *,
    pattern: str = "resnet*_cifar*_image*.txt",
) -> Iterable[Path]:
    base = Path(result_dir)
    yield from sorted(base.glob(pattern))


def analyze_fhe_mp_cnn_result_dir(
    result_dir: str | Path,
    *,
    pattern: str = "resnet*_cifar*_image*.txt",
) -> List[FheMpCnnLatencyReport]:
    return [parse_fhe_mp_cnn_log(p) for p in iter_fhe_mp_cnn_logs(result_dir, pattern=pattern)]


def summarize_fhe_mp_cnn_reports(reports: Iterable[FheMpCnnLatencyReport]) -> Dict[str, float]:
    reports = list(reports)
    if not reports:
        return {
            "runs": 0.0,
            "mean_total_time_ms": 0.0,
            "mean_boot_time_ms": 0.0,
            "mean_nonboot_time_ms": 0.0,
            "boot_fraction_of_total": 0.0,
        }

    total_times = [r.total_time_ms for r in reports if r.total_time_ms is not None]
    boot_times = [r.boot_time_ms() for r in reports]
    nonboot_times = [r.nonboot_time_ms() for r in reports]

    mean_total = (sum(total_times) / len(total_times)) if total_times else 0.0
    mean_boot = sum(boot_times) / len(boot_times)
    mean_nonboot = sum(nonboot_times) / len(nonboot_times)
    boot_fraction = (mean_boot / mean_total) if mean_total > 0 else 0.0

    return {
        "runs": float(len(reports)),
        "mean_total_time_ms": float(mean_total),
        "mean_boot_time_ms": float(mean_boot),
        "mean_nonboot_time_ms": float(mean_nonboot),
        "boot_fraction_of_total": float(boot_fraction),
    }


def _format_ms(ms: Optional[float]) -> str:
    if ms is None:
        return "n/a"
    if ms >= 1000:
        return f"{ms/1000.0:.2f}s"
    return f"{ms:.2f}ms"


def print_fhe_mp_cnn_report(report: FheMpCnnLatencyReport, top_ops: int = 10) -> None:
    totals_by_kind = report.totals_by_kind_ms()
    totals_by_op = report.totals_by_op_ms()
    ordered_ops = sorted(totals_by_op.items(), key=lambda kv: kv[1], reverse=True)[: max(0, top_ops)]

    header = f"{report.model or 'unknown'} {report.dataset or ''} image={report.image_index}"
    print(header.strip())
    print(f"  total: {_format_ms(report.total_time_ms)}  boot: {_format_ms(report.boot_time_ms())}")
    print("  by_kind_ms:")
    for k, v in sorted(totals_by_kind.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    - {k}: {v:.2f}")
    if ordered_ops:
        print(f"  top_ops_ms (top {len(ordered_ops)}):")
        for op, v in ordered_ops:
            print(f"    - {op}: {v:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FHE-MP-CNN latency statistics from result log files.")
    parser.add_argument(
        "path",
        type=str,
        help="A log file path, or a result directory containing resnet*_cifar*_image*.txt",
    )
    parser.add_argument("--top-ops", type=int, default=10)
    args = parser.parse_args()

    p = Path(args.path)
    if p.is_dir():
        reports = analyze_fhe_mp_cnn_result_dir(p)
        for r in reports:
            print_fhe_mp_cnn_report(r, top_ops=args.top_ops)
        print(summarize_fhe_mp_cnn_reports(reports))
    else:
        r = parse_fhe_mp_cnn_log(p)
        print_fhe_mp_cnn_report(r, top_ops=args.top_ops)
