from __future__ import annotations

import json
import re
import signal
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import onnx

from model_registry import ModelRequest
from onnx_export import OnnxExportConfig, export_onnx
from paths import (
    DEFAULT_CKKS_CONFIG,
    DEFAULT_HECATE_DIR,
    DEFAULT_MLIR_DIR,
    DEFAULT_ONNX_DIR,
    DEFAULT_TEMP_DIR,
    PIPELINE_ROOT,
    ensure_parent,
)
from tools import (
    compile_with_onnx_mlir,
    default_hecate_args_for_onnx,
    default_onnx_mlir_args_for_hecate,
    find_tool,
    format_hecate_pipeline_arg,
)


DEFAULT_COMPARE_MODELS = ("resnet18", "resnet34", "resnet50")
POOL_OPS = {"AveragePool", "GlobalAveragePool", "MaxPool", "MaxPoolSingleOut", "ReduceMeanV13"}
MLIR_ONNX_OP_RE = re.compile(r'"onnx\.([A-Za-z0-9]+)"')


@dataclass(frozen=True)
class CompareVariant:
    name: str
    emit: str
    onnx_mlir_args: tuple[str, ...]
    prefer_onnx_mlir: bool = False


@dataclass
class CompareRow:
    model: str
    variant: str
    export_returncode: int
    compile_returncode: int | None
    hecate_onnx_returncode: int | None
    hecate_trytry_returncode: int | None
    hecate_onnx_status: str
    hecate_trytry_status: str
    hecate_input: str | None
    onnx_pool_ops: list[str]
    mlir_pool_ops: list[str]
    mlir_pool_line: str
    export_log: str
    compile_log: str | None
    hecate_onnx_log: str | None
    hecate_trytry_log: str | None
    hecate_onnx_excerpt: str
    hecate_trytry_excerpt: str


@dataclass(frozen=True)
class LoggedResult:
    command: list[str]
    returncode: int
    log_path: Path
    elapsed_seconds: float


def default_variants() -> tuple[CompareVariant, ...]:
    return (
        CompareVariant("onnxir", "EmitONNXIR", ()),
        CompareVariant("no_transform", "EmitONNXIR", ("--onnx-op-transform-threshold=0",)),
        CompareVariant("basic", "EmitONNXBasic", ()),
        CompareVariant(
            "upstream",
            "EmitONNXIR",
            tuple(default_onnx_mlir_args_for_hecate()),
            prefer_onnx_mlir=True,
        ),
    )


def compare_pooling_modes(
    models: list[str] | None = None,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    num_classes: int = 100,
    pretrained: bool = False,
    timeout_seconds: int = 90,
    ckks_config: Path = DEFAULT_CKKS_CONFIG,
    run_trytry_control: bool = True,
) -> tuple[Path, list[CompareRow]]:
    selected_models = models or list(DEFAULT_COMPARE_MODELS)
    log_dir = DEFAULT_TEMP_DIR / "test_logs" / time.strftime("pooling_compare_%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[CompareRow] = []
    for model_name in selected_models:
        output_stem = f"{model_name}_cmp"
        export_log = log_dir / f"{output_stem}_export.log"
        export_returncode = export_model_to_log(
            model_name=model_name,
            output_stem=output_stem,
            input_shape=input_shape,
            num_classes=num_classes,
            pretrained=pretrained,
            log_path=export_log,
        )
        onnx_path = DEFAULT_ONNX_DIR / f"{output_stem}.onnx"
        onnx_pool_ops = inspect_onnx_pool_ops(onnx_path) if export_returncode == 0 else []

        for variant in default_variants():
            compile_log = log_dir / f"{output_stem}_{variant.name}_onnx_mlir.log"
            hecate_onnx_log = log_dir / f"{output_stem}_{variant.name}_hecate_onnx.log"
            hecate_trytry_log = log_dir / f"{output_stem}_{variant.name}_hecate_trytry.log"
            compile_returncode: int | None = None
            hecate_onnx_returncode: int | None = None
            hecate_trytry_returncode: int | None = None
            hecate_onnx_status = "skip"
            hecate_trytry_status = "skip"
            hecate_onnx_excerpt = ""
            hecate_trytry_excerpt = ""
            hecate_input: str | None = None
            mlir_pool_ops: list[str] = []
            mlir_pool_line = ""

            if export_returncode == 0:
                mlir_stem = f"{output_stem}_{variant.name}"
                compile_result = run_compile_to_log(onnx_path, variant, mlir_stem, compile_log)
                compile_returncode = compile_result.returncode

                mlir_path = find_mlir_output_by_stem(mlir_stem, prefer_onnx_mlir=variant.prefer_onnx_mlir)
                if compile_returncode == 0 and mlir_path is not None:
                    hecate_input = str(mlir_path)
                    mlir_pool_ops, mlir_pool_line = inspect_mlir_pool_ops(mlir_path)
                    hecate_onnx_result = run_hecate_to_log(
                        input_path=mlir_path,
                        output_path=DEFAULT_HECATE_DIR / f"{output_stem}_{variant.name}_onnx.mlir",
                        ckks_config=ckks_config,
                        pipeline="onnx",
                        extra_args=tuple(default_hecate_args_for_onnx()) if variant.name == "upstream" else (),
                        log_path=hecate_onnx_log,
                        timeout_seconds=timeout_seconds,
                    )
                    hecate_onnx_returncode = hecate_onnx_result.returncode
                    hecate_onnx_status = classify_hecate_result(hecate_onnx_result)
                    hecate_onnx_excerpt = log_excerpt(hecate_onnx_log)

                    if run_trytry_control:
                        hecate_trytry_result = run_hecate_to_log(
                            input_path=mlir_path,
                            output_path=DEFAULT_HECATE_DIR / f"{output_stem}_{variant.name}_trytry.mlir",
                            ckks_config=ckks_config,
                            pipeline="trytry",
                            extra_args=(),
                            log_path=hecate_trytry_log,
                            timeout_seconds=timeout_seconds,
                        )
                        hecate_trytry_returncode = hecate_trytry_result.returncode
                        hecate_trytry_status = classify_hecate_result(hecate_trytry_result)
                        hecate_trytry_excerpt = log_excerpt(hecate_trytry_log)
                else:
                    hecate_onnx_excerpt = log_excerpt(compile_log)

            rows.append(
                CompareRow(
                    model=model_name,
                    variant=variant.name,
                    export_returncode=export_returncode,
                    compile_returncode=compile_returncode,
                    hecate_onnx_returncode=hecate_onnx_returncode,
                    hecate_trytry_returncode=hecate_trytry_returncode,
                    hecate_onnx_status=hecate_onnx_status,
                    hecate_trytry_status=hecate_trytry_status,
                    hecate_input=hecate_input,
                    onnx_pool_ops=onnx_pool_ops,
                    mlir_pool_ops=mlir_pool_ops,
                    mlir_pool_line=mlir_pool_line,
                    export_log=str(export_log),
                    compile_log=str(compile_log) if compile_log.exists() else None,
                    hecate_onnx_log=str(hecate_onnx_log) if hecate_onnx_log.exists() else None,
                    hecate_trytry_log=str(hecate_trytry_log) if hecate_trytry_log.exists() else None,
                    hecate_onnx_excerpt=hecate_onnx_excerpt,
                    hecate_trytry_excerpt=hecate_trytry_excerpt,
                )
            )

    summary_path = log_dir / "summary.json"
    summary_path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_path, rows


def export_model_to_log(
    model_name: str,
    output_stem: str,
    input_shape: tuple[int, ...],
    num_classes: int,
    pretrained: bool,
    log_path: Path,
) -> int:
    ensure_parent(log_path)
    command_text = (
        f"export_onnx(model={model_name}, output_stem={output_stem}, "
        f"input_shape={input_shape}, num_classes={num_classes}, pretrained={pretrained})"
    )
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {command_text}\n")
        try:
            result = export_onnx(
                OnnxExportConfig(
                    model=ModelRequest(
                        name=model_name,
                        num_classes=num_classes,
                        pretrained=pretrained,
                    ),
                    input_shape=input_shape,
                    output_dir=DEFAULT_ONNX_DIR,
                    output_name_override=output_stem,
                )
            )
        except Exception as exc:
            log.write(f"error: {exc}\n")
            return 1
        log.write(f"onnx_path={result.onnx_path}\n")
        log.write(f"metadata_path={result.metadata_path}\n")
    return 0


def run_compile_to_log(
    onnx_path: Path,
    variant: CompareVariant,
    output_stem: str,
    log_path: Path,
) -> LoggedResult:
    result = compile_with_onnx_mlir(
        onnx_path=onnx_path,
        output_dir=DEFAULT_MLIR_DIR,
        emit=variant.emit,
        extra_args=list(variant.onnx_mlir_args),
        output_stem=output_stem,
    )
    return write_command_result(log_path, result.command, result.returncode, result.stdout, result.stderr, 0.0)


def find_mlir_output_by_stem(stem: str, prefer_onnx_mlir: bool = False) -> Path | None:
    if prefer_onnx_mlir:
        candidates = (
            DEFAULT_MLIR_DIR / f"{stem}.onnx.mlir",
            DEFAULT_MLIR_DIR / f"{stem}.tmp",
            DEFAULT_MLIR_DIR / f"{stem}.mlir",
        )
    else:
        candidates = (
            DEFAULT_MLIR_DIR / f"{stem}.tmp",
            DEFAULT_MLIR_DIR / f"{stem}.onnx.mlir",
            DEFAULT_MLIR_DIR / f"{stem}.mlir",
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_hecate_to_log(
    input_path: Path,
    output_path: Path,
    ckks_config: Path,
    pipeline: str,
    extra_args: tuple[str, ...],
    log_path: Path,
    timeout_seconds: int,
) -> LoggedResult:
    lookup = find_tool("hecate-opt")
    if lookup.path is None:
        raise FileNotFoundError("hecate-opt was not found under ./build or PATH")

    ensure_parent(output_path)
    command = [str(lookup.path), format_hecate_pipeline_arg(pipeline), f"--ckks-config={ckks_config}"]
    command.extend(extra_args)
    command.extend([f"--input={input_path}", f"--out={output_path}"])
    start = time.monotonic()
    ensure_parent(log_path)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=PIPELINE_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
            log.write(f"\nTIMEOUT after {timeout_seconds}s\n")
    elapsed = time.monotonic() - start
    return LoggedResult(command=command, returncode=returncode, log_path=log_path, elapsed_seconds=elapsed)


def write_command_result(
    log_path: Path,
    command: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
    elapsed_seconds: float,
) -> LoggedResult:
    ensure_parent(log_path)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(command) + "\n")
        if stdout:
            log.write(stdout)
        if stderr:
            log.write(stderr)
    return LoggedResult(command=command, returncode=returncode, log_path=log_path, elapsed_seconds=elapsed_seconds)


def inspect_onnx_pool_ops(path: Path) -> list[str]:
    model = onnx.load(str(path))
    counter = Counter(node.op_type for node in model.graph.node if node.op_type in POOL_OPS)
    return sorted(counter)


def inspect_mlir_pool_ops(path: Path) -> tuple[list[str], str]:
    if not path.exists():
        return [], ""
    text = path.read_text(encoding="utf-8", errors="replace")
    pool_ops = sorted({match.group(1) for match in MLIR_ONNX_OP_RE.finditer(text) if match.group(1) in POOL_OPS})
    pool_line = ""
    for line in text.splitlines():
        if any(pool_op in line for pool_op in POOL_OPS):
            pool_line = line.strip()
            break
    return pool_ops, pool_line


def classify_hecate_result(result: LoggedResult) -> str:
    if result.returncode == 0:
        return "ok"
    if result.returncode == 124:
        return "timeout"
    if result.returncode < 0:
        try:
            return f"signal:{signal.Signals(-result.returncode).name}"
        except ValueError:
            return f"signal:{-result.returncode}"
    excerpt = log_excerpt(result.log_path)
    if "Assertion" in excerpt or "assert" in excerpt or "stl_vector" in excerpt:
        return "assert"
    return "fail"


def log_excerpt(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    interesting = []
    for line in text.splitlines():
        if any(
            marker in line
            for marker in (
                "Assertion",
                "assert",
                "AutoLayout.cpp",
                "GetGraphLatency.cpp",
                "TIMEOUT",
                "error:",
                "stl_vector",
            )
        ):
            interesting.append(line.strip())
    if interesting:
        return " | ".join(interesting[-4:])[:1000]
    return ""
