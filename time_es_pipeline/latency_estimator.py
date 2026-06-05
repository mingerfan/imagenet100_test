from __future__ import annotations

import json
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from onnx_export import OnnxExportConfig, OnnxExportResult, export_onnx
from paths import (
    DEFAULT_CKKS_CONFIG,
    DEFAULT_HECATE_DIR,
    DEFAULT_MLIR_DIR,
    DEFAULT_TEMP_DIR,
    PIPELINE_ROOT,
    ensure_parent,
)
from tools import (
    CommandResult,
    compile_with_onnx_mlir,
    default_hecate_args_for_onnx,
    default_onnx_mlir_args_for_hecate,
    find_onnx_mlir_output,
    find_tool,
    format_hecate_pipeline_arg,
)


LATENCY_PATTERNS = (
    re.compile(r"\[LATENCY\]\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"FINAL LATENCY IS\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"final latency is\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
)


@dataclass(frozen=True)
class LatencyEstimateResult:
    status: str
    model: str
    latency: float | None
    latency_line: str | None
    elapsed_seconds: float
    export_result: dict[str, object] | None
    onnx_path: str | None
    mlir_path: str | None
    hecate_output_path: str | None
    summary_path: str
    export_log: str
    compile_log: str
    hecate_log: str
    onnx_mlir_command: list[str] | None
    hecate_command: list[str] | None
    onnx_mlir_returncode: int | None
    hecate_returncode: int | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def estimate_latency(
    export_config: OnnxExportConfig,
    onnx_mlir_tool: str | Path | None = None,
    onnx_mlir_args: list[str] | None = None,
    hecate_tool: str | Path | None = None,
    ckks_config: Path = DEFAULT_CKKS_CONFIG,
    hecate_pipeline: str = "onnx",
    hecate_args: list[str] | None = None,
    prefer_tmp_mlir: bool = False,
    timeout_seconds: int | None = None,
    log_dir: Path | None = None,
    ld_library_path_prepend: list[str] | None = None,
) -> LatencyEstimateResult:
    started_at = time.monotonic()
    stem = export_config.output_name_override or export_config.model.name
    resolved_log_dir = log_dir or (
        DEFAULT_TEMP_DIR / "test_logs" / "latency_estimates" / time.strftime(f"{stem}_%Y%m%d_%H%M%S")
    )
    resolved_log_dir.mkdir(parents=True, exist_ok=True)

    export_log = resolved_log_dir / "export.log"
    compile_log = resolved_log_dir / "onnx_mlir.log"
    hecate_log = resolved_log_dir / "hecate.log"
    summary_path = resolved_log_dir / "summary.json"

    export_result: OnnxExportResult | None = None
    compile_result: CommandResult | None = None
    hecate_result: LoggedHecateResult | None = None
    mlir_path: Path | None = None
    hecate_output_path: Path | None = None

    try:
        with export_log.open("w", encoding="utf-8", errors="replace") as log:
            log.write(f"model={export_config.model.name}\n")
            log.write(f"input_shape={export_config.input_shape}\n")
            export_result = export_onnx(export_config)
            log.write(json.dumps(export_result_payload(export_result), indent=2, sort_keys=True))
            log.write("\n")

        resolved_onnx_mlir_args = list(
            default_onnx_mlir_args_for_hecate() if onnx_mlir_args is None else onnx_mlir_args
        )
        compile_result = compile_with_onnx_mlir(
            onnx_path=export_result.onnx_path,
            output_dir=DEFAULT_MLIR_DIR,
            emit="EmitONNXIR",
            tool_path=onnx_mlir_tool,
            extra_args=resolved_onnx_mlir_args,
            output_stem=stem,
        )
        write_command_log(compile_log, compile_result.command, compile_result.returncode, compile_result.stdout, compile_result.stderr)
        if not compile_result.ok:
            return write_latency_summary(
                LatencyEstimateResult(
                    status="onnx_mlir_failed",
                    model=export_config.model.name,
                    latency=None,
                    latency_line=None,
                    elapsed_seconds=time.monotonic() - started_at,
                    export_result=export_result_payload(export_result),
                    onnx_path=str(export_result.onnx_path),
                    mlir_path=None,
                    hecate_output_path=None,
                    summary_path=str(summary_path),
                    export_log=str(export_log),
                    compile_log=str(compile_log),
                    hecate_log=str(hecate_log),
                    onnx_mlir_command=compile_result.command,
                    hecate_command=None,
                    onnx_mlir_returncode=compile_result.returncode,
                    hecate_returncode=None,
                    error="onnx-mlir returned a non-zero exit code",
                ),
                summary_path,
            )

        mlir_path = find_onnx_mlir_output(
            export_result.onnx_path,
            DEFAULT_MLIR_DIR,
            prefer_tmp=prefer_tmp_mlir,
            prefer_onnx_mlir=hecate_pipeline == "onnx" and not prefer_tmp_mlir,
        )
        if mlir_path is None:
            raise FileNotFoundError(f"expected MLIR output was not found for {export_result.onnx_path}")

        hecate_output_path = DEFAULT_HECATE_DIR / f"{stem}_hecate.mlir"
        resolved_hecate_args = list(default_hecate_args_for_onnx() if hecate_args is None else hecate_args)
        hecate_result = run_hecate_to_log(
            input_path=mlir_path,
            output_path=hecate_output_path,
            ckks_config=ckks_config,
            pipeline=hecate_pipeline,
            tool_path=hecate_tool,
            extra_args=resolved_hecate_args,
            log_path=hecate_log,
            timeout_seconds=timeout_seconds,
            ld_library_path_prepend=ld_library_path_prepend or [],
        )
        latency, latency_line = parse_latency_from_log(hecate_log)
        status = "ok" if hecate_result.returncode == 0 and latency is not None else classify_returncode(hecate_result.returncode)
        if hecate_result.returncode == 0 and latency is None:
            status = "latency_not_found"

        return write_latency_summary(
            LatencyEstimateResult(
                status=status,
                model=export_config.model.name,
                latency=latency,
                latency_line=latency_line,
                elapsed_seconds=time.monotonic() - started_at,
                export_result=export_result_payload(export_result),
                onnx_path=str(export_result.onnx_path),
                mlir_path=str(mlir_path),
                hecate_output_path=str(hecate_output_path),
                summary_path=str(summary_path),
                export_log=str(export_log),
                compile_log=str(compile_log),
                hecate_log=str(hecate_log),
                onnx_mlir_command=compile_result.command,
                hecate_command=hecate_result.command,
                onnx_mlir_returncode=compile_result.returncode,
                hecate_returncode=hecate_result.returncode,
                error=None if status == "ok" else "hecate-opt did not produce a parsed latency",
            ),
            summary_path,
        )
    except Exception as exc:
        return write_latency_summary(
            LatencyEstimateResult(
                status="error",
                model=export_config.model.name,
                latency=None,
                latency_line=None,
                elapsed_seconds=time.monotonic() - started_at,
                export_result=export_result_payload(export_result) if export_result else None,
                onnx_path=str(export_result.onnx_path) if export_result else None,
                mlir_path=str(mlir_path) if mlir_path else None,
                hecate_output_path=str(hecate_output_path) if hecate_output_path else None,
                summary_path=str(summary_path),
                export_log=str(export_log),
                compile_log=str(compile_log),
                hecate_log=str(hecate_log),
                onnx_mlir_command=compile_result.command if compile_result else None,
                hecate_command=hecate_result.command if hecate_result else None,
                onnx_mlir_returncode=compile_result.returncode if compile_result else None,
                hecate_returncode=hecate_result.returncode if hecate_result else None,
                error=str(exc),
            ),
            summary_path,
        )


@dataclass(frozen=True)
class LoggedHecateResult:
    command: list[str]
    returncode: int
    log_path: Path


def run_hecate_to_log(
    input_path: Path,
    output_path: Path,
    ckks_config: Path,
    pipeline: str,
    tool_path: str | Path | None,
    extra_args: list[str],
    log_path: Path,
    timeout_seconds: int | None,
    ld_library_path_prepend: list[str],
) -> LoggedHecateResult:
    lookup = find_tool("hecate-opt", explicit=tool_path)
    if lookup.path is None:
        raise FileNotFoundError("hecate-opt was not found under ./build or PATH")

    ensure_parent(output_path)
    command = [
        str(lookup.path),
        format_hecate_pipeline_arg(pipeline),
        f"--ckks-config={ckks_config}",
        *extra_args,
        f"--input={input_path}",
        f"--out={output_path}",
    ]
    env = None
    if ld_library_path_prepend:
        import os

        env = os.environ.copy()
        previous = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join([*ld_library_path_prepend, previous] if previous else ld_library_path_prepend)

    ensure_parent(log_path)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=PIPELINE_ROOT,
                env=env,
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
    return LoggedHecateResult(command=command, returncode=returncode, log_path=log_path)


def parse_latency_from_log(log_path: Path) -> tuple[float | None, str | None]:
    if not log_path.exists():
        return None, None
    for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        for pattern in LATENCY_PATTERNS:
            match = pattern.search(line)
            if match:
                value = float(match.group(1))
                return value, line.strip()
    return None, None


def classify_returncode(returncode: int) -> str:
    if returncode == 0:
        return "ok"
    if returncode == 124:
        return "timeout"
    if returncode < 0:
        try:
            return f"signal:{signal.Signals(-returncode).name}"
        except ValueError:
            return f"signal:{-returncode}"
    return "hecate_failed"


def write_command_log(log_path: Path, command: list[str], returncode: int, stdout: str, stderr: str) -> None:
    ensure_parent(log_path)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.write(f"returncode={returncode}\n")
        if stdout:
            log.write(stdout)
        if stderr:
            log.write(stderr)


def write_latency_summary(result: LatencyEstimateResult, summary_path: Path) -> LatencyEstimateResult:
    ensure_parent(summary_path)
    summary_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def export_result_payload(result: OnnxExportResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "model_name": result.model_name,
        "onnx_path": str(result.onnx_path),
        "metadata_path": str(result.metadata_path),
        "input_shape": list(result.input_shape),
        "output_shape": list(result.output_shape),
        "opset": result.opset,
    }
