from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from paths import (
    DEFAULT_BUILD_DIR,
    DEFAULT_CKKS_CONFIG,
    DEFAULT_HECATE_DIR,
    DEFAULT_MLIR_DIR,
    ensure_parent,
)


@dataclass(frozen=True)
class ToolLookup:
    name: str
    path: Path | None
    source: str | None

    @property
    def available(self) -> bool:
        return self.path is not None


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def command_string(self) -> str:
        return " ".join(shlex.quote(part) for part in self.command)


def find_tool(name: str, explicit: str | Path | None = None, build_dir: Path = DEFAULT_BUILD_DIR) -> ToolLookup:
    if explicit:
        explicit_path = Path(explicit).expanduser().resolve()
        if explicit_path.is_file():
            return ToolLookup(name=name, path=explicit_path, source="explicit")
        if explicit_path.is_dir():
            nested = _find_in_directory(name, explicit_path)
            if nested is not None:
                return ToolLookup(name=name, path=nested, source="explicit-dir")
        return ToolLookup(name=name, path=None, source=None)

    direct = build_dir / name
    if direct.is_file():
        return ToolLookup(name=name, path=direct, source="build")

    nested = _find_in_directory(name, build_dir)
    if nested is not None:
        return ToolLookup(name=name, path=nested, source="build-recursive")

    from_path = shutil.which(name)
    if from_path:
        return ToolLookup(name=name, path=Path(from_path), source="PATH")

    return ToolLookup(name=name, path=None, source=None)


def _find_in_directory(name: str, directory: Path) -> Path | None:
    if not directory.exists():
        return None
    matches = [path for path in directory.rglob(name) if path.is_file()]
    if not matches:
        return None
    executable_matches = [path for path in matches if path.stat().st_mode & 0o111]
    return sorted(executable_matches or matches)[0]


def default_onnx_mlir_args_for_hecate() -> list[str]:
    return ["--enable-conv-opt-pass=false"]


def default_hecate_args_for_onnx() -> list[str]:
    return ["--waterline=46", "--allow-unregistered-dialect"]


def compile_with_onnx_mlir(
    onnx_path: Path,
    output_dir: Path = DEFAULT_MLIR_DIR,
    emit: str = "EmitMLIR",
    tool_path: str | Path | None = None,
    extra_args: list[str] | None = None,
    output_stem: str | None = None,
) -> CommandResult:
    lookup = find_tool("onnx-mlir", explicit=tool_path)
    if lookup.path is None:
        raise FileNotFoundError("onnx-mlir was not found under ./build or PATH")

    output_base = output_dir / (output_stem or onnx_path.stem)
    ensure_parent(output_base)
    command = [
        str(lookup.path),
        f"--{emit}",
        *(extra_args or []),
        str(onnx_path),
        "-o",
        str(output_base),
    ]
    return run_command(command)


def find_onnx_mlir_output(
    onnx_path: Path,
    output_dir: Path = DEFAULT_MLIR_DIR,
    prefer_tmp: bool = False,
    prefer_onnx_mlir: bool = False,
) -> Path | None:
    candidates = []
    if prefer_onnx_mlir:
        candidates.append(output_dir / f"{onnx_path.stem}.onnx.mlir")
    elif prefer_tmp:
        candidates.append(output_dir / f"{onnx_path.stem}.tmp")
    candidates.extend(
        [
            output_dir / f"{onnx_path.stem}.onnx.mlir",
            output_dir / f"{onnx_path.stem}.mlir",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    patterns = [f"{onnx_path.stem}*.mlir"]
    if prefer_onnx_mlir:
        patterns.insert(0, f"{onnx_path.stem}*.onnx.mlir")
    elif prefer_tmp:
        patterns.insert(0, f"{onnx_path.stem}*.tmp")

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(output_dir.glob(pattern)))
    return matches[0] if matches else None


def optimize_with_hecate(
    input_path: Path,
    output_path: Path | None = None,
    ckks_config: str | Path | None = DEFAULT_CKKS_CONFIG,
    pipeline: str | None = "trytry",
    tool_path: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> CommandResult:
    lookup = find_tool("hecate-opt", explicit=tool_path)
    if lookup.path is None:
        raise FileNotFoundError("hecate-opt was not found under ./build or PATH")

    resolved_output = output_path or default_hecate_output_path(input_path)
    ensure_parent(resolved_output)
    command = [str(lookup.path)]
    if pipeline:
        command.append(format_hecate_pipeline_arg(pipeline))
    if ckks_config is not None:
        command.append(f"--ckks-config={Path(ckks_config)}")
    command.extend(extra_args or [])
    command.extend([f"--input={input_path}", f"--out={resolved_output}"])
    return run_command(command)


def format_hecate_pipeline_arg(pipeline: str) -> str:
    if pipeline.startswith("--"):
        return pipeline
    if "(" in pipeline or ")" in pipeline:
        return f"--pass-pipeline={pipeline}"
    return f"--{pipeline}"


def default_hecate_output_path(input_path: Path) -> Path:
    if input_path.suffix == ".tmp":
        return DEFAULT_HECATE_DIR / f"{input_path.stem}.mlir"
    return DEFAULT_HECATE_DIR / input_path.name


def run_command(command: list[str]) -> CommandResult:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return CommandResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
