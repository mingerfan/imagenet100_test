from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import onnx
import torch

from model_registry import ModelRequest, create_model
from paths import DEFAULT_ONNX_DIR


@dataclass(frozen=True)
class OnnxExportConfig:
    model: ModelRequest = field(default_factory=ModelRequest)
    input_shape: tuple[int, ...] = (1, 3, 224, 224)
    input_name: str = "input"
    output_name: str = "output"
    opset: int = 18
    output_dir: Path = DEFAULT_ONNX_DIR
    output_name_override: str | None = None
    dynamic_batch: bool = False
    verify: bool = True

    @property
    def onnx_path(self) -> Path:
        file_stem = self.output_name_override or self.model.name
        return self.output_dir / f"{file_stem}.onnx"

    @property
    def metadata_path(self) -> Path:
        return self.onnx_path.with_suffix(".json")


@dataclass(frozen=True)
class OnnxExportResult:
    onnx_path: Path
    metadata_path: Path
    model_name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    opset: int


def parse_shape(raw_shape: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw_shape.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid shape {raw_shape!r}; expected comma-separated ints") from exc
    if not values:
        raise ValueError("Input shape cannot be empty")
    if any(value <= 0 for value in values):
        raise ValueError(f"Input shape dimensions must be positive: {raw_shape!r}")
    return values


def export_onnx(config: OnnxExportConfig) -> OnnxExportResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(config.model)
    model.eval()
    model.cpu()

    sample_input = torch.randn(config.input_shape, dtype=torch.float32)
    with torch.no_grad():
        sample_output = model(sample_input)

    if not isinstance(sample_output, torch.Tensor):
        raise TypeError(
            f"Expected model output to be a torch.Tensor, got {type(sample_output).__name__}"
        )

    dynamic_axes = None
    if config.dynamic_batch:
        dynamic_axes = {
            config.input_name: {0: "batch"},
            config.output_name: {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            (sample_input,),
            str(config.onnx_path),
            export_params=True,
            input_names=[config.input_name],
            output_names=[config.output_name],
            opset_version=config.opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    if config.verify:
        loaded = onnx.load(str(config.onnx_path))
        onnx.checker.check_model(loaded)

    result = OnnxExportResult(
        onnx_path=config.onnx_path,
        metadata_path=config.metadata_path,
        model_name=config.model.name,
        input_shape=config.input_shape,
        output_shape=tuple(int(dim) for dim in sample_output.shape),
        opset=config.opset,
    )
    write_metadata(config, result)
    return result


def write_metadata(config: OnnxExportConfig, result: OnnxExportResult) -> None:
    payload: dict[str, Any] = {
        "model": {
            "name": config.model.name,
            "num_classes": config.model.num_classes,
            "pretrained": config.model.pretrained,
            "extra_args": config.model.extra_args,
        },
        "onnx": {
            "path": str(result.onnx_path),
            "opset": config.opset,
            "input_name": config.input_name,
            "output_name": config.output_name,
            "input_shape": list(config.input_shape),
            "output_shape": list(result.output_shape),
            "dynamic_batch": config.dynamic_batch,
        },
    }
    config.metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def result_to_json(result: OnnxExportResult) -> str:
    payload = asdict(result)
    payload["onnx_path"] = str(result.onnx_path)
    payload["metadata_path"] = str(result.metadata_path)
    return json.dumps(payload, indent=2, sort_keys=True)
