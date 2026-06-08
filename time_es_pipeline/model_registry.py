from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch.nn as nn

from paths import PROJECT_ROOT


@dataclass(frozen=True)
class ModelRequest:
    name: str = "resnet20"
    num_classes: int = 100
    pretrained: bool = False
    extra_args: dict[str, Any] = field(default_factory=dict)

    def kwargs(self) -> dict[str, Any]:
        values = dict(self.extra_args)
        values.setdefault("num_classes", self.num_classes)
        values.setdefault("pretrained", self.pretrained)
        return values


def add_project_root_to_path(project_root: Path = PROJECT_ROOT) -> None:
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def import_project_models():
    add_project_root_to_path()
    import models  # type: ignore

    return models


def list_registered_models() -> list[str]:
    models = import_project_models()
    return sorted(models.MODEL_REGISTRY.list_models())


def create_model(request: ModelRequest) -> nn.Module:
    models = import_project_models()
    return models.get_model(request.name, **request.kwargs())


def parse_key_value_items(items: list[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in model argument: {item}")
        parsed[key] = parse_value(raw_value.strip())
    return parsed


def parse_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None

    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return raw_value
