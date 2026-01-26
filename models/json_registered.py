"""
Register JSON-defined NAS networks for train.py usage.
"""

import json
import os
from typing import Any, Dict

from .registry import register_model


def _load_config_dict(json_path: str) -> Dict[str, Any]:
    if not json_path:
        raise ValueError("json_path is required")
    resolved = os.path.abspath(os.path.expanduser(json_path))
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"JSON config not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("config"), dict):
        return data["config"]
    if isinstance(data, dict):
        return data
    raise ValueError("JSON root must be an object or contain a 'config' object")


@register_model("nas-json")
def nas_json_model(
    json_path: str,
    num_classes: int = 100,
    pretrained: bool = False,
    **kwargs,
):
    """
    Build a model from a NAS JSON config.

    Args:
        json_path: Path to JSON file.
        num_classes: Override num_classes in JSON config.
        pretrained: Unused, kept for API compatibility.
    """
    _ = pretrained
    _ = kwargs
    # Local import to avoid circular dependency during package initialization
    from network_gen.network_config import NetworkConfig
    from network_gen.network_generator import create_network

    config_dict = _load_config_dict(json_path)
    config = NetworkConfig.from_dict(config_dict)
    if num_classes is not None:
        config.num_classes = num_classes
    return create_network(config)
