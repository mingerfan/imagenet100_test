import argparse
import difflib
import json
import os
import sys
from pathlib import Path

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

# Ensure project root is on sys.path when running from elsewhere.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from models import get_model  # noqa: E402
from network_gen import create_network  # noqa: E402
from network_gen.network_config import NetworkConfig  # noqa: E402


def _load_json_config(path: Path) -> NetworkConfig:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    config_dict = data.get("config", data)
    return NetworkConfig.from_dict(config_dict)


def _target_name(target):
    if isinstance(target, str):
        return target
    name = getattr(target, "__name__", None)
    if name:
        return name
    return repr(target)


def _node_signature(gm, node, idx):
    mod_name = ""
    if node.op == "call_module":
        try:
            mod = gm.get_submodule(node.target)
            mod_name = mod.__class__.__name__
        except Exception:
            mod_name = "<?>"
    shape = ""
    dtype = ""
    meta = node.meta.get("tensor_meta")
    if meta is not None:
        shape = str(tuple(meta.shape))
        dtype = str(meta.dtype).replace("torch.", "")
    return f"{idx:03d} {node.op:12} {_target_name(node.target):30} {mod_name:18} {shape:18} {dtype}"


def _graph_signatures(gm):
    return [_node_signature(gm, node, idx) for idx, node in enumerate(gm.graph.nodes)]


def _trace_with_shapes(model, input_size):
    model.eval()
    gm = symbolic_trace(model)
    dummy = torch.zeros(1, 3, input_size, input_size)
    ShapeProp(gm).propagate(dummy)
    return gm


def _param_count(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Compare two models with torch.fx")
    parser.add_argument("--json", type=str, required=True, help="Path to JSON config/result")
    parser.add_argument("--model", type=str, required=True, help="Registered model name")
    parser.add_argument("--num_classes", type=int, default=100, help="Num classes override")
    parser.add_argument("--input_size", type=int, default=224, help="Input size for tracing")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    config = _load_json_config(json_path)
    if args.num_classes is not None:
        config.num_classes = args.num_classes

    model_json = create_network(config)
    model_gate = get_model(args.model, num_classes=args.num_classes, pretrained=False)

    gm_json = _trace_with_shapes(model_json, args.input_size)
    gm_gate = _trace_with_shapes(model_gate, args.input_size)

    print("=" * 80)
    print("Model Summary")
    print("=" * 80)
    print(f"JSON model params: {_param_count(model_json):,}")
    print(f"Gate model params: {_param_count(model_gate):,}")

    print("\n" + "=" * 80)
    print("FX Node List (JSON)")
    print("=" * 80)
    sig_json = _graph_signatures(gm_json)
    print("\n".join(sig_json))

    print("\n" + "=" * 80)
    print("FX Node List (Gate)")
    print("=" * 80)
    sig_gate = _graph_signatures(gm_gate)
    print("\n".join(sig_gate))

    print("\n" + "=" * 80)
    print("FX Diff (Gate vs JSON)")
    print("=" * 80)
    diff = difflib.unified_diff(
        sig_json,
        sig_gate,
        fromfile="json_model",
        tofile="gate_model",
        lineterm="",
    )
    print("\n".join(diff))


if __name__ == "__main__":
    main()
