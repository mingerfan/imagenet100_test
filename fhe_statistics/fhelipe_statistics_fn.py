from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable

import torch
from torch import nn
from torch.fx.node import Node

from .statistics_fn import FheInfo
from .boot_optimizer import BootOptimizer, NodeInfo as BootNodeInfo


@dataclass(frozen=True)
class LayoutInfo:
    kind: str  # dense | gapped | compacted | vector | unknown
    stride_factor: int = 1


class FhelipeInfo(FheInfo):
    """Fhelipe-style latency estimator.

    - Reuses base op cost model from statistics_fn
    - Applies heuristic ciphertext compaction after gap-generating ops
    - Uses DP boot placement with adjusted ciphertext counts
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        model_name: Optional[str] = None,
        optimize_boot: bool = True,
        compaction_max_factor: int = 8,
        shortcut_boot_reduction: bool = True,
        layout_conv_penalty: float = 1.0,
    ):
        super().__init__(model, input_shape, model_name, optimize_boot)
        self.compaction_max_factor = max(1, int(compaction_max_factor))
        self.shortcut_boot_reduction = shortcut_boot_reduction
        self.layout_conv_penalty = max(0.0, float(layout_conv_penalty))
        self.layout_conv_base = max(1, int(math.log2(max(2, self.slots_num))))
        self.layout_info: Dict[Node, LayoutInfo] = {}
        self._layout_overrides: Dict[Node, LayoutInfo] = {}

    # ---- Compaction heuristics ----

    def _estimate_stride_factor(self, node: Node) -> int:
        if node.op != "call_module":
            return 1
        module = self.traced.get_submodule(str(node.target))
        stride = getattr(module, "stride", None)
        if stride is None:
            stride = getattr(module, "kernel_size", None)
        if stride is None:
            return 1
        if isinstance(stride, int):
            stride_h, stride_w = stride, stride
        else:
            stride_h, stride_w = stride[0], stride[1]
        factor = max(1, int(stride_h) * int(stride_w))
        return min(factor, self.compaction_max_factor)

    def _estimate_area_factor(self, node: Node) -> int:
        in_shape = self.get_in_shape(node)
        out_meta = self.get_tensor_meta(node)
        out_shape = out_meta.shape if out_meta else None
        if not in_shape or not out_shape or len(in_shape) < 4 or len(out_shape) < 4:
            return 1
        in_area = max(1, int(in_shape[2]) * int(in_shape[3]))
        out_area = max(1, int(out_shape[2]) * int(out_shape[3]))
        factor = math.ceil(in_area / out_area)
        return min(max(1, factor), self.compaction_max_factor)

    def _compaction_factor(self, node: Node, op_type: str) -> int:
        if op_type not in {"conv", "maxpool", "avepool", "adaptive_avgpool"}:
            return 1
        stride_factor = self._estimate_stride_factor(node)
        if stride_factor > 1:
            return stride_factor
        return self._estimate_area_factor(node)

    def _apply_compaction(self, node: Node, node_meta, op_type: str) -> None:
        factor = self._compaction_factor(node, op_type)
        if factor <= 1:
            return
        node_meta.out_ct = max(1, math.ceil(node_meta.out_ct / factor))

    # ---- Boot heuristics ----

    def _boot_ct(self, node: Node, meta) -> int:
        if not self.shortcut_boot_reduction or meta.op_type != "add":
            return max(1, meta.out_ct)
        # Heuristic: skip some shortcut bootstraps by using the smaller input ct
        in_cts = []
        for in_node in node.all_input_nodes:
            if in_node in self.node_meta_list:
                in_cts.append(self.node_meta_list[in_node].out_ct)
        if not in_cts:
            return max(1, meta.out_ct)
        return max(1, min(in_cts))

    def _calc_boot_fhelipe(self, node: Node, node_meta) -> None:
        boots_before_in = node_meta.in_depth // self.level
        boots_before_out = node_meta.out_depth // self.level
        boot_triggered = max(0, boots_before_out - boots_before_in)
        node_meta.boot_count = boot_triggered
        node_meta.boot_latency = boot_triggered * self._boot_ct(node, node_meta) * self.boot_cost

    # ---- Preserve layout for fused/pass-through nodes ----

    def placeholder_statistics(self, node: Node):
        super().placeholder_statistics(node)
        self.layout_info[node] = LayoutInfo(kind="dense", stride_factor=1)

    def output_statistics(self, node: Node):
        super().output_statistics(node)
        self.layout_info[node] = self._input_layout(node)

    def pass_through_statistics(self, node: Node, op_type: str = "pass_through"):
        super().pass_through_statistics(node, op_type)
        self.layout_info[node] = self._input_layout(node)

    # ---- Layout heuristics ----

    def _default_layout(self) -> LayoutInfo:
        return LayoutInfo(kind="dense", stride_factor=1)

    def _input_layout(self, node: Node) -> LayoutInfo:
        inputs = node.all_input_nodes
        if not inputs:
            return self._default_layout()
        return self.layout_info.get(inputs[0], self._default_layout())

    def _infer_layout(self, node: Node, op_type: str) -> LayoutInfo:
        if op_type in {"relu", "relu6", "swish", "learnable_swish", "learnable_relu",
                       "sigmoid", "poly4", "pass_through", "add", "mul"}:
            return self._input_layout(node)

        if op_type in {"conv", "maxpool", "avepool", "adaptive_avgpool"}:
            factor = self._compaction_factor(node, op_type)
            if factor > 1:
                return LayoutInfo(kind="compacted", stride_factor=factor)
            return LayoutInfo(kind="dense", stride_factor=1)

        if op_type in {"linear", "cat"}:
            return LayoutInfo(kind="vector", stride_factor=1)

        return self._input_layout(node)

    def _layout_distance(self, a: LayoutInfo, b: LayoutInfo) -> int:
        if a == b:
            return 0
        distance = 1
        if a.kind != b.kind:
            distance += 1
        if a.stride_factor != b.stride_factor:
            distance += max(1, int(math.log2(max(a.stride_factor, b.stride_factor))))
        return distance

    def _conversion_rotations(self, from_layout: LayoutInfo, to_layout: LayoutInfo, ct: int) -> int:
        if from_layout == to_layout or ct <= 0:
            return 0
        distance = self._layout_distance(from_layout, to_layout)
        base = max(1, int(self.layout_conv_base * self.layout_conv_penalty))
        return int(ct * base * distance)

    def _choose_binary_layout(
        self,
        node: Node,
        inputs: Iterable[Node],
    ) -> Tuple[LayoutInfo, int]:
        input_nodes = list(inputs)
        if len(input_nodes) < 2:
            return self._input_layout(node), 0

        left, right = input_nodes[0], input_nodes[1]
        left_layout = self.layout_info.get(left, self._default_layout())
        right_layout = self.layout_info.get(right, self._default_layout())
        left_ct = self.node_meta_list.get(left).out_ct if left in self.node_meta_list else 1
        right_ct = self.node_meta_list.get(right).out_ct if right in self.node_meta_list else 1

        cost_left_to_right = self._conversion_rotations(left_layout, right_layout, left_ct)
        cost_right_to_left = self._conversion_rotations(right_layout, left_layout, right_ct)

        if cost_left_to_right <= cost_right_to_left:
            return right_layout, cost_left_to_right
        return left_layout, cost_right_to_left

    def _finalize_node(self, node: Node, node_meta, op_type: str, is_fused: bool = False):
        node_meta.op_type = op_type
        node_meta.is_fused = is_fused
        if not is_fused:
            self._apply_compaction(node, node_meta, op_type)
        self._calc_boot_fhelipe(node, node_meta)
        self._calc_latency(node_meta)
        self.node_meta_list[node] = node_meta
        if not is_fused:
            if node in self._layout_overrides:
                self.layout_info[node] = self._layout_overrides.pop(node)
            else:
                self.layout_info[node] = self._infer_layout(node, op_type)

    def _optimize_boot_insertion(self):
        # Collect nodes in topo order (excluding fused)
        boot_nodes = []
        node_to_idx: Dict[Node, int] = {}
        boot_ct_map: Dict[int, int] = {}

        for node in self.traced.graph.nodes:
            if node not in self.node_meta_list:
                continue
            meta = self.node_meta_list[node]
            if meta.is_fused:
                continue

            depth_delta = meta.out_depth - meta.in_depth
            ct_num = self._boot_ct(node, meta)
            boot_node_info = BootNodeInfo(
                index=len(boot_nodes),
                name=node.name,
                depth_delta=depth_delta,
                ct_num=ct_num,
                op_type=meta.op_type,
            )
            boot_ct_map[len(boot_nodes)] = ct_num
            boot_nodes.append(boot_node_info)
            node_to_idx[node] = len(boot_nodes) - 1

        if not boot_nodes:
            return

        optimizer = BootOptimizer(self.level, self.boot_cost)
        boot_plan, _ = optimizer.optimize(boot_nodes)

        for node, meta in self.node_meta_list.items():
            meta.boot_count = 0
            meta.boot_latency = 0

        active_node_list = []
        for node in self.traced.graph.nodes:
            if node in node_to_idx:
                active_node_list.append(node)

        for boot_idx, boot_count in boot_plan.items():
            if boot_idx < len(active_node_list):
                node = active_node_list[boot_idx]
                meta = self.node_meta_list[node]
                ct_num = boot_ct_map.get(boot_idx, self._boot_ct(node, meta))
                meta.boot_count = boot_count
                meta.boot_latency = boot_count * ct_num * self.boot_cost

    # ---- Override binary ops to include layout conversion ----

    def add_statistics(self, node: Node):
        inputs = node.all_input_nodes
        max_depth = 0
        max_ct = 1
        for in_node in inputs:
            if in_node in self.node_meta_list:
                meta = self.node_meta_list[in_node]
                max_depth = max(max_depth, meta.out_depth)
                max_ct = max(max_ct, meta.out_ct)

        node_meta = super()._init_node_meta(node, out_depth_delta=0)[0]
        node_meta.in_depth = max_depth
        node_meta.out_depth = max_depth
        node_meta.in_ct = max_ct
        node_meta.out_ct = max_ct
        node_meta.rotation = 0
        node_meta.mul_single = 0
        node_meta.mul_both = 0
        node_meta.rescale = 0

        layout, conv_rot = self._choose_binary_layout(node, inputs)
        node_meta.rotation += conv_rot
        self._layout_overrides[node] = layout

        self._finalize_node(node, node_meta, "add")

    def mul_statistics(self, node: Node):
        inputs = node.all_input_nodes
        max_depth = 0
        max_ct = 1
        for in_node in inputs:
            if in_node in self.node_meta_list:
                meta = self.node_meta_list[in_node]
                max_depth = max(max_depth, meta.out_depth)
                max_ct = max(max_ct, meta.out_ct)

        node_meta = super()._init_node_meta(node, out_depth_delta=1)[0]
        node_meta.in_depth = max_depth
        node_meta.out_depth = max_depth + 1
        node_meta.in_ct = max_ct
        node_meta.out_ct = max_ct
        node_meta.mul_both = max_ct
        node_meta.mul_single = 0
        node_meta.rotation = 0
        self._set_rescale(node_meta, include_single=False)

        layout, conv_rot = self._choose_binary_layout(node, inputs)
        node_meta.rotation += conv_rot
        self._layout_overrides[node] = layout

        self._finalize_node(node, node_meta, "mul")

    def cat_statistics(self, node: Node):
        inputs = node.all_input_nodes
        max_depth = 0
        total_ct = 0
        for in_node in inputs:
            if in_node in self.node_meta_list:
                meta = self.node_meta_list[in_node]
                max_depth = max(max_depth, meta.out_depth)
                total_ct += meta.out_ct

        tensor_meta = self.get_tensor_meta(node)
        out_shape = tensor_meta.shape if tensor_meta else None
        out_ct = self.calc_ct(out_shape) if out_shape and len(out_shape) >= 4 else total_ct

        node_meta = super()._init_node_meta(node, out_depth_delta=0)[0]
        node_meta.in_depth = max_depth
        node_meta.out_depth = max_depth
        node_meta.in_ct = total_ct
        node_meta.out_ct = out_ct
        node_meta.rotation = total_ct
        node_meta.mul_single = 0
        node_meta.mul_both = 0
        node_meta.rescale = 0

        # Align all inputs to the first input layout
        if inputs:
            target_layout = self.layout_info.get(inputs[0], self._default_layout())
            conv_rot = 0
            for in_node in inputs[1:]:
                in_layout = self.layout_info.get(in_node, self._default_layout())
                in_ct = self.node_meta_list.get(in_node).out_ct if in_node in self.node_meta_list else 1
                conv_rot += self._conversion_rotations(in_layout, target_layout, in_ct)
            node_meta.rotation += conv_rot
            self._layout_overrides[node] = target_layout

        self._finalize_node(node, node_meta, "cat")


def analyze_model_fhelipe(
    model: nn.Module,
    model_name: str | None = None,
    output_folder: str | None = None,
    plot_folder: str | None = None,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    print_detailed: bool = True,
    optimize_boot: bool = True,
    compaction_max_factor: int = 8,
    shortcut_boot_reduction: bool = True,
    layout_conv_penalty: float = 1.0,
):
    """Analyze model using Fhelipe-style heuristics."""
    if model_name:
        print(f"\nAnalyzing model (Fhelipe): {model_name}")

    fhelipe_info = FhelipeInfo(
        model,
        input_shape,
        model_name,
        optimize_boot=optimize_boot,
        compaction_max_factor=compaction_max_factor,
        shortcut_boot_reduction=shortcut_boot_reduction,
        layout_conv_penalty=layout_conv_penalty,
    )
    fhelipe_info.run_statistics()

    fhelipe_info.print_statistics(output_folder)

    if print_detailed and output_folder:
        fhelipe_info.print_detailed_statistics(output_folder)
    elif print_detailed:
        fhelipe_info.print_detailed_statistics(None)

    if plot_folder:
        fhelipe_info.plot_statistics(plot_folder=plot_folder, show=False)

    return fhelipe_info
