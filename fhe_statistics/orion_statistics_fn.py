from __future__ import annotations

import math
from typing import Dict, Tuple, Optional, List

import torch
from torch import nn
from torch.fx.node import Node

from .statistics_fn import FheInfo, NodeMeta, next_power_of_2_float


ORION_ACTIVATION_CONFIGS: Dict[str, Dict[str, int]] = {
    "relu": {"depth_delta": 14, "mul_both_factor": 33, "mul_single_factor": 33},
    "relu6": {"depth_delta": 14, "mul_both_factor": 33, "mul_single_factor": 33},
    "swish": {"depth_delta": 7, "mul_both_factor": 16, "mul_single_factor": 16},
    "learnable_swish": {"depth_delta": 7, "mul_both_factor": 16, "mul_single_factor": 16},
    "learnable_relu": {"depth_delta": 14, "mul_both_factor": 33, "mul_single_factor": 33},
    "sigmoid": {"depth_delta": 7, "mul_both_factor": 15, "mul_single_factor": 15},
    "poly4": {"depth_delta": 3, "mul_both_factor": 8, "mul_single_factor": 8},
}


class OrionInfo(FheInfo):
    """Orion-style latency estimator based on packing + BSGS + auto-boot ideas."""

    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        model_name: Optional[str] = None,
        optimize_boot: bool = True,
        l_eff: Optional[int] = None,
        embedding_method: str = "hybrid",
        bsgs_ratio: float = 2.0,
        use_orion_boot_cost: bool = True,
    ):
        super().__init__(model, input_shape, model_name, optimize_boot)
        if l_eff is not None:
            self.level = int(l_eff)
        self.embedding_method = str(embedding_method).lower()
        self.bsgs_ratio = float(bsgs_ratio)
        if use_orion_boot_cost:
            self.boot_cost = self._orion_boot_cost(self.level)

        self.gap_info: Dict[Node, int] = {}
        self.fhe_shape_info: Dict[Node, Optional[torch.Size]] = {}
        self._last_linear = self._find_last_linear_node()

    def _orion_boot_cost(self, l_eff: int) -> float:
        a, b, c = 3.41, 0.18, 4.81
        return a * math.exp(b * float(l_eff)) + c

    def _find_last_linear_node(self) -> Optional[Node]:
        last = None
        for node in reversed(list(self.traced.graph.nodes)):
            if node.op != "call_module":
                continue
            module = self.traced.get_submodule(str(node.target))
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                last = node
                break
        return last

    def _input_gap(self, node: Node) -> int:
        inputs = node.all_input_nodes
        if not inputs:
            return 1
        gaps = [self.gap_info.get(n, 1) for n in inputs]
        return max(gaps) if gaps else 1

    def _input_fhe_shape(self, node: Node) -> Optional[torch.Size]:
        inputs = node.all_input_nodes
        if not inputs:
            return self.get_in_shape(node)
        return self.fhe_shape_info.get(inputs[0], self.get_in_shape(node))

    def _set_layout(self, node: Node, gap: int, fhe_shape: Optional[torch.Size]) -> None:
        self.gap_info[node] = int(max(1, gap))
        self.fhe_shape_info[node] = fhe_shape

    def _new_node_meta(self, node: Node, out_depth_delta: int, in_ct: int, out_ct: int) -> NodeMeta:
        node_meta = NodeMeta()
        in_depth = self.get_in_depth(node)
        node_meta.in_depth = in_depth
        node_meta.out_depth = in_depth + out_depth_delta
        node_meta.in_ct = in_ct
        node_meta.out_ct = out_ct
        return node_meta

    def _calc_ct_from_shape(self, shape: Optional[torch.Size]) -> int:
        if shape is None:
            return 1
        if len(shape) >= 4:
            return super().calc_ct(shape)
        if len(shape) == 2:
            return int(math.ceil(shape[1] / self.slots_num))
        return 1

    def _estimate_bsgs_rotations(self, diag_count: int) -> int:
        d = max(1, int(diag_count))
        ratio = max(1e-6, self.bsgs_ratio)
        n1 = max(1, int(math.ceil(math.sqrt(d / ratio))))
        n2 = int(math.ceil(d / n1))
        return n1 + n2

    def placeholder_statistics(self, node: Node):
        super().placeholder_statistics(node)
        meta = self.get_tensor_meta(node)
        shape = meta.shape if meta else None
        self._set_layout(node, gap=1, fhe_shape=shape)

    def output_statistics(self, node: Node):
        super().output_statistics(node)
        gap = self._input_gap(node)
        fhe_shape = self._input_fhe_shape(node)
        self._set_layout(node, gap=gap, fhe_shape=fhe_shape)

    def pass_through_statistics(self, node: Node, op_type: str = "pass_through"):
        super().pass_through_statistics(node, op_type)
        gap = self._input_gap(node)
        fhe_shape = self._input_fhe_shape(node)
        self._set_layout(node, gap=gap, fhe_shape=fhe_shape)

    def conv_statistics(self, node: Node):
        module = self.traced.get_submodule(str(node.target))
        in_shape = self.get_in_shape(node)
        out_meta = self.get_tensor_meta(node)
        out_shape = out_meta.shape if out_meta else None

        input_gap = self._input_gap(node)
        stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
        output_gap = input_gap * int(stride)

        if in_shape is None or out_shape is None or len(in_shape) < 4 or len(out_shape) < 4:
            fhe_out_shape = out_shape
        else:
            Hi, Wi = in_shape[2], in_shape[3]
            _, Co, Ho, Wo = out_shape
            on_Co = math.ceil(Co / (output_gap ** 2))
            on_Ho = max(Hi, Ho * output_gap)
            on_Wo = max(Wi, Wo * output_gap)
            fhe_out_shape = torch.Size((out_shape[0], on_Co, on_Ho, on_Wo))

        fhe_in_shape = self._input_fhe_shape(node)
        in_ct = self._calc_ct_from_shape(fhe_in_shape)
        out_ct = self._calc_ct_from_shape(fhe_out_shape)

        node_meta = self._new_node_meta(node, out_depth_delta=1, in_ct=in_ct, out_ct=out_ct)

        # Estimate diagonal/block counts similar to Orion packing
        if fhe_in_shape is None or fhe_out_shape is None:
            diag_count = max(1, int(module.kernel_size[0] * module.kernel_size[1]))
            rotations = self._estimate_bsgs_rotations(diag_count)
            total_diags = diag_count
            output_rotations = 0
        else:
            _, on_Ci, on_Hi, on_Wi = fhe_in_shape
            _, on_Co, on_Ho, on_Wo = fhe_out_shape
            padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
            dilation = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation
            groups = getattr(module, "groups", 1)
            on_Ci_eff = max(1, math.ceil(on_Ci / max(1, groups)))

            Hi_pad = on_Hi + 2 * padding * input_gap
            Wi_pad = on_Wi + 2 * padding * input_gap
            matrix_height = on_Co * on_Ho * on_Wo
            matrix_width = on_Ci_eff * Hi_pad * Wi_pad

            num_block_rows = int(math.ceil(matrix_height / self.slots_num))
            num_block_cols = int(math.ceil(matrix_width / self.slots_num))

            if self.embedding_method == "hybrid" and num_block_rows == 1 and node is not self._last_linear:
                block_height = int(next_power_of_2_float(matrix_height))
                output_rotations = int(math.log2(max(1, self.slots_num // block_height)))
            else:
                block_height = self.slots_num
                output_rotations = 0

            kernel_h, kernel_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
            kernel_elems = kernel_h * kernel_w * (module.in_channels // max(1, groups))
            diag_per_block = max(1, int(min(self.slots_num, kernel_elems)))
            total_diags = diag_per_block * num_block_rows * num_block_cols
            rotations = self._estimate_bsgs_rotations(diag_per_block) * num_block_rows * num_block_cols

        node_meta.mul_single = total_diags
        node_meta.rotation = rotations + output_rotations
        node_meta.mul_both = 0
        self._set_rescale(node_meta, include_single=True)
        self._finalize_node(node, node_meta, "conv")
        self._set_layout(node, gap=output_gap, fhe_shape=fhe_out_shape)

    def avepool_statistics(self, node: Node):
        self._pool_statistics(node, op_type="avepool")

    def adaptiveavepool2d_statistics(self, node: Node):
        if node.op != "call_module":
            super().adaptiveavepool2d_statistics(node)
            in_shape = self.get_in_shape(node)
            out_meta = self.get_tensor_meta(node)
            out_shape = out_meta.shape if out_meta else None
            input_gap = self._input_gap(node)
            if in_shape and out_shape and len(in_shape) >= 4 and len(out_shape) >= 4:
                stride_h = in_shape[2] // out_shape[2]
                output_gap = input_gap * max(1, int(stride_h))
            else:
                output_gap = input_gap
            if out_shape and len(out_shape) >= 4 and out_shape[2] == 1 and out_shape[3] == 1:
                output_gap = 1
            self._set_layout(node, gap=output_gap, fhe_shape=out_shape)
            return
        self._pool_statistics(node, op_type="adaptive_avgpool")

    def linear_statistics(self, node: Node):
        module = self.traced.get_submodule(str(node.target))
        in_features = module.in_features
        out_features = module.out_features

        in_shape = self.get_in_shape(node)
        out_meta = self.get_tensor_meta(node)
        out_shape = out_meta.shape if out_meta else None

        fhe_in_shape = self._input_fhe_shape(node)
        in_ct = self._calc_ct_from_shape(fhe_in_shape)
        out_ct = self._calc_ct_from_shape(out_shape)

        node_meta = self._new_node_meta(node, out_depth_delta=1, in_ct=in_ct, out_ct=out_ct)

        num_slots = self.slots_num
        num_block_rows = int(math.ceil(out_features / num_slots))
        num_block_cols = int(math.ceil(in_features / num_slots))

        if self.embedding_method == "hybrid" and num_block_rows == 1 and node is not self._last_linear:
            block_height = int(next_power_of_2_float(out_features))
            output_rotations = int(math.log2(max(1, num_slots // block_height)))
        else:
            block_height = num_slots
            output_rotations = 0

        diag_per_block = max(1, int(block_height))
        total_diags = diag_per_block * num_block_rows * num_block_cols
        rotations = self._estimate_bsgs_rotations(diag_per_block) * num_block_rows * num_block_cols

        node_meta.mul_single = total_diags
        node_meta.rotation = rotations + output_rotations
        node_meta.mul_both = 0
        self._set_rescale(node_meta, include_single=True)
        self._finalize_node(node, node_meta, "linear")
        self._set_layout(node, gap=1, fhe_shape=out_shape)

    def activation_statistics(self, node: Node, activation_type: str):
        config = ORION_ACTIVATION_CONFIGS.get(activation_type, ORION_ACTIVATION_CONFIGS["relu"])
        fhe_in_shape = self._input_fhe_shape(node)
        in_ct = self._calc_ct_from_shape(fhe_in_shape)
        out_ct = in_ct

        node_meta = self._new_node_meta(node, out_depth_delta=config["depth_delta"], in_ct=in_ct, out_ct=out_ct)
        node_meta.mul_both = in_ct * config["mul_both_factor"]
        node_meta.mul_single = in_ct * config["mul_single_factor"]
        self._set_rescale(node_meta, include_single=True)
        self._finalize_node(node, node_meta, activation_type)
        self._set_layout(node, gap=self._input_gap(node), fhe_shape=fhe_in_shape)

    def relu_statistics(self, node: Node):
        self.activation_statistics(node, "relu")

    def relu6_statistics(self, node: Node):
        self.activation_statistics(node, "relu6")

    def learnable_swish_statistics(self, node: Node):
        self.activation_statistics(node, "learnable_swish")

    def swish_statistics(self, node: Node):
        self.activation_statistics(node, "swish")

    def learnable_relu_statistics(self, node: Node):
        self.activation_statistics(node, "learnable_relu")

    def sigmoid_statistics(self, node: Node):
        self.activation_statistics(node, "sigmoid")

    def poly4_statistics(self, node: Node):
        self.activation_statistics(node, "poly4")

    def add_statistics(self, node: Node):
        inputs = node.all_input_nodes
        max_depth = 0
        max_ct = 1
        gap = 1
        fhe_shape = None
        for in_node in inputs:
            if in_node in self.node_meta_list:
                meta = self.node_meta_list[in_node]
                max_depth = max(max_depth, meta.out_depth)
                max_ct = max(max_ct, meta.out_ct)
            gap = max(gap, self.gap_info.get(in_node, 1))
            if fhe_shape is None:
                fhe_shape = self.fhe_shape_info.get(in_node)

        node_meta = NodeMeta()
        node_meta.in_depth = max_depth
        node_meta.out_depth = max_depth
        node_meta.in_ct = max_ct
        node_meta.out_ct = max_ct
        self._finalize_node(node, node_meta, "add")
        self._set_layout(node, gap=gap, fhe_shape=fhe_shape)

    def mul_statistics(self, node: Node):
        inputs = node.all_input_nodes
        max_depth = 0
        max_ct = 1
        gap = 1
        fhe_shape = None
        for in_node in inputs:
            if in_node in self.node_meta_list:
                meta = self.node_meta_list[in_node]
                max_depth = max(max_depth, meta.out_depth)
                max_ct = max(max_ct, meta.out_ct)
            gap = max(gap, self.gap_info.get(in_node, 1))
            if fhe_shape is None:
                fhe_shape = self.fhe_shape_info.get(in_node)

        node_meta = NodeMeta()
        node_meta.in_depth = max_depth
        node_meta.out_depth = max_depth + 1
        node_meta.in_ct = max_ct
        node_meta.out_ct = max_ct
        node_meta.mul_both = max_ct
        self._set_rescale(node_meta, include_single=False)
        self._finalize_node(node, node_meta, "mul")
        self._set_layout(node, gap=gap, fhe_shape=fhe_shape)

    def cat_statistics(self, node: Node):
        inputs = node.all_input_nodes
        max_depth = 0
        total_ct = 0
        gap = 1
        for in_node in inputs:
            if in_node in self.node_meta_list:
                meta = self.node_meta_list[in_node]
                max_depth = max(max_depth, meta.out_depth)
                total_ct += meta.out_ct
            gap = max(gap, self.gap_info.get(in_node, 1))

        tensor_meta = self.get_tensor_meta(node)
        out_shape = tensor_meta.shape if tensor_meta else None
        out_ct = self._calc_ct_from_shape(out_shape)

        node_meta = NodeMeta()
        node_meta.in_depth = max_depth
        node_meta.out_depth = max_depth
        node_meta.in_ct = total_ct
        node_meta.out_ct = max(out_ct, total_ct)
        node_meta.rotation = total_ct
        self._finalize_node(node, node_meta, "cat")
        self._set_layout(node, gap=gap, fhe_shape=out_shape)

    def _pool_statistics(self, node: Node, op_type: str):
        module = self.traced.get_submodule(str(node.target))
        in_shape = self.get_in_shape(node)
        out_meta = self.get_tensor_meta(node)
        out_shape = out_meta.shape if out_meta else None

        input_gap = self._input_gap(node)

        if isinstance(module, nn.AdaptiveAvgPool2d) and in_shape and out_shape:
            stride_h = in_shape[2] // out_shape[2]
            stride_w = in_shape[3] // out_shape[3]
            stride = (stride_h, stride_w)
        else:
            stride = module.stride if hasattr(module, "stride") else 1
            if isinstance(stride, int):
                stride = (stride, stride)

        output_gap = input_gap * int(stride[0])
        if out_shape and len(out_shape) >= 4 and out_shape[2] == 1 and out_shape[3] == 1:
            output_gap = 1

        if in_shape is None or out_shape is None or len(in_shape) < 4 or len(out_shape) < 4:
            fhe_out_shape = out_shape
        else:
            Hi, Wi = in_shape[2], in_shape[3]
            _, Co, Ho, Wo = out_shape
            on_Co = math.ceil(Co / (output_gap ** 2))
            on_Ho = max(Hi, Ho * output_gap)
            on_Wo = max(Wi, Wo * output_gap)
            fhe_out_shape = torch.Size((out_shape[0], on_Co, on_Ho, on_Wo))

        fhe_in_shape = self._input_fhe_shape(node)
        in_ct = self._calc_ct_from_shape(fhe_in_shape)
        out_ct = self._calc_ct_from_shape(fhe_out_shape)

        node_meta = self._new_node_meta(node, out_depth_delta=1, in_ct=in_ct, out_ct=out_ct)

        kernel_size = module.kernel_size if hasattr(module, "kernel_size") else 1
        if isinstance(kernel_size, int):
            kernel_h, kernel_w = kernel_size, kernel_size
        else:
            kernel_h, kernel_w = kernel_size[0], kernel_size[1]

        diag_count = max(1, int(kernel_h * kernel_w))
        rotations = self._estimate_bsgs_rotations(diag_count)
        node_meta.mul_single = diag_count * out_ct
        node_meta.rotation = rotations * out_ct
        node_meta.mul_both = 0
        self._set_rescale(node_meta, include_single=True)
        self._finalize_node(node, node_meta, op_type)
        self._set_layout(node, gap=output_gap, fhe_shape=fhe_out_shape)


def analyze_model_orion(
    model: nn.Module,
    model_name: str | None = None,
    output_folder: str | None = None,
    plot_folder: str | None = None,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    print_detailed: bool = True,
    optimize_boot: bool = True,
    l_eff: Optional[int] = None,
    embedding_method: str = "hybrid",
    bsgs_ratio: float = 2.0,
) -> OrionInfo:
    if model_name:
        print(f"\nAnalyzing model (Orion): {model_name}")

    fhe_info = OrionInfo(
        model=model,
        input_shape=input_shape,
        model_name=model_name,
        optimize_boot=optimize_boot,
        l_eff=l_eff,
        embedding_method=embedding_method,
        bsgs_ratio=bsgs_ratio,
    )
    fhe_info.run_statistics()
    fhe_info.print_statistics(output_folder)

    if print_detailed and output_folder:
        fhe_info.print_detailed_statistics(output_folder)
    elif print_detailed:
        fhe_info.print_detailed_statistics(None)

    if plot_folder:
        fhe_info.plot_statistics(plot_folder=plot_folder, show=False)

    return fhe_info
