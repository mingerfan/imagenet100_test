import operator
import math
import sys
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

import torch
import torchvision
from torch import fx, nn
from torch.fx.node import Node
from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata

# 尝试导入自定义模块
try:
    sys.path.insert(0, '..')
    from models.gate_net_cmp.block_def import (
        LearnableSwish, LearnableRelu, StablePoly7, Relu,
        SelfGated, BasicBlock, BottleneckBlock, BasicSelfGatedBlock, BottleneckSelfGatedBlock
    )
    HAS_CUSTOM_MODULES = True
except ImportError:
    HAS_CUSTOM_MODULES = False


    def next_power_of_2_float(n):
        if n <= 0:
            return 1
        return 2 ** math.ceil(math.log2(n))


def get_script_dir():
    """获取脚本所在目录的绝对路径"""
    return os.path.dirname(os.path.abspath(__file__))


def ensure_dir_exists(dir_path: str):
    """确保目录存在，如果不存在则创建"""
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def generate_unique_filename(model_name: str, extension: str, folder_path: str | None = None) -> str:
    """生成唯一的文件名
    
    Args:
        model_name: 模型名称
        extension: 文件扩展名（如 '.txt', '.png'）
        folder_path: 文件夹路径，如果为 None 则使用脚本所在目录
    
    Returns:
        完整的文件路径
    """
    # 确保扩展名以点开头
    if not extension.startswith('.'):
        extension = '.' + extension
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成文件名
    filename = f"{model_name}_{timestamp}{extension}"
    
    # 确定文件夹路径
    if folder_path is None:
        folder_path = get_script_dir()
    else:
        # 如果是相对路径，转换为相对于脚本所在目录的绝对路径
        if not os.path.isabs(folder_path):
            folder_path = os.path.join(get_script_dir(), folder_path)
    
    # 确保目录存在
    ensure_dir_exists(folder_path)
    
    # 返回完整路径
    return os.path.join(folder_path, filename)


class NodeMeta:
    def __init__(self):
        self.rotation = 0
        self.mul_single = 0
        self.rescale = 0
        self.mul_both = 0
        self.in_depth = 0
        self.out_depth = 0
        self.in_ct = 0
        self.out_ct = 0

        # 新增字段
        self.op_type = "unknown"  # 算子类型
        self.boot_count = 0       # 该算子触发的boot次数
        self.boot_latency = 0     # boot耗时
        self.latency = 0          # 总耗时(不含boot)
        self.is_fused = False     # 是否被fused（不计入统计）


class FheInfo:
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 224, 224)):
        # 成本参数
        self.rotation_cost = 200
        self.rescale_cost = 50
        self.mul_single_cost = 9.5
        self.mul_double_cost = 253
        self.boot_cost = 98136
        self.level = 10

        self.slots_num = 32768
        self.input_shape = input_shape

        self.model = model.eval()
        self.traced = fx.symbolic_trace(model)
        dummy_input = torch.randn(*input_shape)
        ShapeProp(self.traced).propagate(dummy_input)

        self.node_meta_list: Dict[Node, NodeMeta] = {}

        # 统计汇总
        self.op_stats: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0,
            "rotation": 0,
            "mul_single": 0,
            "mul_both": 0,
            "rescale": 0,
            "latency": 0,
            "boot_latency": 0,
            "boot_count": 0,
        })
        self.total_boot_count = 0
        self.total_boot_latency = 0
        self.total_latency = 0

    def get_in_depth(self, node: Node) -> int:
        inputs = node.all_input_nodes
        if not inputs:
            return 0
        cur_depth_max = 0
        for in_node in inputs:
            if in_node in self.node_meta_list:
                in_meta = self.node_meta_list[in_node]
                cur_depth_max = max(cur_depth_max, in_meta.out_depth)
        return cur_depth_max

    def get_in_shape(self, node: Node) -> Optional[torch.Size]:
        inputs = node.all_input_nodes
        if not inputs:
            return None
        input_node = inputs[0]
        meta = input_node.meta.get("tensor_meta")
        if meta is None:
            return None
        return meta.shape

    def get_in_ct(self, node: Node) -> int:
        """获取输入的ct数量"""
        inputs = node.all_input_nodes
        if not inputs:
            return 1
        in_node = inputs[0]
        if in_node in self.node_meta_list:
            return self.node_meta_list[in_node].out_ct
        return 1

    def calc_ct(self, shape: torch.Size) -> int:
        if len(shape) < 4:
            # 对于 flatten 后的向量，假设1个ct
            return 1
        resovle_prod = shape[2] * shape[3]
        if resovle_prod > self.slots_num:
            min_ct_per_feature_map = next_power_of_2_float(resovle_prod / self.slots_num)
            ct_num = min_ct_per_feature_map * shape[1]
        else:
            ct_num = math.ceil(shape[1] / math.floor(self.slots_num / resovle_prod))
        return ct_num

    def calc_channel_per_ct(self, shape: torch.Size) -> int:
        if len(shape) < 4:
            return 1
        return math.floor(self.slots_num / (shape[2] * shape[3]))

    def get_tensor_meta(self, node: Node) -> Optional[TensorMetadata]:
        return node.meta.get("tensor_meta")

    def _init_node_meta(self, node: Node, out_depth_delta: int = 0) -> Tuple[NodeMeta, Optional[torch.Size], Optional[torch.Size], int, int]:
        in_shape = self.get_in_shape(node)
        tensor_meta = self.get_tensor_meta(node)
        out_shape = tensor_meta.shape if tensor_meta else None

        in_ct = self.calc_ct(in_shape) if in_shape and len(in_shape) >= 4 else self.get_in_ct(node)
        out_ct = self.calc_ct(out_shape) if out_shape and len(out_shape) >= 4 else 1

        node_meta = NodeMeta()
        in_depth = self.get_in_depth(node)
        node_meta.in_depth = in_depth
        node_meta.out_depth = in_depth + out_depth_delta
        node_meta.in_ct = in_ct
        node_meta.out_ct = out_ct

        return node_meta, in_shape, out_shape, in_ct, out_ct

    def _set_rescale(self, node_meta: NodeMeta, include_single: bool = False):
        if include_single:
            node_meta.rescale = node_meta.mul_both + node_meta.mul_single
        else:
            node_meta.rescale = node_meta.mul_both

    def _calc_boot(self, node_meta: NodeMeta):
        """计算该节点触发的boot次数和耗时"""
        in_boots = math.ceil(node_meta.in_depth / self.level) if node_meta.in_depth > 0 else 0
        out_boots = math.ceil(node_meta.out_depth / self.level) if node_meta.out_depth > 0 else 0
        boot_triggered = max(0, out_boots - in_boots)
        node_meta.boot_count = boot_triggered
        node_meta.boot_latency = boot_triggered * node_meta.out_ct * self.boot_cost

    def _calc_latency(self, node_meta: NodeMeta):
        """计算该节点的操作耗时（不含boot）"""
        node_meta.latency = (
            node_meta.rotation * self.rotation_cost +
            node_meta.mul_single * self.mul_single_cost +
            node_meta.mul_both * self.mul_double_cost +
            node_meta.rescale * self.rescale_cost
        )

    def _finalize_node(self, node: Node, node_meta: NodeMeta, op_type: str, is_fused: bool = False):
        """完成节点处理：计算boot、延迟、记录"""
        node_meta.op_type = op_type
        node_meta.is_fused = is_fused
        self._calc_boot(node_meta)
        self._calc_latency(node_meta)
        self.node_meta_list[node] = node_meta

    # ========== 各算子统计函数 ==========

    def placeholder_statistics(self, node: Node):
        """输入节点"""
        tensor_meta = self.get_tensor_meta(node)
        node_meta = NodeMeta()
        node_meta.in_depth = 0
        node_meta.out_depth = 0
        if tensor_meta and len(tensor_meta.shape) >= 4:
            node_meta.out_ct = self.calc_ct(tensor_meta.shape)
        else:
            node_meta.out_ct = 1
        node_meta.in_ct = node_meta.out_ct
        node_meta.op_type = "placeholder"
        node_meta.is_fused = True  # 不计入统计
        self.node_meta_list[node] = node_meta

    def output_statistics(self, node: Node):
        """输出节点"""
        node_meta = NodeMeta()
        in_depth = self.get_in_depth(node)
        node_meta.in_depth = in_depth
        node_meta.out_depth = in_depth
        node_meta.in_ct = self.get_in_ct(node)
        node_meta.out_ct = node_meta.in_ct
        node_meta.op_type = "output"
        node_meta.is_fused = True
        self.node_meta_list[node] = node_meta

    def conv_statistics(self, node: Node):
        module = self.traced.get_submodule(str(node.target))
        out_channels = module.out_channels
        kernel_size = module.kernel_size

        node_meta, in_shape, out_shape, in_ct, out_ct = self._init_node_meta(node, out_depth_delta=2)

        if in_shape and len(in_shape) >= 4:
            ch_per_ct = self.calc_channel_per_ct(in_shape)
        else:
            ch_per_ct = 1
        if out_shape and len(out_shape) >= 4:
            out_ch_per_ct = self.calc_channel_per_ct(out_shape)
        else:
            out_ch_per_ct = 1

        kernel_ratio = kernel_size[0] * kernel_size[1]
        node_meta.mul_single = out_channels * in_ct * kernel_ratio
        node_meta.rotation = math.ceil(math.log2(max(1, ch_per_ct))) + out_ch_per_ct * out_ct
        node_meta.mul_both = 0

        self._set_rescale(node_meta, include_single=True)
        self._finalize_node(node, node_meta, "conv")

    def relu_statistics(self, node: Node):
        node_meta, in_shape, out_shape, in_ct, out_ct = self._init_node_meta(node, out_depth_delta=15)
        node_meta.mul_both = in_ct * 33
        node_meta.mul_single = in_ct * 33
        self._set_rescale(node_meta, include_single=True)
        self._finalize_node(node, node_meta, "relu")

    def relu6_statistics(self, node: Node):
        node_meta, in_shape, out_shape, in_ct, out_ct = self._init_node_meta(node, out_depth_delta=15)
        node_meta.mul_both = in_ct * 33
        node_meta.mul_single = in_ct * 33
        self._set_rescale(node_meta, include_single=True)
        self._finalize_node(node, node_meta, "relu6")
        
    def learnable_swish_statistics(self, node: Node):
        node_meta, in_shape, out_shape, in_ct, out_ct = self._init_node_meta(node, out_depth_delta=15)
        node_meta.mul_both = in_ct * 33
        node_meta.mul_single = in_ct * 33
        self._set_rescale(node_meta, include_single=True)
        self._finalize_node(node, node_meta, "learnable_swish")

    def swish_statistics(self, node: Node):
        node_meta, in_shape, out_shape, in_ct, out_ct = self._init_node_meta(node, out_depth_delta=15)
        node_meta.mul_both = in_ct * 32
        node_meta.mul_single = in_ct * 32
        self._set_rescale(node_meta, include_single=True)
        self._finalize_node(node, node_meta, "learnable_swish")

    def learnable_relu_statistics(self, node: Node):
        """LearnableRelu: maximum(beta*x, 0) 使用sign实现，与relu类似"""
        node_meta, in_shape, out_shape, in_ct, out_ct = self._init_node_meta(node, out_depth_delta=15)
        node_meta.mul_both = in_ct * 33
        node_meta.mul_single = in_ct * 33
        self._set_rescale(node_meta, include_single=True)
        self._finalize_node(node, node_meta, "learnable_relu")

    def sigmoid_statistics(self, node: Node):
        node_meta, in_shape, out_shape, in_ct, out_ct = self._init_node_meta(node, out_depth_delta=15)
        node_meta.mul_both = in_ct * 31
        node_meta.mul_single = in_ct * 31
        self._set_rescale(node_meta, include_single=True)
        self._finalize_node(node, node_meta, "learnable_relu")


    def poly7_statistics(self, node: Node):
        """StablePoly7: 4次多项式，深度3"""
        node_meta, in_shape, out_shape, in_ct, out_ct = self._init_node_meta(node, out_depth_delta=3)
        node_meta.mul_both = in_ct * 8
        node_meta.mul_single = in_ct * 8
        self._set_rescale(node_meta, include_single=True)
        self._finalize_node(node, node_meta, "poly7")

    def maxpool_statistics(self, node: Node):
        node_meta, in_shape, out_shape, in_ct, out_ct = self._init_node_meta(node, out_depth_delta=30)
        node_meta.rotation = in_ct * 2
        node_meta.mul_both = 64 * in_ct
        node_meta.mul_single = 64 * in_ct
        self._set_rescale(node_meta, include_single=True)
        self._finalize_node(node, node_meta, "maxpool")

    def avepool_statistics(self, node: Node):
        node_meta, in_shape, out_shape, in_ct, out_ct = self._init_node_meta(node, out_depth_delta=1)
        node_meta.rotation = in_ct * 2
        node_meta.mul_single = in_ct
        self._set_rescale(node_meta, include_single=True)
        self._finalize_node(node, node_meta, "avepool")

    def adaptiveavepool2d_statistics(self, node: Node):
        in_shape = self.get_in_shape(node)
        in_ct = self.calc_ct(in_shape) if in_shape and len(in_shape) >= 4 else self.get_in_ct(node)

        node_meta = NodeMeta()
        in_depth = self.get_in_depth(node)
        node_meta.in_depth = in_depth
        node_meta.out_depth = in_depth + 1
        node_meta.in_ct = in_ct
        node_meta.out_ct = 1

        if in_shape and len(in_shape) >= 4:
            feature_map_size = in_shape[2] * in_shape[3]
            log2_map_size = math.ceil(math.log2(max(1, feature_map_size)))
            node_meta.rotation = log2_map_size * in_ct + int(0.5 * in_shape[1])
        else:
            node_meta.rotation = in_ct

        node_meta.mul_single = 1
        node_meta.mul_both = 0
        self._set_rescale(node_meta, include_single=False)
        self._finalize_node(node, node_meta, "adaptive_avgpool")

    def linear_statistics(self, node: Node):
        """全连接层：涉及大量旋转和密文明文乘法"""
        module = self.traced.get_submodule(str(node.target))
        in_features = module.in_features
        # out_features = module.out_features

        node_meta, in_shape, out_shape, in_ct, out_ct = self._init_node_meta(node, out_depth_delta=2)

        # 密文明文乘法
        node_meta.mul_single = in_features

        node_meta.rotation = in_features - 1

        node_meta.mul_both = 0
        self._set_rescale(node_meta, include_single=True)
        self._finalize_node(node, node_meta, "linear")

    def add_statistics(self, node: Node):
        """残差连接：深度取max，加法本身无额外消耗"""
        inputs = node.all_input_nodes
        max_depth = 0
        max_ct = 1
        for in_node in inputs:
            if in_node in self.node_meta_list:
                meta = self.node_meta_list[in_node]
                max_depth = max(max_depth, meta.out_depth)
                max_ct = max(max_ct, meta.out_ct)

        node_meta = NodeMeta()
        node_meta.in_depth = max_depth
        node_meta.out_depth = max_depth  # 不增加深度
        node_meta.in_ct = max_ct
        node_meta.out_ct = max_ct
        node_meta.rotation = 0
        node_meta.mul_single = 0
        node_meta.mul_both = 0
        node_meta.rescale = 0
        self._finalize_node(node, node_meta, "add")

    def mul_statistics(self, node: Node):
        """元素乘法：深度+1"""
        inputs = node.all_input_nodes
        max_depth = 0
        max_ct = 1
        for in_node in inputs:
            if in_node in self.node_meta_list:
                meta = self.node_meta_list[in_node]
                max_depth = max(max_depth, meta.out_depth)
                max_ct = max(max_ct, meta.out_ct)

        node_meta = NodeMeta()
        node_meta.in_depth = max_depth
        node_meta.out_depth = max_depth + 1  # 乘法增加深度
        node_meta.in_ct = max_ct
        node_meta.out_ct = max_ct
        node_meta.mul_both = max_ct
        node_meta.mul_single = 0
        node_meta.rotation = 0
        self._set_rescale(node_meta, include_single=False)
        self._finalize_node(node, node_meta, "mul")

    def cat_statistics(self, node: Node):
        """拼接操作：主要是旋转重排"""
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

        node_meta = NodeMeta()
        node_meta.in_depth = max_depth
        node_meta.out_depth = max_depth
        node_meta.in_ct = total_ct
        node_meta.out_ct = out_ct
        # cat 可能需要旋转来重排数据
        node_meta.rotation = total_ct  # 估计每个输入ct需要一次旋转
        node_meta.mul_single = 0
        node_meta.mul_both = 0
        node_meta.rescale = 0
        self._finalize_node(node, node_meta, "cat")

    def pass_through_statistics(self, node: Node, op_type: str = "pass_through"):
        """处理被fused或无操作的节点"""
        in_shape = self.get_in_shape(node)
        tensor_meta = self.get_tensor_meta(node)
        out_shape = tensor_meta.shape if tensor_meta else None

        if in_shape and len(in_shape) >= 4:
            in_ct = self.calc_ct(in_shape)
        else:
            in_ct = self.get_in_ct(node)

        if out_shape and len(out_shape) >= 4:
            out_ct = self.calc_ct(out_shape)
        else:
            out_ct = in_ct

        node_meta = NodeMeta()
        in_depth = self.get_in_depth(node)
        node_meta.in_depth = in_depth
        node_meta.out_depth = in_depth
        node_meta.in_ct = in_ct
        node_meta.out_ct = out_ct
        node_meta.rotation = 0
        node_meta.mul_single = 0
        node_meta.mul_both = 0
        node_meta.rescale = 0
        node_meta.op_type = op_type
        node_meta.is_fused = True  # 不计入统计
        self.node_meta_list[node] = node_meta

    # ========== 主遍历逻辑 ==========

    def run_statistics(self):
        """遍历FX图，对每个节点进行统计"""
        for node in self.traced.graph.nodes:
            if node.op == "placeholder":
                self.placeholder_statistics(node)
            elif node.op == "output":
                self.output_statistics(node)
            elif node.op == "call_module":
                self._handle_module(node)
            elif node.op == "call_function":
                self._handle_function(node)
            elif node.op == "call_method":
                self._handle_method(node)
            elif node.op == "get_attr":
                self.pass_through_statistics(node, "get_attr")
            else:
                raise ValueError(f"未知OP：{node.op}")
                self.pass_through_statistics(node, f"unknown_{node.op}")

        # 汇总统计
        self._aggregate_stats()

    def _handle_module(self, node: Node):
        """处理 call_module 节点"""
        module = self.traced.get_submodule(str(node.target))

        # Conv2d
        if isinstance(module, nn.Conv2d):
            self.conv_statistics(node)
        # Linear
        elif isinstance(module, nn.Linear):
            self.linear_statistics(node)
        # ReLU
        elif isinstance(module, nn.ReLU):
            self.relu_statistics(node)
        # ReLU6
        elif isinstance(module, nn.ReLU6):
            self.relu6_statistics(node)
        # Swish
        elif isinstance(module, nn.SiLU):
            self.swish_statistics(node)
        # Sigmoid
        elif isinstance(module, nn.Sigmoid):
            self.sigmoid_statistics(node)
        # MaxPool2d
        elif isinstance(module, nn.MaxPool2d):
            self.maxpool_statistics(node)
        # AvgPool2d
        elif isinstance(module, nn.AvgPool2d):
            self.avepool_statistics(node)
        # AdaptiveAvgPool2d
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            self.adaptiveavepool2d_statistics(node)
        # Fused operators: BatchNorm, Dropout, Identity
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            self.pass_through_statistics(node, "batchnorm")
        elif isinstance(module, nn.Dropout):
            self.pass_through_statistics(node, "dropout")
        elif isinstance(module, nn.Identity):
            self.pass_through_statistics(node, "identity")
        # 自定义激活函数
        elif HAS_CUSTOM_MODULES:
            if isinstance(module, LearnableSwish):
                self.learnable_swish_statistics(node)
            elif isinstance(module, LearnableRelu):
                self.learnable_relu_statistics(node)
            elif isinstance(module, StablePoly7):
                self.poly7_statistics(node)
            elif isinstance(module, Relu):
                self.relu_statistics(node)
            else:
                raise ValueError(f"未知激活函数{type(module).__name__}")
                self.pass_through_statistics(node, f"unknown_module_{type(module).__name__}")
        else:
            raise ValueError(f"未知模块{type(module).__name__}")
            self.pass_through_statistics(node, f"unknown_module_{type(module).__name__}")

    def _handle_function(self, node: Node):
        """处理 call_function 节点"""
        target = node.target

        # add
        if target in (torch.add, operator.add):
            self.add_statistics(node)
        # mul
        elif target in (torch.mul, operator.mul):
            self.mul_statistics(node)
        # flatten
        elif target == torch.flatten:
            self.pass_through_statistics(node, "flatten")
        # cat
        elif target == torch.cat:
            self.cat_statistics(node)
        # relu
        elif target == torch.relu or target == torch.nn.functional.relu:
            self.relu_statistics(node)
        # sigmoid
        elif target == torch.sigmoid or target == torch.nn.functional.sigmoid:
            self.sigmoid_statistics(node)
        elif target == torch.nn.functional.adaptive_avg_pool2d:
            self.adaptiveavepool2d_statistics(node)
        elif target == getattr:
            self.pass_through_statistics(node, "getattr")
        elif "stochastic_depth" in str(target):
            self.pass_through_statistics(node)
        else:
            raise ValueError(f"未知函数{getattr(target, '__name__', str(target))}")
            self.pass_through_statistics(node, f"unknown_func_{getattr(target, '__name__', str(target))}")

    def _handle_method(self, node: Node):
        """处理 call_method 节点"""
        method_name = node.target

        if method_name == "flatten":
            self.pass_through_statistics(node, "flatten")
        elif method_name == "view":
            self.pass_through_statistics(node, "view")
        elif method_name == "reshape":
            self.pass_through_statistics(node, "reshape")
        elif method_name == "contiguous":
            self.pass_through_statistics(node, "contiguous")
        else:
            raise ValueError(f"未知方法{method_name}")
            self.pass_through_statistics(node, f"unknown_method_{method_name}")

    def _aggregate_stats(self):
        """汇总统计结果"""
        self.op_stats.clear()
        self.total_boot_count = 0
        self.total_boot_latency = 0
        self.total_latency = 0

        for node, meta in self.node_meta_list.items():
            if meta.is_fused:
                continue  # 跳过fused算子

            op_type = meta.op_type
            self.op_stats[op_type]["count"] += 1
            self.op_stats[op_type]["rotation"] += meta.rotation
            self.op_stats[op_type]["mul_single"] += meta.mul_single
            self.op_stats[op_type]["mul_both"] += meta.mul_both
            self.op_stats[op_type]["rescale"] += meta.rescale
            self.op_stats[op_type]["latency"] += meta.latency
            self.op_stats[op_type]["boot_latency"] += meta.boot_latency
            self.op_stats[op_type]["boot_count"] += meta.boot_count

            self.total_boot_count += meta.boot_count
            self.total_boot_latency += meta.boot_latency
            self.total_latency += meta.latency

    # ========== 输出与可视化 ==========

    def print_statistics(self, output_folder: Optional[str] = None):
        """打印统计结果并保存到文件
        
        Args:
            output_folder: 输出文件夹路径（如果为None则不保存到文件）
        """
        lines = []
        model_name = type(self.model).__name__
        lines.append("=" * 110)
        lines.append(f"FHE Statistics for {model_name}")
        lines.append("=" * 110)

        total_with_boot = self.total_latency + self.total_boot_latency

        # 表头
        header = f"{'Op Type':<20} {'Count':>8} {'Rotation':>12} {'MulSingle':>12} {'MulBoth':>12} {'Latency':>16} {'Boot':>16} {'Pct(%)':>8}"
        lines.append(header)
        lines.append("-" * 110)

        # 各算子统计
        for op_type, stats in sorted(self.op_stats.items(), key=lambda x: x[1]["latency"], reverse=True):
            latency = stats["latency"]
            boot = stats["boot_latency"]
            pct = (stats["latency"] + stats["boot_latency"]) / total_with_boot * 100 if total_with_boot > 0 else 0
            line = f"{op_type:<20} {stats['count']:>8} {stats['rotation']:>12} {stats['mul_single']:>12} {stats['mul_both']:>12} {latency:>16.2f} {boot:>16.2f} {pct:>8.2f}"
            lines.append(line)

        lines.append("-" * 110)

        # Boot 单独统计
        boot_pct = self.total_boot_latency / total_with_boot * 100 if total_with_boot > 0 else 0
        lines.append(f"{'Boot (Total)':<20} {self.total_boot_count:>8} {'-':>12} {'-':>12} {'-':>12} {'-':>16} {self.total_boot_latency:>16.2f} {boot_pct:>8.2f}")

        lines.append("=" * 110)
        lines.append(f"Total Latency (without boot): {self.total_latency:.2f}")
        lines.append(f"Total Boot Latency: {self.total_boot_latency:.2f}")
        lines.append(f"Total Latency (with boot): {total_with_boot:.2f}")
        lines.append(f"Max Depth: {max(m.out_depth for m in self.node_meta_list.values())}")
        lines.append("=" * 110)

        output = "\n".join(lines)
        print(output)

        if output_folder:
            # 生成唯一文件名
            output_file = generate_unique_filename(model_name, "txt", output_folder)
            with open(output_file, "w") as f:
                f.write(output)
            print(f"\nResults saved to {output_file}")

    def plot_statistics(self, plot_folder: Optional[str] = None, show: bool = True):
        """绘制统计结果并保存到文件
        
        Args:
            plot_folder: 图表保存文件夹路径（如果为None则不保存到文件）
            show: 是否显示图表
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, skipping plot")
            return

        total_with_boot = self.total_latency + self.total_boot_latency
        if total_with_boot == 0:
            print("No data to plot")
            return

        # 准备数据
        op_types = []
        latencies = []
        boot_latencies = []

        for op_type, stats in sorted(self.op_stats.items(), key=lambda x: x[1]["latency"] + x[1]["boot_latency"], reverse=True):
            op_types.append(op_type)
            latencies.append(stats["latency"])
            boot_latencies.append(stats["boot_latency"])

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 饼图：各算子占比（含boot）
        ax1 = axes[0]
        sizes = [stats["latency"] + stats["boot_latency"] for stats in self.op_stats.values()]
        labels = list(self.op_stats.keys())
        # 添加单独的boot占比
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title("Latency Distribution by Operator (including boot)")

        # 柱状图：各算子耗时（分开显示操作和boot）
        ax2 = axes[1]
        x = range(len(op_types))
        width = 0.35
        bars1 = ax2.bar([i - width/2 for i in x], latencies, width, label='Operation', color='steelblue')
        bars2 = ax2.bar([i + width/2 for i in x], boot_latencies, width, label='Boot', color='coral')
        ax2.set_xlabel('Operator Type')
        ax2.set_ylabel('Latency')
        ax2.set_title('Latency by Operator Type')
        ax2.set_xticks(x)
        ax2.set_xticklabels(op_types, rotation=45, ha='right')
        ax2.legend()

        plt.tight_layout()

        if plot_folder:
            # 生成唯一文件名
            model_name = type(self.model).__name__
            save_path = generate_unique_filename(model_name, "png", plot_folder)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        if show:
            plt.show()

    def get_per_node_boot_info(self) -> List[Dict]:
        """获取每个节点的boot信息"""
        result = []
        for node, meta in self.node_meta_list.items():
            if meta.is_fused:
                continue
            result.append({
                "name": node.name,
                "op_type": meta.op_type,
                "in_depth": meta.in_depth,
                "out_depth": meta.out_depth,
                "out_ct": meta.out_ct,
                "boot_count": meta.boot_count,
                "boot_latency": meta.boot_latency,
            })
        return result


def exp_statistics(model: nn.Module):
    """实验函数：打印模型的FX图结构"""
    model = model.eval()
    traced = fx.symbolic_trace(model)

    dummy_input = torch.randn(1, 3, 224, 224)
    ShapeProp(traced).propagate(dummy_input)

    for node in traced.graph.nodes:
        tensor_meta = node.meta.get("tensor_meta")
        output_shape = tensor_meta.shape if tensor_meta is not None else "Unknown"
        if node.op == "call_module":
            module = traced.get_submodule(str(node.target))
            print(f"Node name: {node.name} Shape: {output_shape} ", end="")
            if isinstance(module, nn.Conv2d):
                print(f"Conv2d {module.in_channels} {module.out_channels} {module.kernel_size}")
            elif isinstance(module, nn.Linear):
                print(f"Linear {module.in_features} {module.out_features}")
            elif isinstance(module, nn.ReLU):
                print("ReLU")
            elif isinstance(module, nn.BatchNorm2d):
                print(f"BatchNorm2d {module.num_features}")
            elif isinstance(module, nn.MaxPool2d):
                print(f"MaxPool2d {module.kernel_size}")
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                print(f"AdaptiveAvgPool2d {module.output_size}")
            elif isinstance(module, nn.Dropout):
                print(f"Dropout {module.p}")
            elif isinstance(module, nn.Identity):
                print("Identity")
            else:
                print(f"Unknown module type: {type(module)}")
        elif node.op == "call_function":
            if node.target == torch.add or node.target == operator.add:
                print(f"Add {node.args[0]} {node.args[1]}")
            elif node.target == torch.flatten:
                print(f"Flatten {node.args[0]}")
            else:
                print(f"Unknown function: {node.target}")
        else:
            print(f"Node op: {node.op}")


def analyze_model(model: nn.Module, model_name: str | None = None,
                  output_folder: str | None = None, plot_folder: str | None = None,
                  input_shape: Tuple[int, ...] = (1, 3, 224, 224)):
    """分析模型的FHE统计信息

    Args:
        model: 要分析的模型
        model_name: 模型名称（用于显示）
        output_folder: 统计结果输出文件夹路径（文件名将自动生成）
        plot_folder: 图表保存文件夹路径（文件名将自动生成）
        input_shape: 输入张量形状

    Returns:
        FheInfo: 统计信息对象
    """
    if model_name:
        print(f"\nAnalyzing model: {model_name}")

    fhe_info = FheInfo(model, input_shape)
    fhe_info.run_statistics()
    fhe_info.print_statistics(output_folder)

    if plot_folder:
        fhe_info.plot_statistics(plot_folder=plot_folder, show=False)

    return fhe_info


if __name__ == "__main__":
    # 测试 ResNet18
    print("\n" + "="*50)
    print("Testing ResNet18")
    print("="*50)
    model = torchvision.models.resnet18()
    fhe_info = analyze_model(model, "ResNet18", output_folder=".", plot_folder=".")

    # 测试更多模型
    models_to_test = [
        ("ResNet34", torchvision.models.resnet34),
        ("ResNet50", torchvision.models.resnet50),
        ("VGG16", torchvision.models.vgg16),
        ("MobileNetV2", torchvision.models.mobilenet_v2),
    ]

    # 尝试 EfficientNet（可能需要较新版本的torchvision）
    try:
        models_to_test.append(("EfficientNet_B0", torchvision.models.efficientnet_b0))
    except AttributeError:
        print("EfficientNet not available in this torchvision version")

    for name, model_fn in models_to_test:
        print("\n" + "="*50)
        print(f"Testing {name}")
        print("="*50)
        try:
            model = model_fn()
            analyze_model(model, name, output_folder="results", plot_folder="results")
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
