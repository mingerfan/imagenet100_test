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

def next_power_of_2_float(n):
    if n <= 0:
        return 1
    return 2 ** math.ceil(math.log2(n))


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
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 224, 224), model_name: Optional[str] = None):
        # 成本参数
        self.rotation_cost = 180
        self.rescale_cost = 40
        self.mul_single_cost = 9.5
        self.mul_double_cost = 253
        self.boot_cost = 98136
        self.level = 10

        self.slots_num = 32768
        self.input_shape = input_shape

        self.model = model.eval()
        self.model_name = model_name if model_name else type(model).__name__
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
        boot_triggered = max(0, (node_meta.out_depth - node_meta.in_depth)//self.level)
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
        in_channels = module.in_channels
        kernel_size = module.kernel_size
        groups = module.groups # 需要区分深度卷积，否则对MobileNet不公平

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
        if groups == 1:
            node_meta.mul_single = out_channels * in_ct * kernel_ratio
            node_meta.rotation = math.ceil(math.log2(max(1, ch_per_ct))) + min(out_ch_per_ct, out_channels) * out_ct
            node_meta.mul_both = 0
        elif groups == in_channels:
            # 深度卷积
            node_meta.mul_single = in_ct * kernel_ratio
            node_meta.rotation = 0 # 深度卷积没旋转
            node_meta.mul_both = 0
        else:
            raise ValueError("groups 参数不正确")

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
        self._finalize_node(node, node_meta, "swish")

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
        self._finalize_node(node, node_meta, "sigmoid")


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
        model_name = self.model_name
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
    
    def print_detailed_statistics(self, output_folder: Optional[str] = None):
        """打印详细的逐个算子统计信息（包括拓扑排序和每个算子的详细耗时）
        
        Args:
            output_folder: 输出文件夹路径（如果为None则不保存到文件）
        """
        lines = []
        model_name = self.model_name
        lines.append("=" * 130)
        lines.append(f"Detailed FHE Statistics for {model_name}")
        lines.append("=" * 130)
        
        # 第一部分：拓扑排序
        lines.append("\n" + "=" * 130)
        lines.append("TOPOLOGICAL ORDER")
        lines.append("=" * 130)
        topo_header = f"{'Topo Order':>10} {'Node ID':>15} {'Node Name':<30} {'Op Type':<20} {'In Depth':>10} {'Out Depth':>10} {'In CT':>8} {'Out CT':>8}"
        lines.append(topo_header)
        lines.append("-" * 130)
        
        # 获取拓扑排序（按traced.graph.nodes的顺序）
        topo_order = 0
        for node in self.traced.graph.nodes:
            if node not in self.node_meta_list:
                continue
            meta = self.node_meta_list[node]
            node_id = id(node)
            node_name = node.name
            op_type = meta.op_type
            in_depth = meta.in_depth
            out_depth = meta.out_depth
            in_ct = meta.in_ct
            out_ct = meta.out_ct
            
            line = f"{topo_order:>10} {node_id:>15} {node_name:<30} {op_type:<20} {in_depth:>10} {out_depth:>10} {in_ct:>8} {out_ct:>8}"
            lines.append(line)
            topo_order += 1
        
        # 第二部分：每个算子的详细统计
        lines.append("\n" + "=" * 130)
        lines.append("DETAILED OPERATOR STATISTICS")
        lines.append("=" * 130)
        detail_header = f"{'Topo Order':>10} {'Node Name':<30} {'Op Type':<20} {'Rotation':>10} {'MulSingle':>10} {'MulBoth':>10} {'Rescale':>10} {'Latency':>12} {'Boot Count':>10} {'Boot Latency':>12} {'Total':>12}"
        lines.append(detail_header)
        lines.append("-" * 130)
        
        # 获取拓扑排序
        topo_order = 0
        total_latency = 0
        total_boot = 0
        
        for node in self.traced.graph.nodes:
            if node not in self.node_meta_list:
                continue
            meta = self.node_meta_list[node]
            
            # 跳过fused算子
            if meta.is_fused:
                topo_order += 1
                continue
            
            node_name = node.name
            op_type = meta.op_type
            rotation = meta.rotation
            mul_single = meta.mul_single
            mul_both = meta.mul_both
            rescale = meta.rescale
            latency = meta.latency
            boot_count = meta.boot_count
            boot_latency = meta.boot_latency
            total = latency + boot_latency
            
            total_latency += latency
            total_boot += boot_latency
            
            line = f"{topo_order:>10} {node_name:<30} {op_type:<20} {rotation:>10} {mul_single:>10} {mul_both:>10} {rescale:>10} {latency:>12.2f} {boot_count:>10} {boot_latency:>12.2f} {total:>12.2f}"
            lines.append(line)
            topo_order += 1
        
        # 第三部分：汇总统计
        lines.append("\n" + "=" * 130)
        lines.append("SUMMARY")
        lines.append("=" * 130)
        lines.append(f"Total Operation Latency (without boot): {total_latency:.2f}")
        lines.append(f"Total Boot Latency: {total_boot:.2f}")
        lines.append(f"Total Latency (with boot): {total_latency + total_boot:.2f}")
        lines.append("=" * 130)

        output = "\n".join(lines)
        print(output)

        if output_folder:
            # 生成唯一文件名
            output_file = generate_unique_filename(f"{model_name}_detailed", "txt", output_folder)
            with open(output_file, "w") as f:
                f.write(output)
            print(f"\nDetailed results saved to {output_file}")

    def plot_statistics(self, plot_folder: Optional[str] = None, show: bool = True, 
                       plot_types: Optional[List[str]] = None):
        """绘制统计结果并保存到文件
        
        Args:
            plot_folder: 图表保存文件夹路径（如果为None则不保存到文件）
            show: 是否显示图表
            plot_types: 要绘制的图表类型列表，可选值：
                - 'basic': 基础图表（柱状图）
                - 'operator_stack': 算子堆栈图
                - 'depth_histogram': 深度直方图
                - 'network_comparison': 网络横向比较图（需要传入network_data）
                - 'all': 所有基础图表（默认）
        """
        if plot_types is None:
            plot_types = ['all']
        
        # 如果是'all'，绘制所有图表
        if 'all' in plot_types:
            self.plot_operator_stack(plot_folder=plot_folder, show=show)
            self.plot_depth_histogram(bin_size=10, plot_folder=plot_folder, show=show)
            self._plot_basic_statistics(plot_folder=plot_folder, show=show)
        else:
            # 根据指定类型绘制图表
            if 'basic' in plot_types:
                self._plot_basic_statistics(plot_folder=plot_folder, show=show)
            if 'operator_stack' in plot_types:
                self.plot_operator_stack(plot_folder=plot_folder, show=show)
            if 'depth_histogram' in plot_types:
                self.plot_depth_histogram(bin_size=10, plot_folder=plot_folder, show=show)

    def _plot_basic_statistics(self, plot_folder: Optional[str] = None, show: bool = True):
        """绘制基础统计图表（柱状图）"""
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

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # 柱状图：各算子耗时（分开显示操作和boot）
        x = range(len(op_types))
        width = 0.35
        bars1 = ax.bar([i - width/2 for i in x], latencies, width, label='Operation', color='steelblue')
        bars2 = ax.bar([i + width/2 for i in x], boot_latencies, width, label='Boot', color='coral')
        ax.set_xlabel('Operator Type')
        ax.set_ylabel('Latency')
        ax.set_title('Latency by Operator Type')
        ax.set_xticks(x)
        ax.set_xticklabels(op_types, rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()

        if plot_folder:
            # 生成唯一文件名
            model_name = self.model_name
            save_path = generate_unique_filename(f"{model_name}_basic", "png", plot_folder)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Basic statistics plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

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

    # ========== 数据准备函数 ==========

    def get_operator_breakdown_data(self) -> Dict[str, Dict[str, float]]:
        """获取每个算子类型的操作分解数据（用于算子堆栈图）
        
        Returns:
            Dict结构: {
                "conv": {
                    "rotation": 1000.0,
                    "mul_single": 2000.0,
                    "mul_both": 500.0,
                    "rescale": 2500.0,
                    "total_latency": 6000.0,
                    "total_boot_latency": 1000.0,
                    "boot_by_source": {
                        "conv": 800.0,
                        "relu": 200.0,
                        ...
                    }
                },
                ...
            }
        """
        result = {}
        
        for op_type, stats in self.op_stats.items():
            # 获取该算子类型所有节点的详细数据
            rotation_total = stats["rotation"]
            mul_single_total = stats["mul_single"]
            mul_both_total = stats["mul_both"]
            rescale_total = stats["rescale"]
            total_latency = stats["latency"]
            total_boot = stats["boot_latency"]
            
            # 统计boot按来源算子类型
            boot_by_source = defaultdict(float)
            for node, meta in self.node_meta_list.items():
                if meta.op_type == op_type and not meta.is_fused:
                    if meta.boot_latency > 0:
                        boot_by_source[meta.op_type] += meta.boot_latency
            
            result[op_type] = {
                "rotation": rotation_total,
                "mul_single": mul_single_total,
                "mul_both": mul_both_total,
                "rescale": rescale_total,
                "total_latency": total_latency,
                "total_boot_latency": total_boot,
                "boot_by_source": dict(boot_by_source)
            }
        
        return result

    def get_depth_histogram_data(self, bin_size: int = 10, max_bins: Optional[int] = None) -> Dict:
        """获取按深度范围聚合的数据（用于深度直方图）
        
        Args:
            bin_size: 深度分箱大小（默认为10，对应level）
            max_bins: 最大bin数量（可选），如果深度太大，会自动调整bin_size以控制bin数量
        
        Returns:
            Dict结构: {
                "bins": ["0-10", "10-20", "20-30", ...],
                "op_data": {
                    "conv": [100.0, 200.0, 300.0, ...],
                    "relu": [50.0, 100.0, 150.0, ...],
                    ...
                },
                "boot": [10.0, 20.0, 30.0, ...]  # boot单独列出
            }
        """
        # 收集所有节点的深度信息
        depth_data = defaultdict(lambda: defaultdict(float))
        boot_data = defaultdict(float)
        all_depths = []
        
        for node, meta in self.node_meta_list.items():
            if meta.is_fused:
                continue
            
            # 使用out_depth作为该节点的深度
            depth = meta.out_depth
            
            # 计算该节点的总耗时（不含boot）
            node_latency = meta.latency
            
            # 记录数据
            depth_data[depth][meta.op_type] += node_latency
            boot_data[depth] += meta.boot_latency
            all_depths.append(depth)
        
        if not all_depths:
            return {
                "bins": [],
                "op_data": {},
                "boot": []
            }
        
        # 确定深度分箱
        min_depth = 0
        max_depth = max(all_depths)
        
        # 如果指定了max_bins，调整bin_size
        if max_bins is not None:
            estimated_bins = math.ceil(max_depth / bin_size) + 1
            if estimated_bins > max_bins:
                bin_size = math.ceil(max_depth / max_bins)
        
        num_bins = math.ceil(max_depth / bin_size) + 1
        
        # 创建bins
        bins = [f"{i*bin_size}-{(i+1)*bin_size}" for i in range(num_bins)]
        
        # 初始化结果结构
        op_types = set()
        for depth_dict in depth_data.values():
            op_types.update(depth_dict.keys())
        op_types = sorted(op_types)
        
        op_data = {op: [0.0] * num_bins for op in op_types}
        boot = [0.0] * num_bins
        
        # 按bin聚合数据
        for node, meta in self.node_meta_list.items():
            if meta.is_fused:
                continue
            
            depth = meta.out_depth
            bin_idx = depth // bin_size
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1
            
            op_data[meta.op_type][bin_idx] += meta.latency
            boot[bin_idx] += meta.boot_latency
        
        return {
            "bins": bins,
            "op_data": op_data,
            "boot": boot
        }

    def get_network_comparison_data(self) -> Dict[str, float]:
        """获取网络汇总数据（用于网络横向比较图）
        
        Returns:
            Dict结构: {
                "conv": 7000.0,  # latency + boot_latency
                "relu": 3000.0,
                "maxpool": 2000.0,
                ...
            }
        """
        result = {}
        
        for op_type, stats in self.op_stats.items():
            # 总耗时 = 操作耗时 + boot耗时
            result[op_type] = stats["latency"] + stats["boot_latency"]
        
        return result

    # ========== 可视化函数 ==========

    def plot_operator_stack(self, plot_folder: Optional[str] = None, show: bool = True):
        """绘制算子堆栈图：每个算子类型内部操作的堆叠 + boot按来源堆叠
        
        Args:
            plot_folder: 图表保存文件夹路径（如果为None则不保存到文件）
            show: 是否显示图表
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, skipping plot")
            return

        data = self.get_operator_breakdown_data()
        
        if not data:
            print("No data to plot")
            return

        # 按总耗时排序
        sorted_ops = sorted(data.items(), key=lambda x: x[1]["total_latency"], reverse=True)
        
        # 准备数据
        op_types = [op for op, _ in sorted_ops]
        rotation = [data[op]["rotation"] for op in op_types]
        mul_single = [data[op]["mul_single"] for op in op_types]
        mul_both = [data[op]["mul_both"] for op in op_types]
        rescale = [data[op]["rescale"] for op in op_types]
        boot_total = [data[op]["total_boot_latency"] for op in op_types]
        
        # 收集所有可能的boot来源
        all_boot_sources = set()
        for op, op_data in data.items():
            all_boot_sources.update(op_data["boot_by_source"].keys())
        all_boot_sources = sorted(all_boot_sources)
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # 左图：操作堆叠图
        x = range(len(op_types))
        width = 0.6
        
        # 堆叠操作类型
        bottom1 = [0] * len(op_types)
        for op_data, color, label in [
            (rotation, 'lightblue', 'Rotation'),
            (mul_single, 'steelblue', 'Mul Single'),
            (mul_both, 'navy', 'Mul Both'),
            (rescale, 'darkred', 'Rescale'),
        ]:
            ax1.bar(x, op_data, width, bottom=bottom1, label=label, color=color)
            bottom1 = [b + o for b, o in zip(bottom1, op_data)]
        
        ax1.set_xlabel('Operator Type')
        ax1.set_ylabel('Latency')
        ax1.set_title('Operator Internal Operation Stack')
        ax1.set_xticks(x)
        ax1.set_xticklabels(op_types, rotation=45, ha='right')
        ax1.legend()
        
        # 右图：Boot按来源堆叠图
        bottom2 = [0] * len(op_types)
        boot_colors = plt.cm.Set3(range(len(all_boot_sources)))
        
        for source, color in zip(all_boot_sources, boot_colors):
            boot_values = [data[op].get("boot_by_source", {}).get(source, 0) for op in op_types]
            ax2.bar(x, boot_values, width, bottom=bottom2, label=f'Boot from {source}', color=color)
            bottom2 = [b + o for b, o in zip(bottom2, boot_values)]
        
        ax2.set_xlabel('Operator Type')
        ax2.set_ylabel('Boot Latency')
        ax2.set_title('Boot Latency by Source')
        ax2.set_xticks(x)
        ax2.set_xticklabels(op_types, rotation=45, ha='right')
        ax2.legend()
        
        plt.suptitle(f'{self.model_name} - Operator Breakdown', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if plot_folder:
            model_name = self.model_name
            save_path = generate_unique_filename(f"{model_name}_operator_stack", "png", plot_folder)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Operator stack plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_depth_histogram(self, bin_size: int = 10, max_bins: Optional[int] = 30, plot_folder: Optional[str] = None, show: bool = True):
        """绘制深度直方图：横坐标为深度范围，纵坐标为堆叠的算子耗时 + boot单独列出
        
        Args:
            bin_size: 深度分箱大小（默认为10）
            max_bins: 最大bin数量（默认为30），如果深度太大，会自动调整bin_size以控制bin数量
            plot_folder: 图表保存文件夹路径（如果为None则不保存到文件）
            show: 是否显示图表
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, skipping plot")
            return

        data = self.get_depth_histogram_data(bin_size, max_bins)
        
        if not data["bins"]:
            print("No data to plot")
            return

        bins = data["bins"]
        op_data = data["op_data"]
        boot_data = data["boot"]
        
        # 获取所有算子类型并按总耗时排序
        op_totals = {}
        for op_type, values in op_data.items():
            op_totals[op_type] = sum(values)
        sorted_ops = sorted(op_totals.items(), key=lambda x: x[1], reverse=True)
        op_types = [op for op, _ in sorted_ops]
        
        # 创建子图
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        x = range(len(bins))
        width = 0.8
        
        # 堆叠各算子类型
        bottom = [0] * len(bins)
        colors = plt.cm.tab20(range(len(op_types)))
        
        for op_type, color in zip(op_types, colors):
            values = op_data[op_type]
            ax.bar(x, values, width, bottom=bottom, label=op_type, color=color)
            bottom = [b + v for b, v in zip(bottom, values)]
        
        # 添加boot（单独显示，不堆叠在算子中）
        # 为了更清晰，我们在每个bin上叠加显示boot
        boot_bars = ax.bar(x, boot_data, width, bottom=bottom, label='Boot', color='black', alpha=0.5, hatch='//')
        
        ax.set_xlabel(f'Depth Range (bin_size={bin_size})')
        ax.set_ylabel('Latency')
        ax.set_title(f'{self.model_name} - Latency by Depth Range')
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=45, ha='right')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()

        if plot_folder:
            model_name = self.model_name
            save_path = generate_unique_filename(f"{model_name}_depth_histogram", "png", plot_folder)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Depth histogram plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_network_comparison(network_data: Dict[str, Dict[str, float]], plot_folder: Optional[str] = None, show: bool = True):
        """绘制网络横向比较图：横坐标为不同网络，纵坐标为堆叠的算子耗时
        
        Args:
            network_data: 字典，结构为 {network_name: {op_type: latency, ...}}
            plot_folder: 图表保存文件夹路径（如果为None则不保存到文件）
            show: 是否显示图表
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, skipping plot")
            return

        if not network_data:
            print("No data to plot")
            return

        # 获取所有网络名称和算子类型
        network_names = list(network_data.keys())
        
        # 收集所有算子类型
        all_op_types = set()
        for net_data in network_data.values():
            all_op_types.update(net_data.keys())
        all_op_types = sorted(all_op_types)
        
        # 计算每个算子类型的总耗时（所有网络），用于排序
        op_totals = {}
        for op_type in all_op_types:
            op_totals[op_type] = sum(net_data.get(op_type, 0) for net_data in network_data.values())
        sorted_ops = sorted(op_totals.items(), key=lambda x: x[1], reverse=True)
        op_types = [op for op, _ in sorted_ops]
        
        # 创建图表
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        x = range(len(network_names))
        width = 0.8
        
        # 堆叠各算子类型
        bottom = [0] * len(network_names)
        colors = plt.cm.tab20(range(len(op_types)))
        
        for op_type, color in zip(op_types, colors):
            values = [network_data[net_name].get(op_type, 0) for net_name in network_names]
            ax.bar(x, values, width, bottom=bottom, label=op_type, color=color)
            bottom = [b + v for b, v in zip(bottom, values)]
        
        ax.set_xlabel('Network')
        ax.set_ylabel('Latency')
        ax.set_title('Network Comparison - Latency by Operator Type')
        ax.set_xticks(x)
        ax.set_xticklabels(network_names, rotation=45, ha='right')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()

        if plot_folder:
            save_path = generate_unique_filename("network_comparison", "png", plot_folder)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Network comparison plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


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
                  input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                  print_detailed: bool = True):
    """分析模型的FHE统计信息

    Args:
        model: 要分析的模型
        model_name: 模型名称（用于显示和文件命名）
        output_folder: 统计结果输出文件夹路径（文件名将自动生成）
        plot_folder: 图表保存文件夹路径（文件名将自动生成）
        input_shape: 输入张量形状
        print_detailed: 是否打印详细统计信息（包括拓扑排序和每个算子的详细耗时）

    Returns:
        FheInfo: 统计信息对象
    """
    if model_name:
        print(f"\nAnalyzing model: {model_name}")

    fhe_info = FheInfo(model, input_shape, model_name)
    fhe_info.run_statistics()
    
    # 打印汇总统计
    fhe_info.print_statistics(output_folder)
    
    # 打印详细统计（包括拓扑排序和每个算子的详细耗时）
    if print_detailed and output_folder:
        fhe_info.print_detailed_statistics(output_folder)
    elif print_detailed:
        fhe_info.print_detailed_statistics(None)

    if plot_folder:
        fhe_info.plot_statistics(plot_folder=plot_folder, show=False)

    return fhe_info


def compare_networks(models: List[Tuple[str, nn.Module]], plot_folder: str | None = None,
                     input_shape: Tuple[int, ...] = (1, 3, 224, 224)):
    """比较多个网络的FHE统计信息并绘制横向比较图
    
    Args:
        models: 模型列表，每个元素为 (model_name, model) 元组
        plot_folder: 图表保存文件夹路径（如果为None则不保存到文件）
        input_shape: 输入张量形状
    
    Returns:
        Dict: 包含所有网络比较数据的字典
    """
    network_data = {}
    
    for model_name, model in models:
        print(f"\n{'='*50}")
        print(f"Analyzing {model_name}")
        print(f"{'='*50}")
        
        fhe_info = FheInfo(model, input_shape, model_name)
        fhe_info.run_statistics()
        
        # 打印该网络的统计信息
        fhe_info.print_statistics()
        
        # 获取该网络的汇总数据
        network_data[model_name] = fhe_info.get_network_comparison_data()
    
    # 绘制网络横向比较图
    if plot_folder:
        FheInfo.plot_network_comparison(network_data, plot_folder=plot_folder, show=False)
    else:
        FheInfo.plot_network_comparison(network_data, plot_folder=None, show=True)
    
    return network_data


if __name__ == "__main__":
    # 测试更多模型
    models_to_test = [
        ("ResNet18", torchvision.models.resnet18),
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

    # 第一步：独立分析每个模型
    models_for_comparison = []
    for name, model_fn in models_to_test:
        print("\n" + "="*50)
        print(f"Testing {name}")
        print("="*50)
        try:
            model = model_fn()
            analyze_model(model, name, output_folder="results", plot_folder="results")
            # 将模型加入比较列表
            models_for_comparison.append((name, model))
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            import traceback
            traceback.print_exc()

    # 第二步：生成网络横向比较图
    if models_for_comparison:
        print("\n" + "="*50)
        print("Generating Network Comparison")
        print("="*50)
        try:
            compare_networks(models_for_comparison, plot_folder="results")
            print("\n" + "="*50)
            print("All analysis completed!")
            print("="*50)
        except Exception as e:
            print(f"Error generating comparison: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo models successfully analyzed for comparison.")
