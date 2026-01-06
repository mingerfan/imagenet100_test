from networkx import in_degree_centrality
from setuptools.tests.config.test_apply_pyprojecttoml import core_metadata
from mpmath.tests.test_identify import ceil
from sympy.tensor.array.expressions.array_expressions import get_shape
import operator

import torch
import torchvision
from torch import fx, nn
from torch.fx.node import Node
from torch.fx.passes.shape_prop import ShapeProp
from typing import Dict, Tuple
from torch.fx.passes.shape_prop import TensorMetadata

import math

def next_power_of_2_float(n):
    if n <= 0:
         return 1
    # log2 算出指数，ceil 向上取整，再用 pow 算回 2 的幂
    return 2 ** math.ceil(math.log2(n))

class NodeMeta:
    def __init__(self):
        self.rotation = 0
        self.mul_single = 0
        self.rescale = 0
        self.mul_both = 0
        self.in_depth = -1
        self.out_depth = -1
        self.in_ct = -1
        self.out_ct = -1

class FheInfo:
    def __init__(self, model: nn.Module):
        self.rotation: int = 0
        self.mul_single: int = 0
        self.mul_both: int = 0
        self.boot: int = 0

        self.rotation_cost = 200
        self.rescale_cost = 50
        self.mul_single_cost = 9.5
        self.mul_double_cost = 253
        self.boot_cost = 98136

        self.slots_num = 32768

        self.model = model
        self.traced = fx.symbolic_trace(model)
        dummy_input = torch.randn(1, 3, 224, 224)
        ShapeProp(self.traced).propagate(dummy_input)

        self.node_meta_list: Dict[Node, NodeMeta] = []

    def get_in_depth(self, node: Node) -> int:
        inputs = node.all_input_nodes
        cur_depth_max = 0
        for in_node in inputs:
            in_meta = self.node_meta_list[in_node]
            assert in_meta.out_depth != -1, "Invalid in node depth out"

            cur_depth_max = max(cur_depth_max, in_meta.out_depth)
        return cur_depth_max
      
    def get_in_shape(self, node: Node) -> torch.Size:
        input = node.all_input_nodes[0]
        meta = input.meta["tensor_meta"]

        input_shape = meta.shape

        return input_shape
    
    def calc_ct(self, shape: torch.Size) -> int:
        # 为了通用性，我们假设槽位能够被充分利用
        resovle_prod = shape[2] * shape[3]
        if resovle_prod > self.slots_num:
            min_ct_per_feature_map = next_power_of_2_float(resovle_prod / self.slots_num) # 考虑到数据在内部往往对称放置
            ct_num = min_ct_per_feature_map * shape[1]
        else:
            ct_num = math.ceil(shape[1] / math.floor(self.slots_num / resovle_prod))

        return ct_num

    def calc_channel_per_ct(self, shape: torch.Size) -> int:
        return math.floor(self.slots_num / (shape[2] * shape[3]))

    def get_tensor_meta(self, node: Node) -> TensorMetadata:
        tensor_meta = node.meta["tensor_meta"]
        return tensor_meta
            

    def conv_statistics(self, node: Node):
        tensor_meta = self.get_tensor_meta(node)
        output_shape = tensor_meta.shape
        module = self.traced.get_submodule(str(node.target))
        # in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size
        node_meta = NodeMeta()

        in_depth = self.get_in_depth(node)
        node_meta.in_depth = in_depth
        # 密文明文乘法消耗两个深度，一个是在卷积核的乘法上，一个是在输出结果的mask上
        node_meta.out_depth = in_depth + 2

        # 我们假设密文能够被充分利用
        in_shape = self.get_in_shape(node)
        in_ct = self.calc_ct(in_shape)
        out_ct = self.calc_ct(output_shape)
        
        ch_per_ct = self.calc_channel_per_ct(in_shape)
        out_ch_per_ct = self.calc_channel_per_ct(output_shape)

        # 卷积核与特征图相乘时采用优化的版本，会增加密文明文乘法数量 (gazelle卷积)
        kernel_ratio = kernel_size[0] * kernel_size[0]

        # 每一个in_ct都要与优化的卷积核心乘ratio次，然后再相加，再旋转求和，重复out_channels次，最后这些都要合并到新的ct中
        node_meta.mul_single = out_channels * in_ct * kernel_ratio

        # 旋转相加通道求和+旋转相加得到输出的ct
        node_meta.rotation = math.ceil(math.log2(ch_per_ct)) + out_ch_per_ct * out_ct

        node_meta.mul_both = 0

        node_meta.rescale = node_meta.mul_both + node_meta.mul_single # 每一个乘法后面都需要加rescale

        node_meta.in_ct = in_ct
        node_meta.out_ct = out_ct
        
        self.node_meta_list[node] = node_meta

    def relu_statistics(self, node: Node):
        # tensor_meta = self.get_tensor_meta(node)
        # output_shape = tensor_meta.shape
        # module = self.traced.get_submodule(str(node.target))

        in_shape = self.get_in_shape(node)

        in_ct = self.calc_ct(in_shape)
        out_ct = in_ct

        node_meta = NodeMeta()

        node_meta.mul_both = in_ct * 15 # sign默认14个乘法深度，乘x再多一个乘法深度
        node_meta.in_depth = self.get_in_depth(node)
        node_meta.out_depth = self.get_in_depth(node)

        node_meta.in_ct = in_ct
        node_meta.out_ct = out_ct
        
        self.node_meta_list[node] = node_meta

    def maxpool_statistics(self, node: Node):
        # max的实现为(a+b)/2 + sign(a-b)/2
        in_shape = self.get_in_shape(node)
        tensor_meta = self.get_tensor_meta(node)
        out_shape = tensor_meta.shape
        node_meta = NodeMeta()
        
        in_ct = self.calc_ct(in_shape)
        out_ct = self.calc_ct(out_shape)

        # 由于不同实现的旋转次数不一样，所以我们按照4合1的2次旋转估计
        node_meta.rotation = in_ct * 2

        # abcd得到最大值最少需要比较2次，而且是树状的比较，因此深度为(14+1) + (14+1) = 30
        in_depth = self.get_in_depth(node)
        out_depth = in_depth + 30
        node_meta.in_depth = in_depth
        node_meta.out_depth = out_depth


        # 由于所有通道都要max一下，所以我们认为最少需要这么多次
        node_meta.mul_both = 28 * in_ct

        node_meta.mul_single = 2 * in_ct

        node_meta.rescale = node_meta.mul_both + node_meta.mul_single

        node_meta.in_ct = in_ct
        node_meta.out_ct = out_ct

        self.node_meta_list[node] = node_meta

    def avepool_statistics(self, node: Node):
        in_shape = self.get_in_shape(node) 
        tensor_meta = self.get_tensor_meta(node)
        out_shape = tensor_meta.shape
        node_meta = NodeMeta()

        in_ct = self.calc_ct(in_shape)
        out_ct = self.calc_ct(out_shape)

        node_meta.rotation = in_ct * 2

        in_depth = self.get_in_depth(node)
        # 求和乘以0.25
        out_depth = in_depth + 1

        node_meta.in_depth = in_depth
        node_meta.out_depth = out_depth

        node_meta.mul_single = 1 * in_ct

        node_meta.rescale = node_meta.mul_single

        node_meta.in_ct = in_ct
        node_meta.out_ct = out_ct

        self.node_meta_list[node] = node_meta

    def adaptiveavepool2d_statistcs(self, node: Node):
        in_shape = self.get_in_shape(node)
        in_channel = in_shape[1]
        tensor_meta = self.get_tensor_meta(node)
        out_shape = tensor_meta.shape
        
        in_shape  = self.get_in_shape(node)
        in_ct = self.calc_ct(in_shape)
        out_ct = 1
        
        feature_map_size = in_shape[2] * in_shape[3]

        log2_map_size =  math.log2(feature_map_size)
        
        node_meta = NodeMeta()

        # 旋转求和，重复in_ct次，添加一个0.5*channel_size作为重整化的旋转补偿
        node_meta.rotation = log2_map_size * in_ct + int(0.5 * in_shape[1])
        
        # 需要乘mask，所以需要消耗1个乘法深度
        in_depth = self.get_in_depth(node)
        out_depth = in_depth + 1

        node_meta.in_depth = in_depth
        node_meta.out_depth = out_depth

        node_meta.mul_single = 1
        node_meta.mul_both = 0
        node_meta.rescale = node_meta.mul_single

        node_meta.in_ct = in_ct
        node_meta.out_ct = out_ct

def exp_statistics(model: nn.Module):
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
                print(
                    f"Conv2d {module.in_channels} {module.out_channels} {module.kernel_size}"
                )
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


if __name__ == "__main__":
    model = torchvision.models.resnet18()
    exp_statistics(model)
