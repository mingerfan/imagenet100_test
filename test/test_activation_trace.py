#!/usr/bin/env python3
"""测试自定义激活函数在 FX trace 中是否保持为单个节点"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, 'fhe_statistics')
sys.path.insert(0, 'models')

from models.gate_net_cmp.block_def import LearnableSwish, LearnableRelu, StablePoly7, Relu, Swish
from fhe_statistics.statistics_fn import FheInfo


class SimpleModelWithActivations(nn.Module):
    """包含各种自定义激活函数的简单测试模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.act1 = LearnableSwish()

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.act2 = Swish()

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.act3 = LearnableRelu()

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.act4 = StablePoly7()

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.act5 = Relu()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.act5(self.conv5(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def test_activation_tracing():
    """测试激活函数是否被正确trace为单个模块"""
    print("=" * 80)
    print("测试自定义激活函数的 FX Trace 行为")
    print("=" * 80)

    # 创建测试模型
    model = SimpleModelWithActivations()

    # 创建 FheInfo 并进行 trace
    print("\n创建 FheInfo 并 trace 模型...")
    fhe_info = FheInfo(model, input_shape=(1, 3, 32, 32), model_name="TestModel")

    # 打印 FX 图
    print("\n" + "=" * 80)
    print("FX Graph 结构:")
    print("=" * 80)
    for node in fhe_info.traced.graph.nodes:
        if node.op == "call_module":
            module = fhe_info.traced.get_submodule(str(node.target))
            module_type = type(module).__name__
            print(f"Node: {node.name:30s} | Op: {node.op:15s} | Module: {module_type:25s} | Target: {node.target}")
        else:
            print(f"Node: {node.name:30s} | Op: {node.op:15s}")

    # 检查激活函数是否被正确识别
    print("\n" + "=" * 80)
    print("检查激活函数节点:")
    print("=" * 80)

    activation_modules = {
        'LearnableSwish': 0,
        'Swish': 0,
        'LearnableRelu': 0,
        'StablePoly7': 0,
        'Relu': 0
    }

    for node in fhe_info.traced.graph.nodes:
        if node.op == "call_module":
            module = fhe_info.traced.get_submodule(str(node.target))
            module_type = type(module).__name__
            if module_type in activation_modules:
                activation_modules[module_type] += 1
                print(f"✓ 找到激活函数节点: {node.name:30s} | 类型: {module_type}")

    # 运行统计
    print("\n" + "=" * 80)
    print("运行 FHE 统计:")
    print("=" * 80)
    fhe_info.run_statistics()

    # 打印统计结果
    fhe_info.print_statistics()

    # 验证结果
    print("\n" + "=" * 80)
    print("验证结果:")
    print("=" * 80)

    success = True
    for act_type, count in activation_modules.items():
        expected = 1  # 每种激活函数应该出现1次
        if count == expected:
            print(f"✓ {act_type}: {count} 个节点 (预期 {expected})")
        else:
            print(f"✗ {act_type}: {count} 个节点 (预期 {expected})")
            success = False

    # 检查是否有非预期的细粒度操作（如 sigmoid, mul 等）
    print("\n检查是否存在细粒度操作节点（不应该存在）:")
    fine_grained_ops = ['sigmoid', 'mul', 'maximum']
    found_fine_grained = False

    for node in fhe_info.traced.graph.nodes:
        if node.op == "call_function":
            func_name = str(node.target)
            for fg_op in fine_grained_ops:
                if fg_op in func_name.lower():
                    print(f"✗ 发现细粒度操作: {node.name} ({func_name})")
                    found_fine_grained = True
                    success = False

    if not found_fine_grained:
        print("✓ 未发现细粒度操作节点（说明激活函数未被拆分）")

    print("\n" + "=" * 80)
    if success:
        print("测试通过! 所有激活函数都保持为单个节点。")
    else:
        print("测试失败! 某些激活函数被拆分或计数不正确。")
    print("=" * 80)

    return success


if __name__ == "__main__":
    test_activation_tracing()
