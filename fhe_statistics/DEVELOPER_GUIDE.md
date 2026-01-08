# FHE Statistics 重构开发者指南

## 快速开始

### 使用激活函数配置表

**之前**：需要定义 7 个几乎相同的方法
```python
def relu_statistics(self, node: Node):
    node_meta, in_shape, out_shape, in_ct, out_ct = self._init_node_meta(node, out_depth_delta=15)
    node_meta.mul_both = in_ct * 33
    node_meta.mul_single = in_ct * 33
    self._set_rescale(node_meta, include_single=True)
    self._finalize_node(node, node_meta, "relu")

def sigmoid_statistics(self, node: Node):
    # ...几乎相同的代码
```

**现在**：统一处理所有激活函数
```python
from activation_configs import get_activation_config

def activation_statistics(self, node: Node, activation_type: str):
    config = get_activation_config(activation_type)
    node_meta, _, _, in_ct, _ = self._init_node_meta(node, out_depth_delta=config['depth_delta'])
    node_meta.mul_both = in_ct * config['mul_both_factor']
    node_meta.mul_single = in_ct * config['mul_single_factor']
    self._set_rescale(node_meta, include_single=True)
    self._finalize_node(node, node_meta, activation_type)
```

**添加新的激活函数**：只需在 `activation_configs.py` 中添加一行配置
```python
ACTIVATION_CONFIGS['gelu'] = {
    'depth_delta': 15,
    'mul_both_factor': 35,
    'mul_single_factor': 35,
}
```

---

### 使用 FLOPs 计算器

**之前**：FLOPs 计算逻辑散落在三处
```python
# 第一处：_estimate_flops()
per_output_flops = kernel_h * kernel_w * (in_channels // groups)
total_flops = batch * out_h * out_w * out_channels * per_output_flops

# 第二处：get_shallow_layer_metrics()
# ...相同的代码重复
```

**现在**：使用统一的计算器
```python
from flops_calculator import FLOPsCalculator

# 计算 Conv2d FLOPs
flops = FLOPsCalculator.calc_conv2d_flops(
    in_channels=64,
    out_channels=128,
    kernel_size=(3, 3),
    output_shape=output.shape,
    groups=1
)

# 计算 Linear FLOPs
flops = FLOPsCalculator.calc_linear_flops(in_features=512, out_features=1000)
```

**优势**：
- 单一真实来源（SSOT）
- 易于测试和验证
- 支持一致的优化和改进

---

### 使用深度 Binning 工具

**之前**：binning 逻辑在三处重复
```python
# 第一处：get_depth_histogram_data()
bins = defaultdict(list)
for node, meta in self.node_meta_list.items():
    bin_idx = meta.out_depth // bin_size
    bins[bin_idx].append(...)

# 第二处：get_depth_flops_distribution()
# ...相同的逻辑重复
```

**现在**：使用 DepthBinner 工具
```python
from depth_binning import DepthBinner, DepthMetricsCollector

# 方式 1：简单的 binning
binner = DepthBinner(bin_size=1)
for node, meta in self.node_meta_list.items():
    binner.add_item(meta.out_depth, node)

bins = binner.get_bins()  # Dict[int, List]
bin_label = binner.get_bin_label(0)  # "0-0"

# 方式 2：同时收集多种度量
collector = DepthMetricsCollector(bin_size=1)
for node, meta in self.node_meta_list.items():
    collector.add_node_metrics(
        depth=meta.out_depth,
        latency=meta.latency,
        flops=...,
        parameters=...
    )

labels, latencies, flops, parameters = collector.get_metrics_as_lists()
# 直接用于 matplotlib 绘图
```

---

### 使用操作处理器注册表

**之前**：长的 if-elif 链
```python
def _handle_module(self, node: Node):
    module = self.traced.get_submodule(str(node.target))

    if isinstance(module, nn.Conv2d):
        self.conv_statistics(node)
    elif isinstance(module, nn.Linear):
        self.linear_statistics(node)
    elif isinstance(module, nn.ReLU):
        self.relu_statistics(node)
    # ...15+ 个 elif
    else:
        raise ValueError(f"未知模块{type(module).__name__}")
```

**现在**：使用注册表查询
```python
def _handle_module(self, node: Node):
    module = self.traced.get_submodule(str(node.target))

    handler_name = self.op_registry.get_module_handler(module)
    if handler_name:
        handler = getattr(self, handler_name, None)
        if handler and callable(handler):
            handler(node)
    else:
        self.pass_through_statistics(node, f"unknown_module_{type(module).__name__}")
```

**添加自定义操作处理**：
```python
# 方式 1：在初始化时注册
self.op_registry.register_module(CustomActivation, 'custom_activation_statistics')

# 方式 2：添加对应的方法
def custom_activation_statistics(self, node: Node):
    # ...实现
    pass
```

---

## 架构设计

### 模块依赖关系

```
statistics_fn.py (主文件)
    ├── 导入 → activation_configs.py
    ├── 导入 → flops_calculator.py
    ├── 导入 → depth_binning.py
    └── 导入 → operation_registry.py
```

### 类职责划分

| 类/模块 | 职责 | 文件 |
|--------|------|------|
| `ACTIVATION_CONFIGS` | 激活函数参数配置 | activation_configs.py |
| `FLOPsCalculator` | 各类算子的 FLOPs 计算 | flops_calculator.py |
| `DepthBinner` | 按深度分组工具 | depth_binning.py |
| `DepthMetricsCollector` | 按深度统计多种度量 | depth_binning.py |
| `OperationHandlerRegistry` | 操作处理器映射管理 | operation_registry.py |
| `FheInfo` | 主分析引擎 | statistics_fn.py |

---

## 扩展指南

### 添加新的激活函数

1. 在 `activation_configs.py` 中添加配置：
```python
ACTIVATION_CONFIGS['new_activation'] = {
    'depth_delta': 15,
    'mul_both_factor': 30,
    'mul_single_factor': 30,
}
```

2. 在 `operation_registry.py` 中注册：
```python
self._module_handlers[NewActivationModule] = 'activation_statistics'
```

3. 如果需要特殊处理，添加专用方法（可选）

### 添加新的算子

1. 实现计算方法（例如在 `FLOPsCalculator` 中）：
```python
@staticmethod
def calc_new_op_flops(...):
    # 实现计算逻辑
    pass
```

2. 在 `FheInfo` 中实现统计方法：
```python
def new_op_statistics(self, node: Node):
    # 实现
    pass
```

3. 在 `operation_registry.py` 中注册：
```python
self._module_handlers[NewOpModule] = 'new_op_statistics'
```

### 添加新的度量维度

使用 `DepthMetricsCollector`：
```python
collector = DepthMetricsCollector(bin_size=1)
for node, meta in self.node_meta_list.items():
    collector.add_node_metrics(
        depth=meta.out_depth,
        latency=meta.latency,
        flops=compute_flops(meta),
        parameters=compute_params(meta),
        # 可以添加更多自定义指标...
    )
```

---

## 测试用例示例

### 测试激活函数配置
```python
def test_activation_config():
    config = get_activation_config('relu')
    assert config['depth_delta'] == 15
    assert config['mul_both_factor'] == 33

    # 验证所有激活函数都有配置
    for act_type in ['relu', 'sigmoid', 'swish', 'poly7']:
        assert get_activation_config(act_type) is not None
```

### 测试 FLOPs 计算
```python
def test_flops_calculator():
    flops = FLOPsCalculator.calc_conv2d_flops(
        in_channels=3,
        out_channels=64,
        kernel_size=(3, 3),
        output_shape=(1, 64, 224, 224)
    )
    # Conv2d: (224*224) * 64 * (3*3*3) = ~289M
    assert flops > 0
```

### 测试 Binning
```python
def test_depth_binner():
    binner = DepthBinner(bin_size=5)

    # 添加不同深度的项
    binner.add_item(depth=3, item="node1")
    binner.add_item(depth=7, item="node2")
    binner.add_item(depth=15, item="node3")

    # 验证分组
    assert len(binner.get_bins()) == 3
    assert binner.get_bin_index(3) == 0
    assert binner.get_bin_index(7) == 1
    assert binner.get_bin_index(15) == 3
```

---

## 常见问题解答

### Q: 为什么保留原来的激活函数方法？
A: 为了向后兼容。现有代码可以继续调用 `relu_statistics()`，它只是简单地委托给统一的 `activation_statistics()` 方法。

### Q: 如何处理不在注册表中的操作？
A: 它们会被标记为 "unknown_operation"，通过 `pass_through_statistics()` 记录，确保不会中断分析流程。

### Q: 是否可以动态添加新的操作处理器？
A: 可以！使用 `register_module()`, `register_function()`, `register_method()` 方法。

### Q: 重构是否影响现有的统计结果？
A: 不影响。重构仅改进代码结构，所有计算逻辑保持不变。

---

## 维护建议

1. **定期检查**：如果添加新操作，务必更新注册表
2. **配置同步**：激活函数配置应与实际计算保持同步
3. **文档更新**：在 `ACTIVATION_CONFIGS` 中添加详细的参数说明
4. **单元测试**：为新增的计算方法编写测试用例
