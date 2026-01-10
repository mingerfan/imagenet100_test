# FHE Statistics 模块文档

本文档介绍 `fhe_statistics` 模块的功能、使用方法和重构设计。

## 目录

- [快速开始](#快速开始)
- [批量分析工具](#批量分析工具) ⭐ **新增**
- [模块概览](#模块概览)
- [开发者指南](#开发者指南)
- [重构说明](#重构说明)

---

## 快速开始

### 基本使用

```python
from fhe_statistics.statistics_fn import FheInfo

# 创建 FHE 统计对象
model = YourModel()
fhe_info = FheInfo(model, input_shape=(1, 3, 224, 224), model_name="YourModel")

# 运行统计
fhe_info.run_statistics()

# 打印结果
fhe_info.print_statistics(output_folder="results")

# 绘制图表
fhe_info.plot_statistics(plot_folder="results", show=False)
```

### 支持的功能

- ✅ **FHE 操作统计**: 计算每个算子的 rotation、multiplication、rescale 等操作
- ✅ **Boot 计算**: 自动计算 bootstrapping 触发次数和延迟
- ✅ **深度分析**: 追踪密文深度，分析深度分布
- ✅ **可视化**: 生成算子堆栈图、深度直方图、网络对比图等
- ✅ **自定义激活函数**: 支持自定义激活函数（不会被拆分）
- ✅ **批量分析**: 配置驱动的批量网络分析工具 ⭐ **新增**

---

## 批量分析工具

### 概述

批量分析工具（`batch_analyzer.py`）可以根据配置文件批量分析多个神经网络，支持：
- ✅ TorchVision预训练模型
- ✅ 自定义模型（从 `gate_net.py` 等加载）
- ✅ 从checkpoint加载的模型
- ✅ 不同输入分辨率的对比
- ✅ 自动生成综合比较报告

### 快速使用

#### 1. 列出所有可分析的模型

```bash
python fhe_statistics/batch_analyzer.py --list
```

#### 2. 分析所有启用的模型

```bash
python fhe_statistics/batch_analyzer.py
```

#### 3. 只分析特定模型

```bash
python fhe_statistics/batch_analyzer.py --models ResNet18 ResNet34 MobileNetV2
```

#### 4. 使用自定义配置文件

```bash
python fhe_statistics/batch_analyzer.py --config my_config.yaml
```

### 配置文件说明

配置文件位于 `fhe_statistics/batch_analysis_config.yaml`。

#### 全局设置

```yaml
global:
    output_folder: "fhe_statistics/results"  # 统计结果目录
    plot_folder: "fhe_statistics/plots"      # 图表输出目录
    default_input_shape: [1, 3, 224, 224]    # 默认输入形状
    print_detailed: true                     # 打印详细统计
    generate_plots: true                     # 生成图表
    generate_comparison: true                # 生成综合比较图
```

#### 添加模型

支持三种模型来源：

**方式1: TorchVision预训练模型**

```yaml
- name: "ResNet18"
  source: "torchvision"
  model_class: "resnet18"
  input_shape: [1, 3, 224, 224]
  enabled: true
```

**方式2: 自定义模型**

```yaml
- name: "ResNet18-SelfGated-Swish"
  source: "custom"
  module_path: "models.gate_net"
  model_class: "resnet18"
  params:
      block_type: "self_gated"
      activation_type: "swish"
      num_classes: 100
  input_shape: [1, 3, 224, 224]
  enabled: true
```

**方式3: 从checkpoint加载**

```yaml
- name: "MyModel-Trained"
  source: "checkpoint"
  module_path: "models.gate_net"
  model_class: "resnet18"
  checkpoint_path: "checkpoints/model.pth"
  params:
      num_classes: 100
  input_shape: [1, 3, 224, 224]
  enabled: true
```

### 使用场景

#### 场景1: 比较不同分辨率的影响

在配置文件中添加：

```yaml
models:
    - name: "ResNet18-96x96"
      source: "torchvision"
      model_class: "resnet18"
      input_shape: [1, 3, 96, 96]
      enabled: true

    - name: "ResNet18-128x128"
      source: "torchvision"
      model_class: "resnet18"
      input_shape: [1, 3, 128, 128]
      enabled: true

    - name: "ResNet18-224x224"
      source: "torchvision"
      model_class: "resnet18"
      input_shape: [1, 3, 224, 224]
      enabled: true
```

运行分析：

```bash
python fhe_statistics/batch_analyzer.py
```

#### 场景2: 横向比较多个预训练网络

在配置文件中启用多个模型：

```yaml
models:
    - name: "ResNet18"
      source: "torchvision"
      model_class: "resnet18"
      enabled: true

    - name: "ResNet50"
      source: "torchvision"
      model_class: "resnet50"
      enabled: true

    - name: "MobileNetV2"
      source: "torchvision"
      model_class: "mobilenet_v2"
      enabled: true

    - name: "EfficientNet_B0"
      source: "torchvision"
      model_class: "efficientnet_b0"
      enabled: true
      optional: true  # 如果加载失败不中断
```

#### 场景3: 比较训练前后的FHE性能

```yaml
models:
    - name: "ResNet18-Pretrained"
      source: "torchvision"
      model_class: "resnet18"
      input_shape: [1, 3, 224, 224]
      enabled: true

    - name: "ResNet18-MyTraining"
      source: "checkpoint"
      module_path: "models.gate_net"
      model_class: "resnet18"
      checkpoint_path: "checkpoints/my_trained_model.pth"
      params:
          num_classes: 100
      input_shape: [1, 3, 224, 224]
      enabled: true
```

### 输出说明

#### 1. 统计结果文件（`results/`）

- `{model_name}_{timestamp}.txt` - 汇总统计
- `{model_name}_detailed_{timestamp}.txt` - 详细统计（拓扑排序）
- `summary_report_{timestamp}.txt` - 所有模型汇总

#### 2. 可视化图表（`plots/`）

**单个模型图表**：
- `{model_name}_basic_{timestamp}.png` - 基础柱状图
- `{model_name}_operator_stack_{timestamp}.png` - 算子堆栈图
- `{model_name}_depth_histogram_{timestamp}.png` - 深度直方图

**多模型比较图表**：
- `network_comparison_{timestamp}.png` - 网络横向比较
- `network_comprehensive_comparison_{timestamp}.png` - 综合比较（6个子图）
- `network_grouped_comparison_{timestamp}.png` - 分组比较（归一化）

### 高级用法

#### 自定义FHE参数

在配置文件中覆盖默认参数：

```yaml
fhe_params:
    rotation_cost: 200
    rescale_cost: 50
    mul_single_cost: 10
    mul_double_cost: 300
    boot_cost: 100000
    level: 12
    slots_num: 32768
```

#### 自定义比较图

```yaml
comparison:
    plot_types:
        - "network_comparison"
        - "comprehensive_comparison"
        - "grouped_comparison"
    shallow_threshold: 0.2      # 浅层定义（前20%深度）
    depth_bin_size: 10          # 深度分箱大小
    max_depth_bins: 30          # 最大分箱数
```

---

## 模块概览

### 核心模块

```
fhe_statistics/
├── statistics_fn.py          # 主统计模块
├── activation_configs.py     # 激活函数配置表
├── operation_registry.py     # 操作处理器注册表
├── flops_calculator.py       # FLOPs 计算器
├── depth_binning.py          # 深度分箱工具
└── README.md                 # 本文档
```

### 模块职责

| 模块 | 职责 | 代码行数 |
|------|------|---------|
| `statistics_fn.py` | 主统计逻辑、FHE 成本计算、可视化 | ~2200 行 |
| `activation_configs.py` | 激活函数配置数据 | ~50 行 |
| `operation_registry.py` | 操作处理器映射（消除 if-elif 链） | ~100 行 |
| `flops_calculator.py` | FLOPs 计算逻辑 | ~150 行 |
| `depth_binning.py` | 深度相关工具函数 | ~100 行 |

---

## 开发者指南

### 1. 使用激活函数配置表

**之前的做法**（需要定义 7 个几乎相同的方法）:
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

**现在的做法**（统一处理）:
```python
from .activation_configs import get_activation_config

def activation_statistics(self, node: Node, activation_type: str):
    config = get_activation_config(activation_type)
    node_meta, _, _, in_ct, _ = self._init_node_meta(node, out_depth_delta=config['depth_delta'])
    node_meta.mul_both = in_ct * config['mul_both_factor']
    node_meta.mul_single = in_ct * config['mul_single_factor']
    self._set_rescale(node_meta, include_single=True)
    self._finalize_node(node, node_meta, activation_type)
```

**添加新的激活函数**（只需一行配置）:
```python
# 在 activation_configs.py 中添加
ACTIVATION_CONFIGS['gelu'] = {
    'depth_delta': 15,
    'mul_both_factor': 35,
    'mul_single_factor': 35,
}
```

### 2. 使用操作处理器注册表

**之前的做法**（长达 50 行的 if-elif 链）:
```python
def _handle_module(self, node: Node):
    module = self.traced.get_submodule(str(node.target))
    if isinstance(module, nn.Conv2d):
        self.conv_statistics(node)
    elif isinstance(module, nn.Linear):
        self.linear_statistics(node)
    elif isinstance(module, nn.ReLU):
        self.relu_statistics(node)
    # ...50 行类似代码
```

**现在的做法**（查表调用）:
```python
def _handle_module(self, node: Node):
    module = self.traced.get_submodule(str(node.target))
    handler_name = self.op_registry.get_module_handler(module)

    if handler_name:
        handler = getattr(self, handler_name)
        handler(node)
    else:
        self.pass_through_statistics(node, f"unknown_{type(module).__name__}")
```

**注册新的模块类型**:
```python
# 在 __init__ 中注册
self.op_registry.register_module(MyCustomModule, 'my_custom_statistics')
```

### 3. 防止自定义激活函数被拆分

如果你有自定义的激活函数（如 `LearnableSwish`, `StablePoly7`），需要防止 FX tracer 将它们拆分成细粒度操作：

```python
# 在 block_def.py 中定义激活函数
class LearnableSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
```

在 `statistics_fn.py` 中自动处理：
```python
# 自定义 Tracer 会自动将这些模块标记为叶子节点
class CustomTracer(fx.Tracer):
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, (LearnableSwish, LearnableRelu, StablePoly7, Relu, Swish)):
            return True  # 不进入内部 trace
        return super().is_leaf_module(m, module_qualified_name)
```

然后注册处理器：
```python
# 在 FheInfo.__init__ 中
self.op_registry.register_module(LearnableSwish, 'learnable_swish_statistics')
```

### 4. 使用 FLOPs 计算器

**之前的做法**（逻辑散落）:
```python
# FLOPs 计算代码在 3 个不同的地方重复
if isinstance(module, nn.Conv2d):
    per_output_flops = kernel_h * kernel_w * (in_channels // groups)
    total_flops = batch * out_h * out_w * out_channels * per_output_flops
```

**现在的做法**（统一计算器）:
```python
from .flops_calculator import FLOPsCalculator

calculator = FLOPsCalculator()
flops = calculator.calculate_conv2d_flops(module, out_shape)
```

### 5. 使用深度分箱工具

```python
from .depth_binning import DepthBinner, DepthMetricsCollector

# 创建分箱器
binner = DepthBinner(bin_size=10, max_bins=30)

# 收集深度数据
collector = DepthMetricsCollector(binner)
for node, meta in self.node_meta_list.items():
    if not meta.is_fused:
        collector.add_latency(meta.out_depth, meta.op_type, meta.latency)
        collector.add_boot(meta.out_depth, meta.boot_latency)

# 获取分箱后的数据
result = collector.get_binned_data()
```

---

## 重构说明

### 重构目标

1. **消除代码重复**: 7 个激活函数方法 → 1 个统一方法 + 配置表
2. **提高可维护性**: 长 if-elif 链 → 注册表模式
3. **职责分离**: 将 FLOPs 计算、深度分箱等功能独立成模块
4. **提高可扩展性**: 添加新功能只需修改配置，无需改动核心代码

### 重构成果

#### 代码行数变化
- **之前**: `statistics_fn.py` ~2800 行（单文件）
- **之后**:
  - `statistics_fn.py`: ~2200 行（核心逻辑）
  - `activation_configs.py`: ~50 行（配置）
  - `operation_registry.py`: ~100 行（注册表）
  - `flops_calculator.py`: ~150 行（FLOPs）
  - `depth_binning.py`: ~100 行（深度工具）
  - **总计**: ~2600 行（模块化）

#### 代码重复消除
- 激活函数处理: **7 个方法** → **1 个方法 + 配置表**
- FLOPs 计算: **3 处重复** → **1 个计算器类**
- 深度分箱: **2 处重复** → **1 个工具类**

#### 可扩展性提升
- 添加新激活函数: ~~修改 50 行代码~~ → **添加 1 行配置**
- 添加新模块类型: ~~修改 if-elif 链~~ → **注册 1 行**
- 修改 FLOPs 计算: ~~修改 3 处~~ → **修改 1 处**

### 设计模式应用

#### 1. 配置表模式 (Configuration Table)
**文件**: `activation_configs.py`

将激活函数的参数配置从代码中分离到数据表：
```python
ACTIVATION_CONFIGS = {
    'relu': {'depth_delta': 15, 'mul_both_factor': 33, 'mul_single_factor': 33},
    'swish': {'depth_delta': 15, 'mul_both_factor': 16, 'mul_single_factor': 16},
    # ...更多配置
}
```

**好处**:
- ✅ 消除重复代码
- ✅ 配置集中管理
- ✅ 易于扩展

#### 2. 注册表模式 (Registry Pattern)
**文件**: `operation_registry.py`

使用映射表替代 if-elif 链：
```python
class OperationHandlerRegistry:
    def __init__(self):
        self._module_handlers = {}
        self._function_handlers = {}
        self._method_handlers = {}

    def register_module(self, module_class, handler_name):
        self._module_handlers[module_class] = handler_name

    def get_module_handler(self, module):
        for module_class, handler_name in self._module_handlers.items():
            if isinstance(module, module_class):
                return handler_name
        return None
```

**好处**:
- ✅ 消除长 if-elif 链
- ✅ 易于添加新操作
- ✅ 支持运行时注册

#### 3. 单一职责原则 (Single Responsibility Principle)
将不同职责分离到独立模块：
- `FLOPsCalculator`: 只负责 FLOPs 计算
- `DepthBinner`: 只负责深度分箱
- `DepthMetricsCollector`: 只负责深度数据收集

**好处**:
- ✅ 代码更易理解
- ✅ 更容易测试
- ✅ 更容易维护

---

## 常见问题

### Q: 如何添加新的激活函数？

**A**: 只需两步：

1. 在 `activation_configs.py` 中添加配置：
```python
ACTIVATION_CONFIGS['new_activation'] = {
    'depth_delta': 15,
    'mul_both_factor': 30,
    'mul_single_factor': 30,
}
```

2. 在 `FheInfo.__init__` 中注册（如果是自定义类）：
```python
self.op_registry.register_module(NewActivation, 'activation_statistics')
```

### Q: 如何防止自定义模块被 FX trace 拆分？

**A**: 在 `CustomTracer.is_leaf_module` 中添加你的模块类型：
```python
class CustomTracer(fx.Tracer):
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, (YourCustomModule, ...)):
            return True
        return super().is_leaf_module(m, module_qualified_name)
```

### Q: 如何修改 boot 计算逻辑？

**A**: 修改 `FheInfo._calc_boot` 方法：
```python
def _calc_boot(self, node_meta: NodeMeta):
    boots_before_in = node_meta.in_depth // self.level
    boots_before_out = node_meta.out_depth // self.level
    boot_triggered = max(0, boots_before_out - boots_before_in)

    node_meta.boot_count = boot_triggered
    node_meta.boot_latency = boot_triggered * node_meta.out_ct * self.boot_cost
```

### Q: 如何添加新的可视化图表？

**A**: 按照以下模式添加新方法：

1. 添加数据准备方法：
```python
def get_your_chart_data(self) -> Dict:
    # 准备数据
    return data
```

2. 添加绘图方法：
```python
def plot_your_chart(self, plot_folder=None, show=True):
    data = self.get_your_chart_data()
    # 使用 matplotlib 绘图
    plt.savefig(...)
```

---

## 参考资料

- [PyTorch FX 文档](https://pytorch.org/docs/stable/fx.html)
- [测试文档](../test/README.md)
- [项目主文档](../README.md)
