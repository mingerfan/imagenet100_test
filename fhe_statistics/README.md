# FHE Statistics - 全同态加密网络统计分析工具

一个用于分析神经网络在全同态加密（FHE）环境下性能的工具包。

## 核心功能

✅ **FHE操作统计** - 自动计算rotation、multiplication、rescale等密文操作
✅ **Boot优化** - 使用动态规划优化bootstrapping插入位置，最小化boot开销
✅ **网络延迟分析** - 精确计算各个网络的FHE延迟
✅ **批量横向比较** - YAML配置驱动的批量网络对比分析

---

## 快速开始

### 方式1：分析单个模型

```python
from fhe_statistics import analyze_model
import torchvision

model = torchvision.models.resnet18()
fhe_info = analyze_model(
    model=model,
    model_name="ResNet18",
    output_folder="results",
    optimize_boot=True  # 使用动态规划优化boot
)
```

### 方式2：比较多个模型

```python
from fhe_statistics import compare_networks
import torchvision

models = [
    ("ResNet18", torchvision.models.resnet18()),
    ("MobileNetV2", torchvision.models.mobilenet_v2()),
]

compare_networks(models, plot_folder="results")
```

### 方式3：批量分析（推荐）

```bash
# 1. 查看可用模型
python fhe_statistics/batch_analyzer.py --list

# 2. 编辑配置文件
vim fhe_statistics/batch_analysis_config.yaml

# 3. 运行批量分析
python fhe_statistics/batch_analyzer.py

# 4. 或只分析特定模型
python fhe_statistics/batch_analyzer.py --models ResNet18 ResNet34
```

---

## 目录结构

```
fhe_statistics/
├── statistics_fn.py              # 主统计分析模块
├── boot_optimizer.py             # Boot动态规划优化器
├── batch_analyzer.py             # 批量分析工具
├── batch_analysis_config.yaml    # 批量分析配置文件
├── activation_configs.py         # 激活函数配置表
├── operation_registry.py         # 操作处理器注册表
├── depth_binning.py              # 深度分箱工具
├── flops_calculator.py           # FLOPs计算器
├── simple_example.py             # 快速示例脚本
├── __init__.py                   # 模块初始化
└── README.md                     # 本文档
```

---

## 使用说明

### 1. 分析单个模型

```python
from fhe_statistics import FheInfo

# 创建FHE分析对象
model = YourModel()
fhe_info = FheInfo(
    model,
    input_shape=(1, 3, 224, 224),
    model_name="YourModel",
    optimize_boot=True  # 启用boot优化
)

# 运行统计
fhe_info.run_statistics()

# 打印结果
fhe_info.print_statistics(output_folder="results")

# 生成图表
fhe_info.plot_statistics(plot_folder="results", show=False)
```

### 2. Boot动态规划优化

工具会自动使用动态规划优化boot插入位置，最小化boot开销：

```python
# 启用boot优化（默认开启）
fhe_info = FheInfo(model, optimize_boot=True)

# 禁用boot优化（使用简单策略：深度超过level就boot）
fhe_info = FheInfo(model, optimize_boot=False)
```

**优化效果示例**：
- 简单策略：在深度超过level时立即boot（可能在高成本节点boot）
- 优化策略：将boot延迟到低成本节点，节省20-40%的boot开销

### 3. 批量分析配置

编辑 `batch_analysis_config.yaml`：

```yaml
global:
  output_folder: "fhe_statistics/results"
  plot_folder: "fhe_statistics/results"
  generate_comparison: true

models:
  # TorchVision预训练模型
  - name: "ResNet18"
    source: "torchvision"
    model_class: "resnet18"
    input_shape: [1, 3, 224, 224]
    enabled: true

  - name: "MobileNetV2"
    source: "torchvision"
    model_class: "mobilenet_v2"
    enabled: true

  # 自定义模型
  - name: "MyCustomModel"
    source: "custom"
    module_path: "models.gate_net"
    model_class: "resnet18"
    params:
      num_classes: 100
      block_type: "basic"
    enabled: true

  # 从checkpoint加载
  - name: "TrainedModel"
    source: "checkpoint"
    module_path: "models.gate_net"
    model_class: "resnet18"
    checkpoint_path: "checkpoints/model.pth"
    enabled: true

comparison:
  plot_types:
    - "network_comparison"        # 横向比较图
    - "comprehensive_comparison"  # 综合对比（6个子图）
    - "grouped_comparison"        # 分组对比（归一化）
```

### 4. 输出结果

**统计文件** (`results/`):
- `{model}_*.txt` - 汇总统计
- `{model}_detailed_*.txt` - 详细逐层统计
- `summary_report_*.txt` - 所有模型汇总

**可视化图表** (`results/`):
- `{model}_basic_*.png` - 基础柱状图
- `{model}_operator_stack_*.png` - 算子堆栈图
- `{model}_depth_histogram_*.png` - 深度分布图
- `network_comparison_*.png` - 网络横向比较
- `network_comprehensive_comparison_*.png` - 综合对比（6子图）
- `network_grouped_comparison_*.png` - 分组对比

---

## 常见使用场景

### 场景1：比较不同网络架构

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
```

运行：`python fhe_statistics/batch_analyzer.py`

### 场景2：比较不同输入分辨率

```yaml
models:
  - name: "ResNet18-96x96"
    source: "torchvision"
    model_class: "resnet18"
    input_shape: [1, 3, 96, 96]
    enabled: true

  - name: "ResNet18-224x224"
    source: "torchvision"
    model_class: "resnet18"
    input_shape: [1, 3, 224, 224]
    enabled: true
```

### 场景3：分析训练后的模型

```yaml
models:
  - name: "MyTrainedModel"
    source: "checkpoint"
    module_path: "models.gate_net"
    model_class: "resnet18"
    checkpoint_path: "checkpoints/my_model.pth"
    params:
      num_classes: 100
    enabled: true
```

---

## FHE参数配置

在配置文件中自定义FHE参数：

```yaml
fhe_params:
  rotation_cost: 180      # 旋转操作成本
  rescale_cost: 40        # 重缩放成本
  mul_single_cost: 9.5    # 密文-明文乘法成本
  mul_double_cost: 253    # 密文-密文乘法成本
  boot_cost: 98136        # Bootstrapping成本
  level: 10               # 密文深度容量
  slots_num: 32768        # 槽位数量
```

---

## 运行示例脚本

```bash
# 示例1：分析单个模型
python fhe_statistics/simple_example.py --example 1

# 示例2：比较多个模型
python fhe_statistics/simple_example.py --example 2

# 示例3：批量分析说明
python fhe_statistics/simple_example.py --example 3
```

---

## API参考

### `FheInfo` 类

主统计分析类。

```python
FheInfo(
    model: nn.Module,           # PyTorch模型
    input_shape: Tuple[int],    # 输入形状，默认(1,3,224,224)
    model_name: str = None,     # 模型名称
    optimize_boot: bool = True  # 是否优化boot插入
)
```

**主要方法**：
- `run_statistics()` - 运行统计分析
- `print_statistics(output_folder)` - 打印统计结果
- `plot_statistics(plot_folder)` - 生成可视化图表
- `get_max_depth()` - 获取最大深度
- `get_parameter_count()` - 获取参数量
- `get_flops_count()` - 获取FLOPs

### `analyze_model` 函数

快捷分析函数。

```python
analyze_model(
    model: nn.Module,
    model_name: str,
    output_folder: str = None,
    plot_folder: str = None,
    input_shape: Tuple[int] = (1,3,224,224),
    print_detailed: bool = True,
    optimize_boot: bool = True
) -> FheInfo
```

### `BootOptimizer` 类

Boot插入优化器。

```python
from fhe_statistics import BootOptimizer, NodeInfo

# 准备节点信息
nodes = [
    NodeInfo(index=0, name="conv1", depth_delta=2, ct_num=10, op_type="conv"),
    NodeInfo(index=1, name="relu1", depth_delta=1, ct_num=10, op_type="relu"),
    # ...
]

# 运行优化
optimizer = BootOptimizer(level=10, boot_cost=98136)
boot_plan, min_cost = optimizer.optimize(nodes)
```

---

## 故障排除

### Q: 内存不足

**A**: 减少同时分析的模型数量，或使用较小的输入分辨率：

```yaml
global:
  print_detailed: false
  generate_plots: false
```

### Q: 某个模型加载失败

**A**: 将该模型标记为可选：

```yaml
- name: "ProblematicModel"
  enabled: true
  optional: true  # 加载失败不会中断
```

### Q: 找不到自定义模块

**A**: 检查模块路径和类名是否正确：

```yaml
- name: "MyModel"
  source: "custom"
  module_path: "models.gate_net"  # 确保路径正确
  model_class: "resnet18"         # 确保类名存在
```

---

## 技术细节

### Boot优化算法

使用动态规划求解全局最优boot插入位置：

**问题定义**：
- 密文深度容量为 `level`（通常为10）
- 每个节点增加深度 `depth_delta`
- 在节点i后插入boot的成本：`ct_num[i] * boot_cost`
- 目标：最小化总boot成本，同时满足深度约束

**动态规划状态**：
- `dp[i][d]` = 处理到第i个节点，当前深度为d时的最小boot成本

**状态转移**：
1. 不在节点i后boot：`dp[i+1][d + depth_delta] = dp[i][d]`
2. 在节点i后boot：`dp[i+1][d + depth_delta - level] = dp[i][d] + ct_num * boot_cost`

**优化效果**：
- 将boot延迟到ct_num较小的节点执行
- 通常可节省20-40%的boot开销

---

## 开发者指南

### 添加新的激活函数

只需在 `activation_configs.py` 中添加配置：

```python
ACTIVATION_CONFIGS['new_activation'] = {
    'depth_delta': 15,
    'mul_both_factor': 33,
    'mul_single_factor': 33,
}
```

### 注册自定义模块

在初始化时注册：

```python
fhe_info = FheInfo(model)
fhe_info.op_registry.register_module(MyCustomModule, 'my_custom_statistics')
```

---

## 引用

如果使用本工具，请引用：

```bibtex
@software{fhe_statistics,
  title = {FHE Statistics: Neural Network Performance Analysis for Fully Homomorphic Encryption},
  year = {2024},
  author = {Your Name}
}
```

---

## 许可证

MIT License

---

## 联系方式

如有问题或建议，请联系：your.email@example.com
