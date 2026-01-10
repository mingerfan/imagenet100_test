# FHE Statistics 批量分析 - 快速开始

## 第一步：列出所有可用模型

```bash
python fhe_statistics/batch_analyzer.py --list
```

这会列出配置文件中的所有模型及其状态（已启用/已禁用）。

## 第二步：编辑配置文件

编辑 `fhe_statistics/batch_analysis_config.yaml`，启用你想要分析的模型：

```yaml
models:
    # 将 enabled: false 改为 enabled: true
    - name: "ResNet18"
      source: "torchvision"
      model_class: "resnet18"
      input_shape: [1, 3, 224, 224]
      enabled: true  # 👈 修改这里
```

## 第三步：运行批量分析

```bash
python fhe_statistics/batch_analyzer.py
```

分析完成后，结果保存在：
- 统计文件: `fhe_statistics/results/`
- 可视化图表: `fhe_statistics/plots/`

## 常用命令

### 只分析特定模型

```bash
python fhe_statistics/batch_analyzer.py --models ResNet18 ResNet34
```

### 使用自定义配置

```bash
python fhe_statistics/batch_analyzer.py --config my_config.yaml
```

### 运行示例

```bash
# 示例1：列出所有模型
python fhe_statistics/example_usage.py --example 1

# 示例2：分析特定模型
python fhe_statistics/example_usage.py --example 2

# 示例3：通过代码配置
python fhe_statistics/example_usage.py --example 3

# 示例4：比较不同分辨率
python fhe_statistics/example_usage.py --example 4
```

## 常见使用场景

### 场景1：比较多个预训练网络

在配置文件中启用想要比较的网络：

```yaml
models:
    - name: "ResNet18"
      enabled: true

    - name: "ResNet50"
      enabled: true

    - name: "MobileNetV2"
      enabled: true

    - name: "VGG16"
      enabled: true
```

运行：

```bash
python fhe_statistics/batch_analyzer.py
```

### 场景2：比较不同输入分辨率

在配置文件中添加同一个模型的不同分辨率版本：

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

### 场景3：分析自定义训练的模型

```yaml
models:
    - name: "MyModel"
      source: "checkpoint"
      module_path: "models.gate_net"
      model_class: "resnet18"
      checkpoint_path: "checkpoints/my_model.pth"
      params:
          num_classes: 100
          block_type: "basic"
          activation_type: "relu"
      input_shape: [1, 3, 224, 224]
      enabled: true
```

## 查看结果

### 汇总报告

查看 `fhe_statistics/results/summary_report_*.txt`：

```
Model Name                     FHE Latency    Boot Latency           Total    Max Depth    Params(M)     FLOPs(M)
------------------------------------------------------------------------------------------------------------------------
ResNet18                         12345.67        23456.78        35802.45           45        11.69       1820.00
MobileNetV2                       8765.43        15432.10        24197.53           38         3.50        300.78
VGG16                            23456.78        34567.89        58024.67           52       138.36       15503.00
```

### 可视化图表

在 `fhe_statistics/plots/` 目录查看：

1. **网络横向比较图** (`network_comparison_*.png`)
   - 显示不同网络各算子的延迟堆叠图

2. **综合比较图** (`network_comprehensive_comparison_*.png`)
   - 6个子图：FHE延迟、准确率、参数量、最大深度、浅层延迟比例、浅层参数比例

3. **分组比较图** (`network_grouped_comparison_*.png`)
   - 横向对比所有指标，已归一化

4. **单个模型详细图**
   - `{model}_basic_*.png` - 基础统计
   - `{model}_operator_stack_*.png` - 算子堆栈
   - `{model}_depth_histogram_*.png` - 深度分布

## 进阶配置

### 调整FHE参数

```yaml
fhe_params:
    rotation_cost: 180
    rescale_cost: 40
    mul_single_cost: 9.5
    mul_double_cost: 253
    boot_cost: 98136
    level: 10
    slots_num: 32768
```

### 自定义输出目录

```yaml
global:
    output_folder: "my_results"
    plot_folder: "my_plots"
```

### 控制输出详细程度

```yaml
global:
    print_detailed: false      # 关闭详细统计
    generate_plots: false      # 不生成图表
    generate_comparison: true  # 只生成比较图
```

## 故障排除

### 问题1：内存不足

**解决方案**：
- 减少同时分析的模型数量
- 使用较小的输入分辨率
- 关闭详细统计和图表生成

```yaml
global:
    print_detailed: false
    generate_plots: false
```

### 问题2：某个模型加载失败

**解决方案**：将该模型标记为可选

```yaml
- name: "ProblematicModel"
  enabled: true
  optional: true  # 👈 添加这行
```

### 问题3：找不到自定义模型

**解决方案**：检查模块路径是否正确

```yaml
- name: "MyModel"
  source: "custom"
  module_path: "models.gate_net"  # 确保这个路径正确
  model_class: "resnet18"         # 确保类名正确
```

## 完整文档

详细文档请参考：[fhe_statistics/README.md](README.md)
