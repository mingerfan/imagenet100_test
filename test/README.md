# 测试目录说明

本目录包含用于测试 FHE 网络的各种测试脚本。

## 主要测试文件

### 核心功能测试
- **test_gate_net_models.py** - 测试 gate_net.py 中所有模型的 FHE 统计
  - 支持快速测试模式（测试代表性模型）
  - 支持完整测试模式（测试所有模型）
  - 支持单个模型测试
  - 验证激活函数是否正确保留（未被拆分）

- **test_activation_trace.py** - 测试自定义激活函数在 FX trace 中是否保持完整粒度
  - 验证 LearnableSwish, Swish, LearnableRelu, StablePoly4, Relu 不被拆分

- **test_new_statistics.py** - 测试新的 FHE 统计系统

### 特定功能测试
- **test_flops_fix.py** - 测试 FLOPs 计算修复
- **test_epoch_fix.py** - 测试 epoch 相关修复
- **test_stablepoly4_fix.py** - 测试 StablePoly4 激活函数修复
- **test_pretrained_fix.py** - 测试预训练模型加载修复
- **test_resnet_comparison.py** - ResNet 网络横向对比测试

### 其他测试
- **test_special_resnet.py** - 测试特殊配置的 ResNet 模型
- **test_single_model.py** - 单个模型测试工具
- **test_gate_models.py** - Gate 模型测试
- **test_memory_fs.py** - 内存文件系统测试
- **test_regex_matching.py** - 正则匹配测试
- **test_system.py** - 系统测试

## 使用方法

### 快速测试 gate_net 模型
```bash
uv run test/test_gate_net_models.py --mode quick
```

### 测试所有 gate_net 模型
```bash
uv run test/test_gate_net_models.py --mode all
```

### 测试单个模型
```bash
uv run test/test_gate_net_models.py --mode single --model resnet-basic-relu-layer1block1
```

### 测试激活函数保留性
```bash
uv run test/test_gate_net_models.py --mode activation
```

### 测试激活函数 trace 行为
```bash
uv run test/test_activation_trace.py
```

## 测试选项

### test_gate_net_models.py 参数

- `--mode`: 测试模式
  - `quick`: 快速测试（测试 5 个代表性模型）
  - `all`: 测试所有模型
  - `single`: 测试单个模型
  - `activation`: 测试激活函数保留性

- `--model`: 当 mode=single 时，指定要测试的模型名称（默认: resnet-basic-relu-layer1block1）

- `--sample`: 当 mode=all 时，指定测试的样本数量（默认测试全部）

- `--quiet`: 安静模式，不打印详细信息

## 可用的 gate_net 模型

### Basic Block 系列
- resnet-basic-relu-layer1block1
- resnet-basic-swish-layer1block1
- resnet-basic-learnableswish-layer1block1
- resnet-basic-learnablerelu-layer1block1
- resnet-basic-stablepoly4-layer1block1

### Basic Self-Gated Block 系列
- resnet-basic_self_gated-relu-layer1block1
- resnet-basic_self_gated-swish-layer1block1
- resnet-basic_self_gated-learnableswish-layer1block1
- resnet-basic_self_gated-learnablerelu-layer1block1
- resnet-basic_self_gated-stablepoly4-layer1block1

### Bottleneck Block 系列
- resnet-bottleneck-relu-layer1block1
- resnet-bottleneck-learnableswish-layer1block1
- resnet-bottleneck-learnablerelu-layer1block1
- resnet-bottleneck-stablepoly4-layer1block1

### Bottleneck Self-Gated Block 系列
- resnet-bottleneck_self_gated-relu-layer1block1
- resnet-bottleneck_self_gated-learnableswish-layer1block1
- resnet-bottleneck_self_gated-learnablerelu-layer1block1
- resnet-bottleneck_self_gated-stablepoly4-layer1block1

## 注意事项

1. 所有测试都使用较小的输入尺寸 (64x64) 以加快测试速度
2. 测试会验证：
   - 模型能够正确创建
   - FHE 统计能够正常运行
   - 统计结果的基本有效性（延迟 > 0，深度 > 0 等）
   - 激活函数未被拆分成细粒度操作
3. 测试输出包括参数量、FHE 延迟统计和各算子的详细信息
