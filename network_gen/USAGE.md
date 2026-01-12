# 网络生成器配置系统使用指南

## 概述

网络生成器现在支持通过YAML配置文件来控制网络生成，可以针对不同数据集（如ImageNet、CIFAR-10等）设置特定的约束条件。不同配置生成的网络会自动保存到不同的文件夹。

## 主要功能

1. **配置文件控制**: 通过YAML文件定义搜索空间约束
2. **多数据集支持**: 针对不同分辨率的数据集使用不同配置
3. **目录隔离**: 不同配置的网络自动保存到不同文件夹
4. **灵活约束**: 支持控制stem层、前几层的stride、block类型选择等
5. **向后兼容**: 仍然支持纯命令行参数方式

## 快速开始

### 1. 使用预设配置文件

#### ImageNet-100 (224x224)
```bash
cd network_gen
python batch_generator.py --config configs/imagenet_224.yaml -n 100 --verify
```

#### CIFAR-10 (32x32)
```bash
cd network_gen
python batch_generator.py --config configs/cifar10_32.yaml -n 50 --verify
```

### 2. 在Python代码中使用

```python
from network_gen.generator_config import GeneratorConfig, ConfigManager
from network_gen.network_generator import RandomNetworkGenerator

# 加载配置
config = GeneratorConfig.from_yaml("network_gen/configs/imagenet_224.yaml")
print(config.summary())

# 创建生成器
generator = RandomNetworkGenerator(config=config, seed=42)

# 生成单个网络
network_config = generator.generate_random_config()
print(network_config.summary())

# 生成批量网络
batch = generator.generate_batch(
    num_configs=50,
    batch_name="my_batch",
    description="My custom batch"
)

# 使用配置管理器保存
config_manager = ConfigManager(config)
config_manager.save_batch(batch, overwrite=True)
print(config_manager.summary())
```

## 配置文件详解

### 基本结构

```yaml
name: "配置名称"
description: "配置描述"

dataset:
  name: "数据集名称"        # imagenet, cifar10, etc.
  num_classes: 100          # 分类数量
  input_size: 224           # 输入图像大小

search_space:
  ct_slots: 32768           # CT槽位数
  initial_ct_count: 1       # 初始CT数量

  stem:                     # Stem层配置
    enabled: true
    allowed_codes: null     # null表示允许所有

  second_downsample:        # 第二次降分辨率
    enabled: true
    allowed_codes: null

  blocks:                   # Block配置
    allowed_block_ids: null # null表示允许所有24种
    first_layers_constraints: null  # 前几层的特殊约束

  stride:                   # Stride配置
    allowed_block_counts: null  # 允许的block数量
    num_strides: 3          # 降分辨率次数

  ct_policies:              # CT策略
    allowed: ["keep", "half"]

output:
  base_dir: "generated_networks/imagenet_224"
  save_format: "json"
```

### 配置约束示例

#### 示例1: 限制前两层不降分辨率 (CIFAR-10)

```yaml
blocks:
  allowed_block_ids: null
  first_layers_constraints:
    - position: 0           # 第1个block
      stride: 1             # 强制stride=1
      allowed_block_ids: null
    - position: 1           # 第2个block
      stride: 1             # 强制stride=1
      allowed_block_ids: null

stride:
  allowed_block_counts: [6, 8, 10, 12]  # 较少的block
  num_strides: 2            # 只降2次分辨率
```

#### 示例2: 只允许特定的block类型

```yaml
blocks:
  allowed_block_ids: [0, 1, 4, 5, 6, 7, 8, 9]  # 只允许basic和部分bottleneck
  first_layers_constraints: null
```

#### 示例3: 禁用stem和第二次降分辨率

```yaml
stem:
  enabled: false

second_downsample:
  enabled: false
```

## 不同配置的输出目录

每个配置文件的`output.base_dir`指定了输出目录，例如：

```
generated_networks/
├── imagenet_224/          # ImageNet配置的网络
│   ├── batch_20240115_143022.json
│   ├── net_a1b2c3d4.json
│   └── net_e5f6g7h8.json
├── cifar10_32/            # CIFAR-10配置的网络
│   ├── batch_20240115_143555.json
│   ├── net_x1y2z3w4.json
│   └── net_m5n6o7p8.json
└── custom_config/         # 自定义配置
    └── ...
```

## 创建自定义配置

### 步骤1: 复制现有配置

```bash
cp network_gen/configs/imagenet_224.yaml network_gen/configs/my_custom.yaml
```

### 步骤2: 编辑配置文件

修改`my_custom.yaml`中的参数，例如：

```yaml
name: "my_custom_config"
description: "My custom configuration for specific use case"

dataset:
  name: "my_dataset"
  num_classes: 200
  input_size: 128

search_space:
  # ... 自定义约束 ...

output:
  base_dir: "generated_networks/my_custom"
```

### 步骤3: 使用自定义配置

```bash
python batch_generator.py --config network_gen/configs/my_custom.yaml -n 100
```

## 命令行参数参考

### 主要参数

- `-c, --config`: 配置文件路径（YAML格式）
- `-n, --num`: 生成的网络数量（默认: 50）
- `-o, --output`: 输出目录（覆盖配置文件中的设置）
- `--batch-name`: 批次名称（默认: 使用时间戳）
- `--seed`: 随机种子
- `--verify`: 验证生成的网络可以正确构建
- `--save-individual`: 保存每个配置的单独JSON文件

### 向后兼容参数（不使用配置文件时）

- `--ct-slots`: CT槽位数（默认: 32768）
- `--input-size`: 输入图像大小（默认: 224）
- `--num-classes`: 分类数量（默认: 100）

## 高级用法

### 1. 批量生成多个配置的网络

```bash
# ImageNet
python batch_generator.py --config configs/imagenet_224.yaml -n 100 --batch-name imagenet_v1

# CIFAR-10
python batch_generator.py --config configs/cifar10_32.yaml -n 50 --batch-name cifar10_v1
```

### 2. 在代码中动态创建配置

```python
from network_gen.generator_config import (
    GeneratorConfig,
    DatasetConfig,
    SearchSpaceConstraints,
    OutputConfig,
    LayerConstraint,
)

# 创建自定义配置
config = GeneratorConfig(
    name="dynamic_config",
    description="Dynamically created config",
    dataset=DatasetConfig(
        name="custom_dataset",
        num_classes=50,
        input_size=96,
    ),
    search_space=SearchSpaceConstraints(
        ct_slots=32768,
        initial_ct_count=1,
        # ... 其他约束 ...
    ),
    output=OutputConfig(
        base_dir="generated_networks/dynamic",
    ),
)

# 保存配置供以后使用
config.save("my_configs/dynamic.yaml")
```

### 3. 加载和分析已生成的网络

```python
from network_gen.generator_config import GeneratorConfig, ConfigManager

# 加载配置
config = GeneratorConfig.from_yaml("configs/imagenet_224.yaml")
manager = ConfigManager(config)

# 列出所有批次
batches = manager.list_batches()
print(f"可用批次: {batches}")

# 加载特定批次
batch = manager.load_batch("batch_20240115_143022")
print(batch.summary())

# 加载单个网络配置
network_config = manager.load_network_config("net_a1b2c3d4")
print(network_config.summary())
```

## Block类型参考

系统支持24种预定义的block类型（ID 0-23）：

- **0-1**: BasicBlock (poly4, swish)
- **2-3**: BasicSelfGatedBlock (poly4, swish)
- **4-13**: BottleneckBlock (5种factor × 2种激活)
  - factor: 0.25, 0.5, 1.0, 1.5, 2.0
- **14-23**: BottleneckSelfGatedBlock (5种factor × 2种激活)

详细信息见 `search_space.py` 中的 `UNIFIED_BLOCKS`。

## 常见问题

### Q: 配置文件和命令行参数同时使用会怎样？

A: 优先使用配置文件的设置。但是`--output`参数可以覆盖配置文件中的输出目录。

### Q: 如何确保不同配置生成的网络不会混淆？

A: 在配置文件的`output.base_dir`中指定不同的目录即可。例如：
- ImageNet: `generated_networks/imagenet_224`
- CIFAR-10: `generated_networks/cifar10_32`

### Q: first_layers_constraints中的position是什么意思？

A: position指的是网络中第几个block（从0开始），不是block内部的层。
- position=0: 网络的第1个block
- position=1: 网络的第2个block

### Q: 如何禁用某些功能（如stem层）？

A: 在配置文件中设置`enabled: false`：
```yaml
stem:
  enabled: false
```

### Q: 配置约束会影响搜索空间大小吗？

A: 会。约束越多，搜索空间越小。例如：
- 限制block类型：从24种减少到指定的几种
- 限制block数量：从7种减少到指定的几种
- 前几层的stride约束：减少stride组合的可能性

## 参考文档

- 配置文件说明: `network_gen/configs/README.md`
- 搜索空间定义: `network_gen/search_space.py`
- 网络生成器: `network_gen/network_generator.py`
- 配置系统: `network_gen/generator_config.py`
