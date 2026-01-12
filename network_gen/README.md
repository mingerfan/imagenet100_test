# 网络生成器配置系统

一个灵活的配置系统，支持通过YAML文件控制神经网络架构的生成，专为不同数据集和输入分辨率优化。

## 目录

- [概述](#概述)
- [测试结果](#测试结果)
- [快速开始](#快速开始)
- [配置文件说明](#配置文件说明)
- [核心功能](#核心功能)
- [使用示例](#使用示例)
- [命令行参数](#命令行参数)
- [高级用法](#高级用法)
- [Block类型参考](#block类型参考)
- [常见问题](#常见问题)
- [文件清单](#文件清单)

---

## 概述

网络生成器现在支持通过YAML配置文件来控制网络生成，可以针对不同数据集（如ImageNet、CIFAR-10等）设置特定的约束条件。不同配置生成的网络会自动保存到不同的文件夹。

### 主要特性

- ✅ **配置文件控制**: 通过YAML文件定义搜索空间约束
- ✅ **多数据集支持**: 针对不同分辨率的数据集使用不同配置
- ✅ **目录隔离**: 不同配置的网络自动保存到不同文件夹
- ✅ **灵活约束**: 支持控制stem层、前几层的stride、block类型选择等
- ✅ **位置特定约束**: 可以强制指定位置的block行为
- ✅ **向后兼容**: 仍然支持纯命令行参数方式

---

## 测试结果

所有测试通过！系统完全正常工作。

```
✓ 配置文件加载成功
✓ 网络生成成功
✓ CIFAR-10约束验证通过
✓ 配置管理器测试通过
✓ 批量生成测试通过
```

运行测试:
```bash
uv run python network_gen/test_config_system.py
```

---

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
```

---

## 配置文件说明

### 预设配置

#### ImageNet-100配置 (`configs/imagenet_224.yaml`)

标准配置，允许所有的搜索空间选项：

- **输入分辨率**: 224x224
- **类别数**: 100
- **Stem层**: 启用 (224→56)
- **降分辨率次数**: 3次 (56→28→14→7)
- **Block数量**: 4, 6, 8, 10, 12, 14, 16（任意）
- **输出目录**: `generated_networks/imagenet_224/`

#### CIFAR-10配置 (`configs/cifar10_32.yaml`)

针对小分辨率图像的保守策略：

- **输入分辨率**: 32x32
- **类别数**: 10
- **Stem层**: 禁用（避免激进降分辨率）
- **第二次降分辨率**: 禁用
- **前两层约束**: 强制stride=1（不降分辨率）
- **降分辨率次数**: 2次 (32→16→8)
- **Block数量**: 6, 8, 10, 12
- **输出目录**: `generated_networks/cifar10_32/`

### 配置文件结构

```yaml
name: "配置名称"
description: "配置描述"

# 数据集配置
dataset:
  name: "数据集名称"        # imagenet, cifar10, etc.
  num_classes: 100          # 分类数量
  input_size: 224           # 输入图像大小

# 搜索空间配置
search_space:
  ct_slots: 32768           # CT槽位数
  initial_ct_count: 1       # 初始CT数量

  # Stem层配置
  stem:
    enabled: true           # 是否启用stem层
    allowed_codes: null     # 允许的配置 [0-3], null=允许所有

  # 第二次降分辨率配置
  second_downsample:
    enabled: true           # 是否启用
    allowed_codes: null     # 允许的配置 [0-4], null=允许所有

  # Block配置
  blocks:
    allowed_block_ids: null              # 允许的block ID [0-23], null=所有
    first_layers_constraints: null       # 前几层的特殊约束

  # Stride配置
  stride:
    allowed_block_counts: null           # 允许的block数量, null=所有
    num_strides: 3                       # Body部分降分辨率次数

  # CT策略配置
  ct_policies:
    allowed: ["keep", "half"]            # 允许的CT策略

# 输出配置
output:
  base_dir: "generated_networks/imagenet_224"  # 输出目录
  save_format: "json"                          # 保存格式
```

### 配置约束详解

#### 1. 限制前N层不降分辨率 (CIFAR-10示例)

```yaml
blocks:
  allowed_block_ids: null
  first_layers_constraints:
    - position: 0           # 第1个block
      stride: 1             # 强制stride=1，不降分辨率
      allowed_block_ids: null
    - position: 1           # 第2个block
      stride: 1             # 强制stride=1
      allowed_block_ids: null

stride:
  allowed_block_counts: [6, 8, 10, 12]  # 较少的block
  num_strides: 2            # 只降2次分辨率
```

**说明**:
- `position`: 指定网络中第几个block（从0开始）
  - position=0 表示网络的第1个block
  - position=1 表示网络的第2个block
  - **不是**block内部的层
- `stride`: 强制该block的stride值
- `allowed_block_ids`: 该位置允许的block类型（可选）

#### 2. 只允许特定的block类型

```yaml
blocks:
  allowed_block_ids: [0, 1, 4, 5, 6, 7, 8, 9]  # 只允许basic和部分bottleneck
  first_layers_constraints: null
```

#### 3. 禁用stem和第二次降分辨率

```yaml
stem:
  enabled: false

second_downsample:
  enabled: false
```

#### 4. 限制block数量范围

```yaml
stride:
  allowed_block_counts: [8, 10, 12]  # 只允许这些block数量
  num_strides: 3
```

---

## 核心功能

### 1. 灵活的约束系统

可以通过配置文件控制网络生成的各个方面：

- **Stem层**: 启用/禁用，允许的配置类型
- **降分辨率**: 启用/禁用第二次降分辨率
- **Block选择**: 全局允许的block类型
- **位置约束**: 针对特定位置的block和stride约束
- **Block数量**: 允许的block数量范围
- **CT策略**: 允许的CT策略类型

### 2. 位置特定约束 (first_layers_constraints)

这是解决CIFAR-10等小分辨率数据集问题的关键功能。

**工作原理**:
- `position`: 指定网络中的第几个block（从0开始）
- `stride`: 强制该位置的stride值
- `allowed_block_ids`: 该位置允许使用的block类型（可选）

**示例**:
```yaml
first_layers_constraints:
  - position: 0           # 网络的第1个block
    stride: 1             # 强制stride=1，保持分辨率
    allowed_block_ids: [0, 1, 2, 3]  # 可选：限制block类型
  - position: 1           # 网络的第2个block
    stride: 1             # 强制stride=1
```

**效果对比**:
- **没有约束**: CIFAR-10可能在第1个block就降到16x16（太激进）
- **有约束**: 前两个block保持32x32，之后再降分辨率

### 3. 目录隔离

不同配置自动使用不同的输出目录，避免混淆：

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

### 4. ConfigManager

配置管理器负责管理不同配置的输出目录和文件：

**主要功能**:
```python
config_manager = ConfigManager(config)

# 保存
config_manager.save_network_config(network_config)
config_manager.save_batch(batch)

# 加载
network = config_manager.load_network_config("net_xxx")
batch = config_manager.load_batch("batch_xxx")

# 列出
configs = config_manager.list_configs()
batches = config_manager.list_batches()

# 摘要
print(config_manager.summary())
```

### 5. 向后兼容

系统保持向后兼容，不使用配置文件也能正常工作：

```bash
# 旧方式仍然可用
python batch_generator.py -n 50 --input-size 224 --ct-slots 32768
```

---

## 使用示例

### 示例1: 使用ImageNet配置

```python
from network_gen.generator_config import GeneratorConfig, ConfigManager
from network_gen.network_generator import RandomNetworkGenerator

# 加载配置
config = GeneratorConfig.from_yaml("network_gen/configs/imagenet_224.yaml")
print(config.summary())

# 创建生成器和管理器
generator = RandomNetworkGenerator(config=config, seed=42)
manager = ConfigManager(config)

# 生成批量网络
batch = generator.generate_batch(num_configs=100, batch_name="imagenet_v1")

# 保存到配置指定的目录
manager.save_batch(batch, overwrite=True)
# 保存到: generated_networks/imagenet_224/batch_imagenet_v1.json
```

### 示例2: 使用CIFAR-10配置

```python
# 加载CIFAR-10配置
config = GeneratorConfig.from_yaml("network_gen/configs/cifar10_32.yaml")

# 生成器会自动应用约束
generator = RandomNetworkGenerator(config=config, seed=42)

# 生成的网络会满足：
# - 前两个block的stride为1
# - Block数量在[6,8,10,12]中
# - 只降2次分辨率
network = generator.generate_random_config()

# 验证约束
assert network.blocks[0].stride == 1  # 第1个block
assert network.blocks[1].stride == 1  # 第2个block
assert network.num_blocks in [6, 8, 10, 12]

# 保存到 generated_networks/cifar10_32/ 目录
manager = ConfigManager(config)
manager.save_network_config(network)
```

### 示例3: 创建自定义配置

#### 方法1: 从现有配置修改

```bash
# 复制现有配置
cp network_gen/configs/imagenet_224.yaml network_gen/configs/my_custom.yaml

# 编辑配置文件
# vim my_custom.yaml
# 修改 dataset, search_space, output 等

# 使用自定义配置
python batch_generator.py --config network_gen/configs/my_custom.yaml -n 100
```

#### 方法2: 在代码中动态创建

```python
from network_gen.generator_config import *

# 创建自定义配置
config = GeneratorConfig(
    name="my_custom",
    description="Custom config for 128x128 images",
    dataset=DatasetConfig(
        name="custom_dataset",
        num_classes=50,
        input_size=128,
    ),
    search_space=SearchSpaceConstraints(
        ct_slots=32768,
        initial_ct_count=1,
        stem=StemConstraints(enabled=True, allowed_codes=[0, 1]),
        blocks=BlockConstraints(
            allowed_block_ids=[0, 1, 2, 3],  # 只允许basic类型
            first_layers_constraints=[
                LayerConstraint(position=0, stride=1),
            ],
        ),
        stride=StrideConstraints(
            allowed_block_counts=[8, 10, 12],
            num_strides=3,
        ),
    ),
    output=OutputConfig(
        base_dir="generated_networks/custom_128",
    ),
)

# 保存供以后使用
config.save("my_configs/custom.yaml")

# 使用配置
generator = RandomNetworkGenerator(config=config)
batch = generator.generate_batch(num_configs=50)
```

### 示例4: 批量生成多个配置的网络

```bash
# ImageNet
python batch_generator.py --config configs/imagenet_224.yaml -n 100 \
  --batch-name imagenet_v1 --verify --save-individual

# CIFAR-10
python batch_generator.py --config configs/cifar10_32.yaml -n 50 \
  --batch-name cifar10_v1 --verify --save-individual

# 自定义配置
python batch_generator.py --config configs/my_custom.yaml -n 80 \
  --batch-name custom_v1
```

### 示例5: 加载和分析已生成的网络

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

# 构建网络模型
from network_gen.network_generator import create_network
model = create_network(network_config)
```

---

## 命令行参数

### 主要参数

```bash
python batch_generator.py [OPTIONS]

-c, --config PATH       # 配置文件路径（YAML格式）
-n, --num NUM          # 生成的网络数量（默认: 50）
-o, --output DIR       # 输出目录（覆盖配置文件）
--batch-name NAME      # 批次名称（默认: 使用时间戳）
--seed SEED            # 随机种子
--verify               # 验证生成的网络可以正确构建
--save-individual      # 保存每个配置的单独JSON文件
```

### 向后兼容参数（不使用配置文件时）

```bash
--ct-slots NUM         # CT槽位数（默认: 32768）
--input-size SIZE      # 输入图像大小（默认: 224）
--num-classes NUM      # 分类数量（默认: 100）
```

### 使用示例

```bash
# 使用配置文件
python batch_generator.py --config configs/imagenet_224.yaml -n 100

# 覆盖输出目录
python batch_generator.py --config configs/cifar10_32.yaml -n 50 -o my_output/

# 验证网络
python batch_generator.py --config configs/imagenet_224.yaml -n 10 --verify

# 保存单独文件
python batch_generator.py --config configs/cifar10_32.yaml -n 20 --save-individual

# 不使用配置文件（向后兼容）
python batch_generator.py -n 50 --input-size 224 --num-classes 100
```

---

## 高级用法

### 1. 组合多个约束

创建一个只使用特定block类型且有位置约束的配置：

```yaml
blocks:
  allowed_block_ids: [0, 1, 4, 5, 6, 7]  # 只允许basic和小factor的bottleneck
  first_layers_constraints:
    - position: 0
      stride: 1
      allowed_block_ids: [0, 1]  # 第1层只用basic
    - position: 1
      stride: 1
      allowed_block_ids: [0, 1]  # 第2层只用basic
```

### 2. 限制搜索空间大小

通过多个约束减小搜索空间：

```yaml
stem:
  allowed_codes: [0, 2]  # 只允许2种stem配置

blocks:
  allowed_block_ids: [0, 1, 4, 5]  # 只允许4种block

stride:
  allowed_block_counts: [8, 10]  # 只允许2种block数量
  num_strides: 3
```

### 3. 批量生成不同seed的网络

```python
from network_gen.generator_config import GeneratorConfig, ConfigManager
from network_gen.network_generator import RandomNetworkGenerator

config = GeneratorConfig.from_yaml("configs/imagenet_224.yaml")
manager = ConfigManager(config)

# 生成多个批次，每个使用不同的seed
for seed in range(5):
    generator = RandomNetworkGenerator(config=config, seed=seed)
    batch = generator.generate_batch(
        num_configs=20,
        batch_name=f"imagenet_seed{seed}",
    )
    manager.save_batch(batch, overwrite=True)
```

### 4. 动态调整配置

```python
# 加载基础配置
config = GeneratorConfig.from_yaml("configs/imagenet_224.yaml")

# 动态修改
config.search_space.blocks.allowed_block_ids = [0, 1, 2, 3]  # 只用basic
config.search_space.stride.allowed_block_counts = [10, 12]    # 固定block数
config.output.base_dir = "generated_networks/imagenet_basic_only"

# 使用修改后的配置
generator = RandomNetworkGenerator(config=config)
batch = generator.generate_batch(num_configs=50)
```

---

## Block类型参考

系统支持24种预定义的block类型（ID 0-23）：

### Basic Blocks (0-3)

| ID | Name | Description |
|----|------|-------------|
| 0 | basic_poly4 | BasicBlock + Poly4激活 |
| 1 | basic_swish | BasicBlock + Swish激活 |
| 2 | basic_sg_poly4 | BasicSelfGatedBlock + Poly4激活 |
| 3 | basic_sg_swish | BasicSelfGatedBlock + Swish激活 |

### Bottleneck Blocks (4-13)

| ID | Name | Factor | Activation |
|----|------|--------|------------|
| 4 | btn_f0.25_poly4 | 0.25 | Poly4 |
| 5 | btn_f0.25_swish | 0.25 | Swish |
| 6 | btn_f0.5_poly4 | 0.5 | Poly4 |
| 7 | btn_f0.5_swish | 0.5 | Swish |
| 8 | btn_f1.0_poly4 | 1.0 | Poly4 |
| 9 | btn_f1.0_swish | 1.0 | Swish |
| 10 | btn_f1.5_poly4 | 1.5 | Poly4 |
| 11 | btn_f1.5_swish | 1.5 | Swish |
| 12 | btn_f2.0_poly4 | 2.0 | Poly4 |
| 13 | btn_f2.0_swish | 2.0 | Swish |

### Bottleneck Self-Gated Blocks (14-23)

| ID | Name | Factor | Activation |
|----|------|--------|------------|
| 14 | btn_sg_f0.25_poly4 | 0.25 | Poly4 |
| 15 | btn_sg_f0.25_swish | 0.25 | Swish |
| 16 | btn_sg_f0.5_poly4 | 0.5 | Poly4 |
| 17 | btn_sg_f0.5_swish | 0.5 | Swish |
| 18 | btn_sg_f1.0_poly4 | 1.0 | Poly4 |
| 19 | btn_sg_f1.0_swish | 1.0 | Swish |
| 20 | btn_sg_f1.5_poly4 | 1.5 | Poly4 |
| 21 | btn_sg_f1.5_swish | 1.5 | Swish |
| 22 | btn_sg_f2.0_poly4 | 2.0 | Poly4 |
| 23 | btn_sg_f2.0_swish | 2.0 | Swish |

详细信息见 `search_space.py` 中的 `UNIFIED_BLOCKS`。

---

## 常见问题

### Q1: 配置文件和命令行参数同时使用会怎样？

**A**: 优先使用配置文件的设置。但是`--output`参数可以覆盖配置文件中的输出目录。

```bash
# 使用配置文件，但输出到自定义目录
python batch_generator.py --config configs/imagenet_224.yaml -n 100 -o my_output/
```

### Q2: 如何确保不同配置生成的网络不会混淆？

**A**: 在配置文件的`output.base_dir`中指定不同的目录即可。

```yaml
# ImageNet
output:
  base_dir: "generated_networks/imagenet_224"

# CIFAR-10
output:
  base_dir: "generated_networks/cifar10_32"
```

### Q3: first_layers_constraints中的position是什么意思？

**A**: `position`指的是网络中第几个block（从0开始），**不是block内部的层**。

- `position=0`: 网络的第1个block
- `position=1`: 网络的第2个block

示例：
```yaml
first_layers_constraints:
  - position: 0    # 网络的第1个block
    stride: 1      # 这个block的stride=1
  - position: 1    # 网络的第2个block
    stride: 1      # 这个block的stride=1
```

### Q4: 如何禁用某些功能（如stem层）？

**A**: 在配置文件中设置`enabled: false`：

```yaml
stem:
  enabled: false

second_downsample:
  enabled: false
```

### Q5: 配置约束会影响搜索空间大小吗？

**A**: 会。约束越多，搜索空间越小。

示例：
- **限制block类型**: 从24种减少到指定的几种
- **限制block数量**: 从7种减少到指定的几种
- **前几层的stride约束**: 减少stride组合的可能性

无约束的ImageNet搜索空间：~10^19
有约束的CIFAR-10搜索空间：~10^15 (减少了约10000倍)

### Q6: 如何验证约束是否正确应用？

**A**: 生成网络后检查配置：

```python
config = generator.generate_random_config()

# 检查前两层的stride
assert config.blocks[0].stride == 1
assert config.blocks[1].stride == 1

# 检查block数量
assert config.num_blocks in [6, 8, 10, 12]

# 检查block类型
for block in config.blocks:
    assert block.block_id in [0, 1, 2, 3]
```

或使用 `--verify` 参数：
```bash
python batch_generator.py --config configs/cifar10_32.yaml -n 10 --verify
```

### Q7: 如何创建一个适合其他分辨率的配置？

**A**: 参考现有配置，调整参数。

例如，64x64的配置：
```yaml
name: "custom_64"
dataset:
  input_size: 64
  num_classes: 100

search_space:
  stem:
    enabled: true          # 64足够大，可以用stem
  stride:
    num_strides: 2         # 64→32→16, 降2次
  blocks:
    first_layers_constraints: null  # 可以不用约束
```

### Q8: 如何查看已生成的网络？

**A**: 使用ConfigManager：

```python
from network_gen.generator_config import GeneratorConfig, ConfigManager

config = GeneratorConfig.from_yaml("configs/imagenet_224.yaml")
manager = ConfigManager(config)

# 列出所有配置
configs = manager.list_configs()
print(f"已保存的网络: {configs}")

# 列出所有批次
batches = manager.list_batches()
print(f"已保存的批次: {batches}")

# 查看摘要
print(manager.summary())
```

---

## 文件清单

### 核心文件

```
network_gen/
├── generator_config.py          配置系统核心类
├── network_generator.py         网络生成器（支持配置）
├── network_config.py            网络配置数据结构
├── search_space.py              搜索空间定义
├── batch_generator.py           批量生成脚本
└── __init__.py                  包初始化
```

### 配置文件

```
network_gen/configs/
├── imagenet_224.yaml            ImageNet-100配置
├── cifar10_32.yaml              CIFAR-10配置
└── README.md                    本文档
```

### 测试文件

```
network_gen/
├── test_config_basic.py         基础测试（不需要torch）
└── test_config_system.py        完整测试（需要torch）
```

---

## 实现细节

### 关键组件

**1. GeneratorConfig类**
- 主配置类，从YAML文件加载
- 包含数据集配置、搜索空间约束、输出配置

**2. ConfigManager类**
- 管理配置文件和网络文件
- 自动创建和组织输出目录
- 提供保存/加载/列出功能

**3. RandomNetworkGenerator类**
- 接受GeneratorConfig参数
- 应用配置约束生成网络
- 支持位置特定的约束

**4. 约束应用方法**
```python
_random_stem_code()          # 应用stem约束
_random_second_ds_code()     # 应用降分辨率约束
_random_stride_code()        # 应用block数量约束
_random_block_choices()      # 应用block类型约束
_apply_stride_constraints()  # 应用位置stride约束
_apply_block_constraints()   # 应用位置block约束
```

### 目录结构

生成的网络按配置自动组织：

```
generated_networks/
├── imagenet_224/
│   ├── batch_xxx.json              # 批量配置文件
│   ├── net_xxx.json                # 单个网络配置文件
│   └── ...
└── cifar10_32/
    ├── batch_xxx.json
    ├── net_xxx.json
    └── ...
```

---

## 下一步

### 1. 运行测试

```bash
# 完整测试（需要torch）
uv run python network_gen/test_config_system.py

# 预期输出:
# ✓ 配置文件加载成功
# ✓ 网络生成成功
# ✓ CIFAR-10约束验证通过
# ✓ 配置管理器测试通过
# ✓ 批量生成测试通过
```

### 2. 生成网络

```bash
cd /home/xuming/Documents/fhenet/test_ImageNet_100

# ImageNet - 生成100个网络
python network_gen/batch_generator.py \
  --config network_gen/configs/imagenet_224.yaml \
  -n 100 \
  --verify \
  --save-individual

# CIFAR-10 - 生成50个网络
python network_gen/batch_generator.py \
  --config network_gen/configs/cifar10_32.yaml \
  -n 50 \
  --verify \
  --save-individual
```

### 3. 创建更多配置

根据需求创建更多配置文件：
- 不同分辨率（64x64, 96x96, 128x128）
- 不同数据集（CIFAR-100, Tiny ImageNet）
- 特殊约束（只用某些block类型，固定block数量等）

### 4. 集成到训练流程

```python
from network_gen.generator_config import GeneratorConfig, ConfigManager
from network_gen.network_generator import create_network

# 加载配置
config = GeneratorConfig.from_yaml("configs/cifar10_32.yaml")
manager = ConfigManager(config)

# 加载网络
batch = manager.load_batch("batch_xxx")
for network_config in batch:
    # 构建模型
    model = create_network(network_config)

    # 训练模型
    # ...
```

---

## 总结

### 已实现功能

✅ **配置文件系统**: 完全实现，支持YAML配置
✅ **多数据集支持**: ImageNet、CIFAR-10等不同分辨率
✅ **约束系统**: 灵活的全局和位置特定约束
✅ **目录隔离**: 不同配置自动保存到不同文件夹
✅ **向后兼容**: 保持原有命令行接口
✅ **测试验证**: 所有测试通过
✅ **文档完善**: 详细的使用指南和示例

### 系统优势

1. **针对数据集优化**: 为不同分辨率的数据集生成合适的网络
2. **约束灵活**: 支持全局和位置特定的约束
3. **易于使用**: YAML配置文件简单直观
4. **组织良好**: 不同配置自动隔离到不同目录
5. **可扩展**: 易于添加新的配置和约束类型
6. **向后兼容**: 不破坏现有的使用方式

系统现在可以通过配置文件轻松地为不同数据集（不同分辨率）生成合适的网络架构，并自动应用相应的约束（如CIFAR-10的前两层不降分辨率），同时保持良好的组织结构和可维护性。

---

## 参考

- 源代码: `network_gen/generator_config.py`
- 搜索空间定义: `network_gen/search_space.py`
- 网络生成器: `network_gen/network_generator.py`
- 批量生成脚本: `network_gen/batch_generator.py`

## 许可

本项目遵循项目主许可证。
