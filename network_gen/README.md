# 网络生成器配置系统

一个灵活的配置系统，支持通过YAML文件控制神经网络架构的生成，专为不同数据集和输入分辨率优化。

## 主要特性

- ✅ **配置文件控制**: 通过YAML文件定义搜索空间约束
- ✅ **多数据集支持**: 针对不同分辨率的数据集使用不同配置
- ✅ **目录隔离**: 不同配置的网络自动保存到不同文件夹
- ✅ **灵活约束**: 支持控制stem层、前几层的stride、block类型选择等
- ✅ **位置特定约束**: 可以强制指定位置的block行为
- ✅ **向后兼容**: 仍然支持纯命令行参数方式

---

## 快速开始

### 1. 命令行批量生成

#### ImageNet-100 (224x224)
```bash
python batch_generator.py --config configs/imagenet_224.yaml -n 100 --verify
```

#### CIFAR-10 (32x32)
```bash
python batch_generator.py --config configs/cifar10_32.yaml -n 50 --verify
```

### 2. Python代码中使用

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
batch = generator.generate_batch(num_configs=50, batch_name="my_batch")

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
- **Stem层**: 启用
- **降分辨率次数**: 3次
- **输出目录**: `generated_networks/imagenet_224/`

#### CIFAR-10配置 (`configs/cifar10_32.yaml`)

针对小分辨率图像的保守策略：
- **输入分辨率**: 32x32
- **类别数**: 10
- **Stem层**: 禁用
- **前两层约束**: 强制stride=1（不降分辨率）
- **降分辨率次数**: 2次
- **Block数量**: 6, 8, 10, 12
- **输出目录**: `generated_networks/cifar10_32/`

### 配置文件结构

```yaml
name: "配置名称"
description: "配置描述"

# 数据集配置
dataset:
  name: "数据集名称"
  num_classes: 100
  input_size: 224

# 搜索空间配置
search_space:
  ct_slots: 32768
  initial_ct_count: 1

  # Stem层配置
  stem:
    enabled: true
    allowed_codes: null  # null表示允许所有

  # 第二次降分辨率配置
  second_downsample:
    enabled: true
    allowed_codes: null

  # Block配置
  blocks:
    allowed_block_ids: null  # 允许的block ID [0-23]
    first_layers_constraints:  # 前几层的特殊约束
      - position: 0
        stride: 1
        allowed_block_ids: null

  # Stride配置
  stride:
    allowed_block_counts: null  # 允许的block数量
    num_strides: 3

  # CT策略配置
  ct_policies:
    allowed: ["keep", "half"]

# 输出配置
output:
  base_dir: "generated_networks/imagenet_224"
  save_format: "json"
```

### 配置约束详解

#### 1. 限制前N层不降分辨率

```yaml
blocks:
  first_layers_constraints:
    - position: 0  # 第1个block
      stride: 1    # 强制stride=1，不降分辨率
    - position: 1  # 第2个block
      stride: 1

stride:
  allowed_block_counts: [6, 8, 10, 12]
  num_strides: 2
```

**说明**: `position` 指定网络中第几个block（从0开始），不是block内部的层。

#### 2. 只允许特定的block类型

```yaml
blocks:
  allowed_block_ids: [0, 1, 4, 5, 6, 7, 8, 9]  # 只允许basic和部分bottleneck
```

#### 3. 禁用stem和第二次降分辨率

```yaml
stem:
  enabled: false

second_downsample:
  enabled: false
```

---

## 命令行参数

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

---

## Block类型参考

系统支持24种预定义的block类型（ID 0-23）：

### Basic Blocks (0-3)
- **0-1**: BasicBlock (Poly4, Swish)
- **2-3**: BasicSelfGatedBlock (Poly4, Swish)

### Bottleneck Blocks (4-13)
- **4-13**: BottleneckBlock (5种factor × 2种激活)
  - Factor: 0.25, 0.5, 1.0, 1.5, 2.0
  - Activation: Poly4, Swish

### Bottleneck Self-Gated Blocks (14-23)
- **14-23**: BottleneckSelfGatedBlock (5种factor × 2种激活)

详细信息见 `search_space.py` 中的 `UNIFIED_BLOCKS`。

---

## 常见问题

### Q1: 配置文件和命令行参数同时使用会怎样？

**A**: 优先使用配置文件的设置。但是`--output`参数可以覆盖配置文件中的输出目录。

### Q2: 如何确保不同配置生成的网络不会混淆？

**A**: 在配置文件的`output.base_dir`中指定不同的目录即可。

### Q3: first_layers_constraints中的position是什么意思？

**A**: `position`指的是网络中第几个block（从0开始），**不是block内部的层**。
- `position=0`: 网络的第1个block
- `position=1`: 网络的第2个block

### Q4: 如何禁用某些功能（如stem层）？

**A**: 在配置文件中设置`enabled: false`。

### Q5: 配置约束会影响搜索空间大小吗？

**A**: 会。约束越多，搜索空间越小。示例：
- **限制block类型**: 从24种减少到指定的几种
- **限制block数量**: 从7种减少到指定的几种
- **前几层的stride约束**: 减少stride组合的可能性

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
└── cifar10_32.yaml              CIFAR-10配置
```

### 测试文件

```
network_gen/
└── test_config_system.py        完整测试（需要torch）
```

---

## 测试

运行测试确保系统正常工作：

```bash
python network_gen/test_config_system.py
```

预期输出:
```
✓ 配置文件加载成功
✓ 网络生成成功
✓ CIFAR-10约束验证通过
✓ 配置管理器测试通过
✓ 批量生成测试通过
```

---

## 创建自定义配置

### 方法1: 从现有配置修改

```bash
# 复制现有配置
cp network_gen/configs/imagenet_224.yaml network_gen/configs/my_custom.yaml

# 编辑配置文件
# 修改 dataset, search_space, output 等

# 使用自定义配置
python batch_generator.py --config network_gen/configs/my_custom.yaml -n 100
```

### 方法2: 在代码中动态创建

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
            allowed_block_ids=[0, 1, 2, 3],
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

---

## 使用示例

### 示例1: 批量生成并验证

```bash
# 使用配置文件
python batch_generator.py --config configs/imagenet_224.yaml -n 100 --verify

# 覆盖输出目录
python batch_generator.py --config configs/cifar10_32.yaml -n 50 -o my_output/

# 保存单独文件
python batch_generator.py --config configs/imagenet_224.yaml -n 20 --save-individual
```

### 示例2: 加载和分析已生成的网络

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

## 目录结构

生成的网络按配置自动组织：

```
generated_networks/
├── imagenet_224/          # ImageNet配置的网络
│   ├── batch_xxx.json
│   └── net_xxx.json
└── cifar10_32/            # CIFAR-10配置的网络
    ├── batch_xxx.json
    └── net_xxx.json
```

---

## 参考

- 配置系统: `generator_config.py`
- 搜索空间定义: `search_space.py`
- 网络生成器: `network_generator.py`
- 批量生成脚本: `batch_generator.py`
