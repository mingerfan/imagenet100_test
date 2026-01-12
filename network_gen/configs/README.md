# 网络生成器配置文件说明

这个目录包含不同数据集的网络生成器配置文件。每个配置文件定义了针对特定数据集和输入分辨率的搜索空间约束。

## 配置文件列表

- **imagenet_224.yaml**: ImageNet-100数据集配置 (224x224输入)
  - 标准配置，使用完整的搜索空间
  - 激进的降分辨率策略 (224→112→56→28→14→7)

- **cifar10_32.yaml**: CIFAR-10数据集配置 (32x32输入)
  - 针对小分辨率图像的保守策略
  - 禁用激进的stem层
  - 前两个block不降分辨率
  - 只降2次分辨率 (32→16→8)

## 配置文件结构

```yaml
name: "配置名称"
description: "配置描述"

dataset:
  name: "数据集名称"
  num_classes: 分类数量
  input_size: 输入图像大小

search_space:
  ct_slots: 32768                    # CT槽位数
  initial_ct_count: 1                # 初始CT数量

  stem:
    enabled: true/false              # 是否启用stem层
    allowed_codes: [0,1,2,3] 或 null # 允许的stem配置

  second_downsample:
    enabled: true/false              # 是否启用第二次降分辨率
    allowed_codes: [0,1,2,3,4] 或 null

  blocks:
    allowed_block_ids: [0,1,...,23] 或 null  # 允许的block类型
    first_layers_constraints:        # 前几个block的约束
      - position: 0                  # block位置（第几个block）
        stride: 1                    # 强制stride值
        allowed_block_ids: null      # 该位置允许的block类型

  stride:
    allowed_block_counts: [4,6,8,...] 或 null  # 允许的block数量
    num_strides: 3                   # 降分辨率次数

  ct_policies:
    allowed: ["keep", "half"]        # 允许的CT策略

output:
  base_dir: "输出目录路径"
  save_format: "json"
```

## 使用方法

```python
from network_gen.generator_config import GeneratorConfig
from network_gen.network_generator import RandomNetworkGenerator

# 加载配置
config = GeneratorConfig.from_yaml("network_gen/configs/imagenet_224.yaml")

# 使用配置创建生成器
generator = RandomNetworkGenerator(config=config)

# 生成网络
network = generator.generate_random_config()
```

## 约束说明

### Block位置约束 (first_layers_constraints)

`first_layers_constraints` 用于约束网络前几个block的行为：

- `position`: 指定第几个block（从0开始）
  - position=0 表示网络的第1个block
  - position=1 表示网络的第2个block

- `stride`: 强制该block的stride值
  - stride=1: 不降分辨率
  - stride=2: 降分辨率

- `allowed_block_ids`: 该位置允许使用的block类型（0-23）

**示例**：CIFAR-10前两个block不降分辨率
```yaml
first_layers_constraints:
  - position: 0    # 第1个block
    stride: 1      # 保持分辨率
  - position: 1    # 第2个block
    stride: 1      # 保持分辨率
```

### 24种Block类型

系统支持24种预定义的block类型，编号0-23：
- 0-1: BasicBlock (2种激活函数)
- 2-3: BasicSelfGatedBlock (2种激活函数)
- 4-13: BottleneckBlock (5种factor × 2种激活函数)
- 14-23: BottleneckSelfGatedBlock (5种factor × 2种激活函数)

详细信息见 `search_space.py` 中的 `UNIFIED_BLOCKS`。

## 创建自定义配置

1. 复制现有配置文件
2. 修改数据集参数和约束
3. 保存为新的yaml文件
4. 在代码中加载使用

```bash
cp configs/imagenet_224.yaml configs/my_custom.yaml
# 编辑 my_custom.yaml
```
