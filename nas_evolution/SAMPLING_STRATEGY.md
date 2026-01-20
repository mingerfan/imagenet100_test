# 分层采样策略说明

## 概述

为了评估NAS搜索指标的准确性，系统实现了分层采样策略，从进化搜索的所有架构历史中采样三类架构：

1. **Top 15**: 最佳的15个架构（AZ-NAS fitness最高）
2. **Middle 15**: 从中间50%随机抽样15个架构
3. **Worst 15**: 从最差25%随机抽样15个架构

总共 **45个架构** 用于后续的训练和评估实验。

## 分层采样逻辑

### 1. Top 15 架构
- 按AZ-NAS fitness降序排序，取前15个
- 代表搜索空间中的最佳架构
- 用于验证NAS搜索是否能准确找到高性能架构

### 2. Middle 15 架构
- 从排名 25%-75% 的架构池中**随机**抽样15个
- 代表中等性能的架构
- 用于验证NAS搜索指标对中等性能架构的区分能力

### 3. Worst 15 架构
- 从排名 75%-100% 的架构池中**随机**抽样15个
- 代表低性能的架构
- 用于验证NAS搜索指标对低性能架构的识别能力

## 输出目录结构

运行进化搜索后，结果保存在 `nas_results/<run_name>/` 目录：

```
nas_results/
└── <run_name>/
    ├── best_models/              # Top 15 最佳架构
    │   ├── rank1_fitness-1.234.json
    │   ├── rank2_fitness-1.456.json
    │   └── ...
    ├── middle_models/            # Middle 15 中等架构
    │   ├── middle_rank1_fitness-2.345.json
    │   ├── middle_rank2_fitness-2.456.json
    │   └── ...
    ├── worst_models/             # Worst 15 最差架构
    │   ├── worst_rank1_fitness-4.567.json
    │   ├── worst_rank2_fitness-4.678.json
    │   └── ...
    ├── evolution.log             # 详细日志
    ├── evolution_stats.json      # 统计历史
    └── checkpoints/              # 检查点文件
```

## 架构文件格式

每个架构保存为JSON文件，包含完整信息：

```json
{
  "category": "top",              // 类别: "top", "middle", "worst"
  "rank_in_category": 1,          // 在该类别中的排名
  "aznas_fitness": -1.234,        // AZ-NAS适应度分数
  "scores": {                     // 所有评估指标
    "expressivity": 19.25,
    "progressivity": -0.18,
    "trainability": -0.30,
    "fhe_latency": 6707024.0,
    "fhe_boot_count": 37,
    "fhe_max_depth": 377,
    "fhe_operation_latency": 2804919.0,
    "fhe_boot_latency": 3827304.0
  },
  "generation": 2,                // 产生的代数
  "config": {                     // 网络配置（用于重建模型）
    "stem_code": 0,
    "second_ds_code": 2,
    "stride_code": 209,
    "ct_policies": ["keep", "half", "keep"],
    "block_choices": [11, 11, 19, 8, 1, ...],
    "blocks": [...]
  }
}
```

## 网络配置选项

系统现在支持多种网络配置，可通过 `--network_config` 参数灵活选择：

### 配置文件

1. **imagenet_224.yaml** （默认）
   - 块数量：10
   - 支持块类型：所有类型（0-21）
   - 用途：通用架构搜索，包含所有块类型

2. **imagenet_224_resnet_style.yaml**（新增）
   - 块数量：10  
   - 支持块类型：ResNet风格块（16-21）
   - 用途：在FHE中文化限制下，只使用ResNet友好的块

3. **imagenet_224_mbconv_style.yaml**（新增）
   - 块数量：6
   - 支持块类型：MBConv风格块（0-15）
   - 用途：在FHE优化下，只使用MBConv块进行轻量化搜索

### 约束条件说明

#### Poly4激活函数约束
- Poly4块**只能**出现在网络的前50%
- 约束采用**分组感知**方式：位置4及以后的块成对共享，替换时整组替换
- 自动在所有生成的网络中强制执行

#### 分组共享一致性
- 位置0-3：独立块选择
- 位置4-9：成对共享（[4,5]必须相同，[6,7]必须相同，[8,9]必须相同）
- 突变操作自动维护这些一致性

#### 块类型过滤（allowed_block_ids）
- ResNet风格配置：仅允许块 16-21（ResNet块，无激活裁剪）
- MBConv风格配置：仅允许块 0-15（MBConv块，支持激活裁剪）
- 突变算子尊重这些限制，生成的架构始终合法

## 使用方法

### 1. 运行进化搜索

```bash
# 小规模测试，使用默认配置 (population=10, generations=5)
uv run python nas_evolution/run_evolution.py --config nas_evolution/evolution_config_test.yaml

# 完整搜索，使用默认配置 (population=200, generations=100)
uv run python nas_evolution/run_evolution.py --config nas_evolution/evolution_config.yaml

# 使用特定网络配置进行搜索
# ResNet风格（10块，块类型16-21）
uv run python nas_evolution/run_evolution.py \
  --config nas_evolution/evolution_config.yaml \
  --network_config network_gen/configs/imagenet_224_resnet_style.yaml

# MBConv风格（6块，块类型0-15）
uv run python nas_evolution/run_evolution.py \
  --config nas_evolution/evolution_config.yaml \
  --network_config network_gen/configs/imagenet_224_mbconv_style.yaml
```

### 2. 进化统计可视化

进化搜索运行完成后，系统自动生成可视化图表保存在 `nas_results/<run_name>/plots/` 目录：

```
nas_results/<run_name>/plots/
├── fitness_progression.png          # Fitness随代数进化曲线
├── zen_score_progression.png        # ZEN分数随代数进化曲线
├── fhe_latency_progression.png      # FHE延迟随代数进化曲线
├── fhe_boot_count_progression.png   # FHE启动次数随代数进化曲线
└── evolution_summary.png            # 2×2总结图（所有指标一览）
```

这些图表帮助了解进化过程中的性能改进轨迹。

### 3. 分析采样结果

```bash
# 分析统计信息并生成可视化
uv run python nas_evolution/analyze_sampling.py nas_results/<run_name>
```

这将生成：
- 统计摘要（打印到终端）
- `stratified_sampling_distributions.png`: 各指标的分布图
- `stratified_sampling_scatter.png`: Fitness vs 其他指标的散点图
- `architectures_for_training.json`: 所有45个架构的配置（用于训练）

### 4. 训练和评估架构

使用 `architectures_for_training.json` 中的配置重建模型并训练：

```python
from network_gen import create_network
import json

# 加载架构配置
with open('nas_results/<run_name>/architectures_for_training.json') as f:
    architectures = json.load(f)

# 训练每个架构
for arch in architectures:
    # 重建模型
    from network_gen.network_config import NetworkConfig
    config = NetworkConfig.from_dict(arch['config'])
    model = create_network(config)

    # 训练模型
    # ... 你的训练代码 ...

    # 记录结果
    category = arch['category']
    aznas_fitness = arch['aznas_fitness']
    # ... 保存训练准确率等 ...
```

## 论文实验分析

### 实验目的
评估NAS搜索指标（零成本代理 + FHE延迟）与实际训练性能的相关性。

### 实验步骤

1. **运行NAS搜索**
   - 完整进化搜索：population=200, generations=100
   - 获得45个采样架构（15 top + 15 middle + 15 worst）

2. **训练所有架构**
   - 在相同训练设置下训练所有45个架构
   - 记录最终测试准确率、训练时间等

3. **相关性分析**
   - 计算AZ-NAS fitness与实际准确率的相关系数（Spearman, Kendall Tau）
   - 分析各个指标（expressivity, progressivity, trainability, FHE latency）的预测能力
   - 验证top架构的实际性能是否优于middle和worst

4. **可视化结果**
   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from scipy.stats import spearmanr, kendalltau

   # 绘制相关性散点图
   fitness = [arch['aznas_fitness'] for arch in results]
   accuracy = [arch['test_accuracy'] for arch in results]

   plt.scatter(fitness, accuracy, c=['green']*15 + ['orange']*15 + ['red']*15)
   plt.xlabel('AZ-NAS Fitness')
   plt.ylabel('Test Accuracy (%)')

   # 计算相关系数
   spearman_corr, _ = spearmanr(fitness, accuracy)
   kendall_corr, _ = kendalltau(fitness, accuracy)

   print(f"Spearman correlation: {spearman_corr:.4f}")
   print(f"Kendall Tau correlation: {kendall_corr:.4f}")
   ```

### 预期结果

如果NAS搜索指标准确：
- ✅ Top 15架构的平均准确率应该显著高于Middle和Worst
- ✅ AZ-NAS fitness与实际准确率应有较强正相关（ρ > 0.6）
- ✅ 各个零成本代理指标应与准确率有合理相关性
- ✅ FHE延迟应与实际推理时间成比例

## 配置参数

### 进化搜索参数

可以在 `evolution_config.yaml` 中修改采样参数：

```yaml
# 分层采样配置
stratified_sampling:
  top_k: 15          # Top架构数量
  middle_k: 15       # Middle架构数量  
  worst_k: 15        # Worst架构数量
  # top_k 不适用于middle/worst（它们是随机采样的）

# 网络生成配置
network_config: network_gen/configs/imagenet_224.yaml  # 默认配置
```

### 运行时覆盖参数

使用CLI参数覆盖配置文件中的设置：

```bash
# 覆盖网络配置
uv run python nas_evolution/run_evolution.py \
  --config nas_evolution/evolution_config.yaml \
  --network_config network_gen/configs/imagenet_224_mbconv_style.yaml
```

### 块类型配置

在网络配置YAML中定义：

```yaml
# network_gen/configs/imagenet_224_resnet_style.yaml
block_count: 10
allowed_block_ids: [16, 17, 18, 19, 20, 21]  # ResNet风格块

# network_gen/configs/imagenet_224_mbconv_style.yaml  
block_count: 6
allowed_block_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # MBConv块
```

## 注意事项

1. **采样数量限制**
   - 如果总架构数量不足，采样数量会自动调整
   - 例如：总共只有30个架构时，无法采样45个

2. **随机性**
   - Middle和Worst的采样是**随机**的（每次运行结果不同）
   - 可以设置random seed保证可重复性

3. **适应度计算**
   - AZ-NAS fitness使用非线性排名聚合
   - fitness值越高越好（负值较小表示更好）
   - fitness = Σ log(Rank(metric)/m)，惩罚任何弱维度

4. **块约束说明**
   - Poly4约束：自动在所有网络中执行，无需手动配置
   - 块类型过滤：通过 `allowed_block_ids` 配置，在生成和突变时强制执行
   - 分组共享：自动维护位置4+的块对一致性

5. **可视化生成**
   - 图表自动保存到 `nas_results/<run_name>/plots/`
   - 需要 matplotlib 库（通常预装）
   - 若matplotlib不可用，演化继续进行但不生成图表

## 相关文件

- `regularized_evolution.py`: 主进化算法，包含 `_get_stratified_sample()` 方法、配置加载和可视化生成
- `mutations.py`: 突变算子，现在支持 `allowed_block_ids` 参数用于块类型过滤
- `network_generator.py`: 网络生成器，实现Poly4约束和分组共享一致性
- `utils.py`: 包含 `EvolutionLogger` 类的 `plot_evolution_stats()` 方法，生成进化统计图表
- `analyze_sampling.py`: 分析脚本，生成采样统计和可视化
- `evolution_config.yaml`: 主进化配置文件
- `network_gen/configs/imagenet_224*.yaml`: 网络配置文件（3个配置选项）

## 参考

- Real et al. "Regularized Evolution for Image Classifier Architecture Search" (2019)
- Abdelfattah et al. "Zero-Cost Proxies for Lightweight NAS" (2021)
- 论文中的AZ-NAS非线性排名聚合方法
