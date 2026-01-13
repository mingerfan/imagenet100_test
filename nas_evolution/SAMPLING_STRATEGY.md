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

## 使用方法

### 1. 运行进化搜索

```bash
# 小规模测试 (population=10, generations=5)
uv run python nas_evolution/run_evolution.py --config nas_evolution/evolution_config_test.yaml

# 完整搜索 (population=200, generations=100)
uv run python nas_evolution/run_evolution.py --config nas_evolution/evolution_config.yaml
```

### 2. 分析采样结果

```bash
# 分析统计信息并生成可视化
uv run python nas_evolution/analyze_sampling.py nas_results/<run_name>
```

这将生成：
- 统计摘要（打印到终端）
- `stratified_sampling_distributions.png`: 各指标的分布图
- `stratified_sampling_scatter.png`: Fitness vs 其他指标的散点图
- `architectures_for_training.json`: 所有45个架构的配置（用于训练）

### 3. 训练和评估架构

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

可以在 `regularized_evolution.py` 中修改采样数量：

```python
# 在 run() 方法中
stratified_sample = self._get_stratified_sample(
    top_k=15,      # 修改top架构数量
    middle_k=15,   # 修改middle架构数量
    worst_k=15     # 修改worst架构数量
)
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

## 相关文件

- `regularized_evolution.py`: 主进化算法，包含 `_get_stratified_sample()` 方法
- `utils.py`: 保存功能，包含 `save_sampled_architectures()` 函数
- `analyze_sampling.py`: 分析脚本，生成统计和可视化
- `evolution_config.yaml`: 配置文件

## 参考

- Real et al. "Regularized Evolution for Image Classifier Architecture Search" (2019)
- Abdelfattah et al. "Zero-Cost Proxies for Lightweight NAS" (2021)
- 论文中的AZ-NAS非线性排名聚合方法
