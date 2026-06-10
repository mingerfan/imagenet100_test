# 分层采样策略说明

## 概述

为了评估NAS搜索指标的准确性，系统实现了分层采样策略，从进化搜索的所有架构历史中采样三类架构：

1. **Top 15**: 最佳的15个架构（Zen fitness最高）
2. **Middle 15**: 从中间50%随机抽样15个架构
3. **Worst 15**: 从最差25%随机抽样15个架构

总共 **45个架构** 用于后续的训练和评估实验。

## 分层采样逻辑

### 1. Top 15 架构
- 按 Zen fitness 降序排序，取前15个
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
  "zen_fitness": -1.234,          // Zen/FHE适应度分数
  "scores": {                     // 所有评估指标
    "zen_score": 19.25,
    "synflow_score": 123456.0,
    "params": 1200000,
    "flops": 320000000.0,
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

4. **imagenet_224_swish_mbconv.yaml**
   - 块数量：4/6/8/10/12/14/16
   - Stem：固定 `NoSG + Swish`
   - 第二次降采样：`AvgPool`、`NoSG + Swish Conv` 或 `None`
   - Body：仅允许 `MBConv1/4 + Swish`，可带 SE，不含 Poly4/selfgated
   - 用途：Phase 1 结构搜索，先找深度、通道、CT policy 和 MBConv1/4 的权衡

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

# Phase 1 推荐配置：全 Swish、无 selfgated、MBConv1/4
uv run python nas_evolution/run_evolution.py \
  --config nas_evolution/evolution_config_swish_mbconv.yaml \
  --gpus all
```

### 2. 多 GPU 评估

NAS 进化的初始化种群会批量评估，支持按 GPU 并行分发；后续每一代仍按 regularized evolution 的单 offspring 语义串行评估，避免改变选择和老化逻辑。

```bash
# 使用所有 PyTorch 可见 GPU
uv run python nas_evolution/run_evolution.py \
  --config nas_evolution/evolution_config.yaml \
  --gpus all

# 8 卡设备
uv run python nas_evolution/run_evolution.py \
  --config nas_evolution/evolution_config.yaml \
  --gpus 0-7

# 旧机器避开 physical GPU 0
uv run python nas_evolution/run_evolution.py \
  --config nas_evolution/evolution_config.yaml \
  --gpus all \
  --exclude_gpus 0

# 兼容旧单卡写法
uv run python nas_evolution/run_evolution.py \
  --config nas_evolution/evolution_config.yaml \
  --gpu 1
```

配置文件中也可以设置：

```yaml
evaluation:
  gpus: all
  exclude_gpus: []
  parallel_evaluations: true
```

`batch_size` 是每个评估 worker 的 batch size，不是跨 GPU 聚合后的 global batch size。

### 3. 进化统计可视化

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

### 4. 分析采样结果

```bash
# 分析统计信息并生成可视化
uv run python nas_evolution/analyze_sampling.py nas_results/<run_name>
```

这将生成：
- 统计摘要（打印到终端）
- `stratified_sampling_distributions.png`: 各指标的分布图
- `stratified_sampling_scatter.png`: Fitness vs 其他指标的散点图
- `architectures_for_training.json`: 所有45个架构的配置（用于训练）

### 5. CIFAR-100 代理短训

ZCP 在跨网络类型比较时不够稳定，因此当前推荐用 CIFAR-100@224 做离线代理短训，而不是把短训放进 evolution 内循环。

```bash
# 默认选择 top50 + middle20 + worst20，12 epoch 代理评估
uv run python tools/train_nas_architectures.py \
  --nas-results nas_results/swish_mbconv_phase1 \
  --selection top50_middle20_worst20 \
  --dataset cifar100 \
  --input-size 224 \
  --epochs 12 \
  --batch-size 128 \
  --gpus all \
  --download

# 如果要先做更便宜的筛选，可以改成 2 epoch
uv run python tools/train_nas_architectures.py \
  --nas-results nas_results/swish_mbconv_phase1 \
  --selection top50_middle20_worst20 \
  --dataset cifar100 \
  --input-size 224 \
  --epochs 2 \
  --gpus all \
  --download
```

输出：
- `training_results.csv`: `zen_fitness`、`zen_score`、`synflow_score`、`params`、`flops`、`fhe_latency` 与 `best_val_acc`
- `training_summary.json`: 按 category 汇总准确率
- `selected_architectures.json`: 本次短训实际选中的架构

### 6. Phase 2 replacement masks

在 Phase 1 搜出的 plain MBConv 结构上，再离线生成有限数量的替换 mask。默认只改 body block，不碰 stem/second downsample。默认不添加 gated/self-gated 模块，只使用前三个激活替换动作；`gated_lswish` 保留为显式 `--actions` 实验选项。

- `stablepoly4`: Swish `1->0, 3->2, 5->4, 7->6`; ReLU `22->0, 23->2, 24->4, 25->6`
- `hermitepoly4`: `activation_override: poly4_herpn` for Swish/ReLU plain MBConv blocks
- `swish_herpn`: `activation_override: swish_herpn` for Swish plain MBConv blocks only
- `gated_lswish`: Swish `1->9, 3->11, 5->13, 7->15`; ReLU `22->9, 23->11, 24->13, 25->15`

```bash
uv run python tools/nas_replacement_planner.py score-sites \
  --arch nas_results/swish_mbconv_phase1/best_models/rank1_fitness*.json \
  --output results/rank1_replacement_scores.json

uv run python tools/nas_replacement_planner.py generate-masks \
  --arch nas_results/swish_mbconv_phase1/best_models/rank1_fitness*.json \
  --scores results/rank1_replacement_scores.json \
  --output-dir configs/nas_replacement_masks/rank1 \
  --top-site-actions 6 \
  --max-replacements 3 \
  --max-masks 30
```

默认预算：
- 全部 masks 先训 2 epoch
- 用 `promoted8` 选择 8 个训 10 epoch
- 用 `promoted3` 选择 3 个训 20 epoch

一键流程默认不是只按单一“短训最好”排序晋级，而是用
`--replacement-promotion-strategy accuracy_efficiency` 做双分支：
`--replacement-promotion-accuracy-share 0.5` 会让 `promoted8` 拆成 4 个
accuracy-biased 候选和 4 个 efficiency-biased 候选，`promoted3` 拆成 2 个
accuracy-biased 候选和 1 个 efficiency-biased 候选。两个分支都使用
`best_val_acc + weight * fhe_latency_reduction_pct`，accuracy-biased 分支默认用
`--latency-tradeoff-weight 0.1`，efficiency-biased 分支默认用
`--replacement-efficiency-latency-tradeoff-weight 0.3`；两个分支重复时会去重，并按
`--replacement-promotion-metric` 补足 quota。旧的单排序行为可以显式设
`--replacement-promotion-strategy single`。

```bash
uv run python tools/train_nas_architectures.py \
  --json configs/nas_replacement_masks/rank1/*.json \
  --selection all \
  --dataset cifar100 \
  --input-size 224 \
  --epochs 2 \
  --result-dir results/rank1_masks_e2 \
  --gpus all \
  --download

uv run python tools/train_nas_architectures.py \
  --selection promoted8 \
  --training-results results/rank1_masks_e2/training_results.csv \
  --promotion-strategy accuracy_efficiency \
  --promotion-accuracy-share 0.5 \
  --latency-tradeoff-weight 0.1 \
  --efficiency-latency-tradeoff-weight 0.3 \
  --dataset cifar100 \
  --input-size 224 \
  --epochs 10 \
  --result-dir results/rank1_masks_e10 \
  --gpus all \
  --download
```

接受规则：若 `best_acc >= baseline_best_acc - 0.5pp`，保留；或者 `fhe_latency <= 0.9 * baseline_latency` 且 `best_acc >= baseline_best_acc - 1.0pp`，保留。

### 7. 两阶段一键流程、产物位置与大训衔接

推荐使用 `tools/run_nas_two_stage.py` 编排 Phase 1 结构搜索、CIFAR-100@224
代理短训、Phase 2 replacement mask 生成和 `2 -> 10 -> 20` epoch 晋级短训。

```bash
uv run python tools/run_nas_two_stage.py \
  --run-root results/nas_two_stage_swish_mbconv \
  --gpus all \
  --download
```

如果 Phase 1 evolution 已经跑完，可以复用已有目录：

```bash
uv run python tools/run_nas_two_stage.py \
  --nas-results nas_results/swish_mbconv_phase1 \
  --run-root results/nas_two_stage_from_existing \
  --gpus all \
  --download
```

所有训练入口都支持手动指定数据集路径：

```bash
uv run python tools/run_nas_two_stage.py \
  --dataset imagenet100 \
  --train-dir /path/to/imagenet100/train \
  --val-dir /path/to/imagenet100/val \
  --input-size 224 \
  --gpus all
```

产物默认位置：

- Phase 1 搜索结构：
  `results/nas_two_stage/phase1_evolution/best_models/*.json`
- Phase 1 代理短训结果：
  `results/nas_two_stage/phase1_proxy/training_results.csv`
- Phase 2 replacement mask 结构：
  `results/nas_two_stage/replacement_masks/<source_arch_name>/*.json`
- Phase 2 晋级短训结果：
  `results/nas_two_stage/replacement_train_e2/training_results.csv`、
  `results/nas_two_stage/replacement_train_e10/training_results.csv`、
  `results/nas_two_stage/replacement_train_e20/training_results.csv`
- 流程清单：
  `results/nas_two_stage/two_stage_manifest.json`

如果使用自定义 `--run-root`，上述路径都在该目录下。直接运行
`nas_evolution/run_evolution.py` 时，搜索结构保存在配置的
`logging.output_dir/best_models/*.json`；推荐配置
`nas_evolution/evolution_config_swish_mbconv.yaml` 默认写入
`nas_results/swish_mbconv_phase1/best_models/*.json`。

Phase 1 代理短训默认使用 `swish_proxy`，不启用 SmartPAF/AutoFHE。
Phase 2 replacement mask 默认使用两个 no-PAT/no-AT 的正向 preset：
`replacement_autofhe_degree2` 和 `replacement_learned_slow_scale`。AT/PAT 只作为
显式实验选项保留，不走默认路径。

搜索出的 JSON 结构可以直接通过 `nas-json` 接入大规模训练。最终候选可以写入
普通训练 YAML：

```yaml
json_models:
  - name: "final-nas-candidate"
    class: "nas-json"
    json_path: "results/nas_two_stage/replacement_masks/<source>/<candidate>.json"
    params:
      num_classes: 100
      pretrained: false
    epochs: 60
    batch_size: 128
    learning_rate: 0.0007
```

然后启动 ImageNet-100@224 大训：

```bash
uv run python train.py \
  --config configs/final_nas_train.yaml \
  --dataset imagenet100 \
  --train_dir /path/to/imagenet100/train \
  --val_dir /path/to/imagenet100/val \
  --input_size 224 \
  --gpus all
```

也可以直接短训或验证单个 JSON：

```bash
uv run python tools/train_nas_architectures.py \
  --json results/nas_two_stage/replacement_masks/<source>/<candidate>.json \
  --selection all \
  --dataset imagenet100 \
  --train-dir /path/to/imagenet100/train \
  --val-dir /path/to/imagenet100/val \
  --input-size 224 \
  --gpus all
```

### 8. 手动训练和评估架构

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
    zen_fitness = arch['zen_fitness']
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
   - 计算 Zen fitness、ZEN score、SynFlow、FHE latency 与实际准确率的相关系数（Spearman, Kendall Tau）
   - 分析结构复杂度指标（params、FLOPs）和 FHE latency 的预测能力
   - 验证top架构的实际性能是否优于middle和worst

4. **可视化结果**
   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from scipy.stats import spearmanr, kendalltau

   # 绘制相关性散点图
   fitness = [arch['zen_fitness'] for arch in results]
   accuracy = [arch['test_accuracy'] for arch in results]

   plt.scatter(fitness, accuracy, c=['green']*15 + ['orange']*15 + ['red']*15)
   plt.xlabel('Zen Fitness')
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
- ✅ Zen fitness与实际准确率应有较强正相关（ρ > 0.6）
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
   - Zen fitness使用 ZEN score 与 FHE latency multiplier
   - fitness值越高越好
   - 延迟 multiplier 有意保留“过小模型惩罚”，不是单调奖励更小 latency

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
- `mutations.py`: 突变算子，支持 `allowed_block_ids`、stem、second downsample 和 CT policy 限制
- `network_generator.py`: 网络生成器，实现Poly4约束和分组共享一致性
- `utils.py`: 包含 `EvolutionLogger` 类的 `plot_evolution_stats()` 方法，生成进化统计图表
- `analyze_sampling.py`: 分析脚本，生成采样统计和可视化
- `evolution_config.yaml`: 主进化配置文件
- `network_gen/configs/imagenet_224*.yaml`: 网络配置文件
- `tools/train_nas_architectures.py`: NAS JSON 离线短训入口
- `tools/nas_replacement_planner.py`: Phase 2 replacement mask 生成入口

## 参考

- Real et al. "Regularized Evolution for Image Classifier Architecture Search" (2019)
- Abdelfattah et al. "Zero-Cost Proxies for Lightweight NAS" (2021)
- Lin et al. "Zen-NAS: A Zero-Shot NAS for High-Performance Deep Image Recognition" (2021)
