# 项目运行约束

## GPU 使用

- 训练代码和 NAS 进化评估默认使用所有 PyTorch 可见 GPU；4 卡机器默认使用 `0 1 2 3`，8 卡机器默认使用 `0 1 2 3 4 5 6 7`。
- 如某台旧设备的 physical GPU 0 存在 memory/ECC 风险，使用 `--exclude_gpus 0` 或显式 `--gpus 1 2 3` 避开。
- `--exclude_gpus` 使用 physical GPU ID；`train.py` 会打印 requested GPUs、actual GPUs、`CUDA_VISIBLE_DEVICES` 和 logical->physical 映射，避免编号混淆。

推荐命令：

```bash
python train.py
python train.py --gpus all
python train.py --gpus 0-7
```

旧机器避开 GPU0：

```bash
python train.py --exclude_gpus 0
python train.py --gpus 1 2 3
```

NAS 进化同样支持：

```bash
python nas_evolution/run_evolution.py --gpus all
python nas_evolution/run_evolution.py --gpus 0-7
python nas_evolution/run_evolution.py --gpus all --exclude_gpus 0
```

## NAS 搜索约束

- 当前推荐两阶段流程：优先使用 `tools/run_nas_two_stage.py` 一键编排。Phase 1 默认使用 `nas_evolution/evolution_config_swish_mbconv.yaml` 做全 Swish、无 selfgated、MBConv1/4 结构搜索；需要 ReLU 对齐基线时使用 `nas_evolution/evolution_config_relu_mbconv.yaml`。Phase 2 对 promoted 架构生成有限 replacement masks 并按 `2 -> 10 -> 20` epoch 晋级训练。
- NAS evolution 的 `evaluation.batch_size` 是单个评估 worker 的 batch size；短训工具 `tools/train_nas_architectures.py` 的 `--batch-size` 也是单 GPU worker batch size。
- 默认代理训练使用 CIFAR-100@224；ImageNet-100 留作后期筛选，不建议放进 evolution 内循环。
- Phase 1 代理短训按 profile 使用普通 proxy preset：Swish profile 为 `swish_proxy`，ReLU profile 为 `relu_proxy`，不启用 SmartPAF/AutoFHE；replacement mask 默认使用两个正向 no-PAT/no-AT preset：`replacement_autofhe_degree2` 和 `replacement_learned_slow_scale`。AT/PAT 只作为显式实验选项。
- replacement mask 默认只改 body blocks，不改 stem/second downsample；支持 `stablepoly4`、`hermitepoly4`、`swish_herpn`、`gated_lswish`，其中 `swish_herpn` 只用于 Swish body blocks，其余动作支持 Swish/ReLU plain MBConv；不生成 `gated_poly4`。
- Phase 2 replacement mask 晋级默认使用 `accuracy_efficiency` 双分支策略：每个 `promotedN` quota 默认 50% 按 accuracy-biased 加权分支选，剩余按 efficiency-biased 加权分支选；两个分支都使用 `best_val_acc + weight * fhe_latency_reduction_pct`，默认权重分别为 0.1 和 0.3，去重后用原始 promotion metric 补齐；需要旧行为时显式传 `--replacement-promotion-strategy single`。
