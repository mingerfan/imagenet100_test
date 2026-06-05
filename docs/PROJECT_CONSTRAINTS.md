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

- 当前推荐两阶段流程：Phase 1 使用 `nas_evolution/evolution_config_swish_mbconv.yaml` 做全 Swish、无 selfgated、MBConv1/4 结构搜索；Phase 2 使用 `tools/nas_replacement_planner.py` 离线生成有限 replacement masks。
- NAS evolution 的 `evaluation.batch_size` 是单个评估 worker 的 batch size；短训工具 `tools/train_nas_architectures.py` 的 `--batch-size` 也是单 GPU worker batch size。
- 默认代理训练使用 CIFAR-100@224；ImageNet-100 留作后期筛选，不建议放进 evolution 内循环。
- replacement mask 默认只改 body blocks，不改 stem/second downsample；v1 支持 `stablepoly4`、`hermitepoly4`、`swish_herpn`、`gated_lswish`，不生成 `gated_poly4`。
