# 项目运行约束

## GPU 使用

- 本机有 4 张 V100。
- 默认训练 GPU 为 `1 2 3`，对应第 2/3/4 张 V100。
- physical GPU 0 / 第一张 V100 存在 memory/ECC 风险，默认训练路径会强制过滤 GPU 0。
- 如确实需要使用 GPU 0，必须同时显式传入包含 `0` 的 `--gpus` 列表和 `--allow_gpu0`，并自行承担 ECC 风险。
- `train.py` 会打印 requested GPUs、actual GPUs 和 `CUDA_VISIBLE_DEVICES`，避免 physical/logical GPU 编号混淆。

推荐命令：

```bash
python train.py --gpus 1 2 3
```

不推荐但允许的 GPU0 命令：

```bash
python train.py --gpus 0 --allow_gpu0
```
