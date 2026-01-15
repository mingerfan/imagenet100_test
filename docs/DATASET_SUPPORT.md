Dataset Support

This project can train on ImageNet-100, ImageNet-1k, CIFAR-10, and CIFAR-100.
Use the `--dataset` flag to select the dataset and configure paths.

Supported values:
- `imagenet100`
- `imagenet1k`
- `cifar10`
- `cifar100`

ImageNet (ImageFolder)
- Requires `--train_dir` and `--val_dir` for ImageNet-1k.
- ImageNet-1k is not downloaded automatically; paths must exist.
- ImageNet-100 defaults:
  - train: `/home/xuming/Documents/dataset/ImageNet_100/train`
  - val: `/home/xuming/Documents/dataset/ImageNet_100/val`

CIFAR (torchvision)
- Uses `--train_dir` as the CIFAR root (defaults to `./data`).
- `--download` allows auto-download if the dataset is missing.
- `--val_dir` defaults to the same root as `--train_dir`.

Memory filesystem
- `--use_memory_fs` (default on) copies ImageFolder datasets into `/dev/shm`.
- Use `--no_memory_fs` to disable.
- CIFAR datasets are already loaded in-memory by torchvision; memory FS is ignored.

NAS input size
- CIFAR uses 32x32 inputs; ImageNet uses 224x224 by default.
- Make sure NAS configs match the dataset input size.
- `--input_size` overrides the default transforms if needed.

Examples

ImageNet-1k (paths required):
```
python train_nas_architectures_multigpu.py \
  --dataset imagenet1k \
  --train_dir /path/to/imagenet/train \
  --val_dir /path/to/imagenet/val
```

CIFAR-10 (download allowed):
```
python train.py --dataset cifar10 --download
```

CIFAR-100 (local data):
```
python train.py --dataset cifar100 --train_dir ./data --val_dir ./data
```
