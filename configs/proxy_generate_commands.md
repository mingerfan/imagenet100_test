# Proxy model generation commands

Use the same NAS/random generator with smaller input constraints to create cheap proxy models for training-technique ablations.

## CIFAR-10 / 32x32 proxy models

```bash
python network_gen/batch_generator.py \
  --config network_gen/configs/cifar10_32.yaml \
  --num 10 \
  --batch-name proxy_cifar10_32 \
  --output generated_networks/proxy_cifar10_32 \
  --save-individual \
  --verify \
  --seed 42
```

Then enable the `json_model_patterns` section in `configs/proxy_smartpaf_cifar10.yaml`.

## CIFAR-100 / 32x32 proxy models

Reuse the CIFAR-10 shape constraints, but train with `--dataset cifar100` and `num_classes: 100`:

```bash
python network_gen/batch_generator.py \
  --config network_gen/configs/cifar10_32.yaml \
  --num 10 \
  --batch-name proxy_cifar100_32 \
  --output generated_networks/proxy_cifar100_32 \
  --save-individual \
  --verify \
  --seed 43
```

Then enable the `json_model_patterns` section in `configs/proxy_smartpaf_cifar100.yaml`.

## Low-resolution ImageNet-100 proxy

Use the built-in config first:

```bash
python train.py \
  --config configs/proxy_smartpaf_imagenet100_96.yaml \
  --dataset imagenet100 \
  --input_size 96 \
  --gpus 1 \
  --no_parallel \
  --force \
  --no_use_amp
```
