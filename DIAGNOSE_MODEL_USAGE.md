# 如何使用诊断脚本

## 问题：模型类别数不匹配

如果遇到错误：
```
RuntimeError: Error(s) in loading state_dict for GeneratedNetwork:
        size mismatch for fc.weight: copying a param with shape torch.Size([1000, 668]) from checkpoint, 
        the shape in current model is torch.Size([100, 668]).
```

这说明：
- **checkpoint中的模型是1000类** (ImageNet-1k)
- **配置文件中写的是100类** (ImageNet-100)

## 解决方案

脚本已经修复，会自动检测和调整。

### 用法 1: ImageNet-1k 模型（1000类）

如果你的模型是在ImageNet-1k上训练的：

```bash
python diagnose_with_model.py \
    --model_path /path/to/best_model.pth \
    --config_path /path/to/config.json \
    --train_dir /path/to/ImageNet-1k/train \
    --val_dir /path/to/ImageNet-1k/val
```

脚本会自动检测到1000类，并使用相应的数据集。

### 用法 2: ImageNet-100 模型（100类）

如果你的模型是在ImageNet-100上训练的：

```bash
python diagnose_with_model.py \
    --model_path /path/to/best_model.pth \
    --config_path /path/to/config.json \
    --train_dir /path/to/ImageNet-100/train \
    --val_dir /path/to/ImageNet-100/val
```

脚本会自动检测到100类。

## 自动检测功能

脚本现在会：

1. **自动检测模型类别数**: 从checkpoint的fc.weight形状读取
2. **自动调整配置**: 如果配置文件中的num_classes不匹配，自动修正
3. **验证数据集匹配**: 确保数据集类别数与模型一致
4. **给出清晰提示**: 如果不匹配，会告诉你问题所在

## 输出示例

```
================================================================================
带模型的诊断 - 检查训练集真实准确率
================================================================================

使用设备: cuda

预加载checkpoint以检测配置...
  检测到模型类别数: 1000
  → 这是ImageNet-1k模型 (1000类)

创建数据加载器（使用 imagenet1k 数据集）...
  训练集: /path/to/ImageNet-1k/train
  验证集: /path/to/ImageNet-1k/val
  训练集样本数: 1281167
  验证集样本数: 50000
  数据集类别数: 1000

加载模型配置和权重...
  从配置文件加载: /path/to/config.json
  ⚠️  检测到类别数不匹配:
     配置文件: 100 类
     checkpoint: 1000 类
  → 自动调整为 1000 类
  ✓ 模型加载成功

评估: 训练集（用验证transforms）
...
```

## 注意事项

1. **确保数据集路径正确**: train_dir和val_dir必须指向正确的数据集
2. **类别数必须匹配**: 模型和数据集的类别数必须一致
3. **配置文件可选**: 如果checkpoint中有config，可以不提供config_path

## 你的情况

从你的错误信息看：
- 模型是**1000类**（ImageNet-1k）
- 但提供的是**ImageNet-100**的数据集路径（只有100类）

需要：
1. 使用ImageNet-1k的数据集路径，或
2. 如果只是想测试，可以在ImageNet-100上评估（但准确率会很低）

建议运行：
```bash
python diagnose_with_model.py \
    --model_path /root/autodl-tmp/imagenet100_test/nas_results/trained_models/best/rank10_fitness116.8408/best_model.pth \
    --config_path /root/autodl-tmp/imagenet100_test/nas_results/best_models/rank10_fitness116.8408.json \
    --train_dir /path/to/ImageNet-1k/train \
    --val_dir /path/to/ImageNet-1k/val
```

确保提供正确的ImageNet-1k路径。
