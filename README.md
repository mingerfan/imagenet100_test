# ImageNet-100 多模型训练系统

一个模块化、可扩展的深度学习训练框架，支持多GPU并行训练和增量训练。

## 项目结构

```
test_ImageNet_100/
├── configs/                    # 配置文件
│   └── models_list.yaml        # 需要训练的模型列表
├── data/                       # 数据加载模块
│   └── dataset.py              # 数据加载器
├── models/                     # 模型定义
│   ├── registry.py             # 模型注册器
│   └── resnet.py               # ResNet模型定义
├── trainers/                   # 训练器
│   ├── base_trainer.py         # 基础训练器
│   └── multi_gpu_manager.py    # 多GPU管理器
├── utils/                      # 工具函数
│   └── config.py               # 配置管理
├── results/                    # 训练结果（自动生成）
└── train.py                    # 主训练脚本
```

## 核心特性

✅ **模块化设计**：数据、模型、训练完全分离  
✅ **多GPU支持**：自动分配GPU，支持并行训练  
✅ **增量训练**：自动跳过已训练的模型  
✅ **配置驱动**：通过YAML文件管理模型列表  
✅ **预训练权重**：支持使用ImageNet预训练权重  
✅ **混合精度训练**：使用FP16加速训练  
✅ **自动保存**：训练结果自动保存到标准化路径  

## 快速开始

### 0. 快速验证（推荐）

在开始训练之前，强烈建议先运行快速验证脚本：

```bash
# 验证所有模型
uv run python quick_validate.py

# 只验证特定模型
uv run python quick_validate.py --models resnet18

# 使用更大的批次测试
uv run python quick_validate.py --batch_size 64

# 使用内存缓存
uv run python quick_validate.py --use_cache
```

**快速验证功能：**
- ✅ 检查所有模型是否可以正常创建
- ✅ 测试数据加载是否正常
- ✅ 运行前向传播和反向传播
- ✅ 验证GPU内存是否充足
- ✅ **快速失败**：如果有任何模型验证失败，立即停止并报错

**输出示例：**
```
[1/5] 加载配置文件...
✓ 配置加载成功，共 3 个模型待验证

[2/5] 检查GPU...
✓ GPU 0: Tesla V100-SXM2-32GB (31.7 GB)

[3/5] 创建数据加载器...
✓ 训练集加载器创建成功
  批次大小: 32
  训练集批次数: 4030
  验证集批次数: 157

[4/5] 验证所有模型...
✅ resnet18 验证通过
✅ resnet34 验证通过
✅ resnet50 验证通过

[5/5] 验证总结
✅ 所有模型验证通过！(3/3)
🎉 系统健康，可以开始训练
```

### 1. 基本使用

训练配置文件中的所有模型：

```bash
python train.py
```

### 2. 指定GPU

使用特定的GPU：

```bash
python train.py --gpus 0 1
```

### 3. 训练特定模型

只训练指定的模型：

```bash
python train.py --models resnet18 resnet34
```

### 4. 强制重新训练

强制重新训练所有模型（包括已训练的）：

```bash
python train.py --force
```

### 5. 串行训练模式

禁用并行训练：

```bash
python train.py --no_parallel
```

### 6. 自定义配置

使用自定义配置文件：

```bash
python train.py --config my_config.yaml
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `configs/models_list.yaml` |
| `--train_dir` | 训练集目录 | `/home/xuming/Documents/dataset/ImageNet_100/train` |
| `--val_dir` | 验证集目录 | `/home/xuming/Documents/dataset/ImageNet_100/val` |
| `--result_dir` | 结果保存目录 | `./results` |
| `--gpus` | 使用的GPU列表 | `[0, 1, 2, 3]` |
| `--force` | 强制重新训练 | `False` |
| `--no_parallel` | 禁用并行训练 | `False` |
| `--no_cache` | 不使用内存缓存 | `False` |
| `--models` | 只训练指定模型 | `None` |
| `--num_classes` | 类别数量 | `100` |
| `--num_workers` | 数据加载worker数 | `16` |

## 配置文件说明

配置文件使用YAML格式，可以灵活配置多个模型：

```yaml
# 全局设置
global:
  num_classes: 100
  default_epochs: 60
  default_batch_size: 128
  default_learning_rate: 0.001

# 模型列表
models:
  - name: "resnet18"
    params:
      pretrained: true
    epochs: 60
    batch_size: 128
    learning_rate: 0.001
```

### 模型配置参数

| 参数 | 说明 | 必需 |
|------|------|------|
| `name` | 模型名称（必须已注册） | 是 |
| `params` | 模型参数（如pretrained, num_classes） | 否 |
| `epochs` | 训练epoch数 | 否 |
| `batch_size` | 批次大小 | 否 |
| `learning_rate` | 学习率 | 否 |
| `num_workers` | 数据加载worker数 | 否 |

## pretrained 参数说明

`pretrained`参数用于指定是否使用预训练权重：

### 什么是预训练权重？

预训练权重是在大规模数据集（如ImageNet-1K，120万张图片，1000个类别）上预训练好的模型权重。

### 优势

1. **加速收敛**：从已学习的特征开始训练，收敛更快
2. **更高准确率**：通常比从头训练获得更好的性能
3. **节省时间**：减少训练时间
4. **数据不足时特别有效**：在小数据集上效果显著

### 使用示例

```yaml
params:
  pretrained: true    # 使用ImageNet预训练权重
  num_classes: 100    # 修改输出层为100个类别
```

### 对比示例

- `pretrained=true`：在ImageNet-100上达到80.72%（60 epochs）
- `pretrained=false`：可能需要100+ epochs，准确率可能只有70-75%

## 添加新模型

### 方法1：使用已有模型

在配置文件中添加：

```yaml
models:
  - name: "resnet50"  # 已注册的模型
    params:
      pretrained: true
    epochs: 60
```

### 方法2：注册新模型

在`models/`目录下创建新模型文件：

```python
# models/custom_model.py
from .registry import register_model
import torch.nn as nn

@register_model('my_model')
def my_model(num_classes=100, pretrained=True):
    """创建自定义模型"""
    model = ...
    return model
```

然后在`models/__init__.py`中导入：

```python
from .custom_model import my_model
```

## 训练结果

训练完成后，结果保存在`results/`目录下：

```
results/
├── resnet18/
│   ├── best_model.pth          # 最佳模型权重
│   ├── train_history.csv       # 训练历史
│   └── checkpoint_epoch_*.pth  # 检查点
├── resnet34/
│   └── ...
└── resnet50/
    └── ...
```

### 加载训练好的模型

```python
import torch
from models import get_model

# 加载模型
model = get_model('resnet18', num_classes=100)

# 加载权重
checkpoint = torch.load('results/resnet18/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 系统要求

- Python >= 3.8
- PyTorch >= 2.0
- torchvision
- CUDA >= 11.0（如使用GPU）
- PyYAML

## 安装依赖

```bash
pip install torch torchvision pyyaml tqdm
```

## 常见问题

### Q: 如何查看可用的模型？

```python
from models import MODEL_REGISTRY
print(MODEL_REGISTRY.list_models())
```

### Q: 如何调整训练参数？

编辑`configs/models_list.yaml`文件，修改对应模型的参数。

### Q: 如何恢复训练？

系统会自动保存检查点，手动加载checkpoint继续训练。

### Q: 内存不足怎么办？

1. 减小`batch_size`
2. 使用`--no_cache`禁用内存缓存
3. 减少同时训练的模型数量

## 许可证

MIT License