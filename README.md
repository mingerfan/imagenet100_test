# ImageNet-100 多模型训练系统

一个模块化、可扩展的深度学习训练框架，支持多GPU并行训练和增量训练。

**新增特性：** ✨ 支持内存文件系统（/dev/shm）加速数据加载，内存占用降低75%！

## 📚 文档导航

- **[完整文档索引](docs/INDEX.md)** - 查看所有文档
- **[测试文档](test/README.md)** - 测试脚本使用指南
- **[FHE 统计](fhe_statistics/README.md)** - FHE 统计模块文档
- **[项目总结](docs/PROJECT_SUMMARY.md)** - 项目架构和重构历史
- **[优化器改进](docs/OPTIMIZER_IMPROVEMENTS.md)** - 优化器配置指南
- **[正则匹配](docs/REGEX_MATCHING_GUIDE.md)** - 正则表达式匹配模型

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

默认使用所有 PyTorch 可见 GPU；在 4 卡机器上是 `0 1 2 3`，在 8 卡机器上是 `0 1 2 3 4 5 6 7`。如果某台旧机器需要避开 GPU 0，可以显式指定 `--exclude_gpus 0` 或只选择 `--gpus 1 2 3`。

```bash
python train.py --gpus all
python train.py --gpus 0-7
python train.py --exclude_gpus 0
```

并行训练是“多个模型/配置并行，各占一张 GPU”，不是单模型 DDP。配置里的 `batch_size` 和 `num_workers` 都是单个 GPU worker 的值；8 个模型并行时总数据加载 worker 数大约是 `8 * num_workers`。

### 2.1 两阶段 NAS 搜索与代理短训

当前推荐流程是先做结构搜索，再做有限 replacement mask 筛选。一键入口会按以下顺序执行：
Phase 1 evolution -> plain MBConv 架构代理短训 -> promoted 架构生成 replacement masks -> `2 -> 10 -> 20` epoch mask 晋级训练。

```bash
uv run python tools/run_nas_two_stage.py \
  --run-root results/nas_two_stage_swish_mbconv \
  --gpus all \
  --download
```

如果 Phase 1 evolution 已经跑完，可以从已有结果继续：

```bash
uv run python tools/run_nas_two_stage.py \
  --nas-results nas_results/swish_mbconv_phase1 \
  --run-root results/nas_two_stage_from_existing \
  --gpus all \
  --download
```

Phase 1 的代理短训默认按 profile 自动选择普通 proxy preset：Swish profile 用 `swish_proxy`，ReLU profile 用 `relu_proxy`，都不启用 SmartPAF/AutoFHE。replacement mask 训练默认自动选择两个有正向证据的 no-PAT/no-AT preset：2 epoch 筛选用 `replacement_autofhe_degree2`（learned scale + CT + degree2/output_scale0.2 + progressive），10/20 epoch 晋级用 `replacement_learned_slow_scale`（在前者基础上用 `poly_scale_lr_mult=0.1` 控制 scale）。AT/PAT 保留为显式实验选项，不走默认路径。

```bash
uv run python tools/run_nas_two_stage.py \
  --nas-results nas_results/swish_mbconv_phase1 \
  --replacement-training-preset replacement_autofhe_degree2 \
  --gpus all \
  --download
```

Phase 2 默认只改 body blocks，不改 stem 和第二次降采样。默认候选动作是 `stablepoly4`、`hermitepoly4`、`swish_herpn`、`gated_lswish`；`swish_herpn` 只用于 Swish body blocks，其余动作同时支持 Swish/ReLU plain MBConv。第一版不生成 `gated_poly4`。mask 训练建议按 `2 -> 10 -> 20` epoch 晋级：先训练全部 masks 2 epoch，再用 `promoted8` 选前 8 个训 10 epoch，最后用 `promoted3` 训 20 epoch。

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
| `--gpus` | 使用的GPU列表/范围，支持 `all`、`0-7`、`1,2,3` | `all visible` |
| `--exclude_gpus` | 按physical GPU ID排除设备 | `None` |
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

## 内存优化方案

### 问题背景

在多GPU并行训练时，每个GPU都会独立缓存完整的数据集，导致内存占用过高：

```
训练集：128,982张 × 0.5MB ≈ 64GB
验证集：5,000张 × 0.5MB ≈ 2.5GB

并行训练4个模型：
- 每个模型: 64GB + 2.5GB = 66.5GB
- 总计: 4 × 66.5GB = 266GB ❌ 超出系统内存
```

### 解决方案：内存文件系统（/dev/shm）

将数据集复制到内存文件系统，所有GPU进程从同一个内存位置读取：

```
磁盘原始数据 → 复制一次到 /dev/shm → 所有GPU进程独立创建DataLoader并读取
```

**工作原理：**

```python
# 每个GPU进程（独立）
进程 A: DataLoader A → 读取 /dev/shm/imagenet100/train
进程 B: DataLoader B → 读取 /dev/shm/imagenet100/train  # 独立实例
进程 C: DataLoader C → 读取 /dev/shm/imagenet100/train  # 独立实例
进程 D: DataLoader D → 读取 /dev/shm/imagenet100/train  # 独立实例
```

**关键点：**
- 每个进程创建独立的DataLoader实例（线程安全）
- 所有DataLoader从同一个内存文件系统路径读取
- 内存文件系统本身就实现了数据共享
- 无需任何"共享DataLoader"机制

**优势：**
- ✅ 只存储1份数据（64GB）
- ✅ 所有GPU进程共享同一数据源
- ✅ 读写速度接近内存（约50GB/s）
- ✅ 每个进程独立的DataLoader，线程安全
- ✅ 无需修改现有代码
- ✅ 内存需求降低75%

### 使用方法

#### 方式1：自动使用（推荐）

默认启用，无需任何操作：

```bash
python train.py
```

系统会自动：
1. 检查 `/dev/shm` 是否可用
2. 检查容量是否足够（需要 >64GB）
3. 自动复制数据集到内存FS
4. 所有训练进程共享读取

#### 方式2：显式启用

```bash
python train.py --use_memory_fs
```

#### 方式3：禁用内存FS

如果内存不足或遇到问题：

```bash
python train.py --no_cache  # 禁用所有缓存
```

### 性能对比

| 方案 | 内存占用 | 读取速度 | 1 epoch时间 | 60 epochs时间 |
|------|---------|---------|------------|--------------|
| 完全内存缓存 | 266GB | 极快 | 15s | 15min |
| **内存文件系统** | **64GB** | **很快** | **20s** | **20min** |
| 磁盘 + no_cache | <10GB | 中等 | 3.5min | 3.5h |

### 工作原理

**第一次运行：**
```python
1. 检查 /dev/shm 是否存在
2. 检查容量是否足够
3. 自动复制数据集（约30秒）
4. 后续训练直接从内存FS读取
```

**后续运行：**
```python
1. 检查数据是否已复制到 /dev/shm
2. 如果已存在，直接使用（秒级）
3. 如果不存在（重启后），自动重新复制
```

### 注意事项

1. **容量要求**：需要 `/dev/shm` 容量 > 64GB
   ```bash
   # 检查容量
   df -h /dev/shm
   
   # 如果不足，可以增加（需要root权限）
   sudo mount -o remount,size=128G /dev/shm
   ```

2. **数据持久性**：重启后数据会丢失，会自动重新复制

3. **多用户环境**：如果多用户共享系统，建议使用独立目录
   ```python
   manager = MemoryFSManager(
       source_path="/path/to/dataset",
       shm_name=f"imagenet100_{os.getlogin()}"  # 用户专属
   )
   ```

### 手动管理（可选）

#### 手动复制到内存FS

```bash
# 复制数据集
cp -r /home/xuming/Documents/dataset/ImageNet_100 /dev/shm/

# 验证
ls -lh /dev/shm/ImageNet_100/
```

#### 清理内存FS

```python
from data.memory_fs import MemoryFSManager

manager = MemoryFSManager("/path/to/dataset")
manager.cleanup()  # 清理 /dev/shm/imagenet100
```

### 常见问题

**Q: 使用内存FS后训练变慢了？**
- 可能是 `/dev/shm` 容量不足，检查 `df -h /dev/shm`
- 可能是复制过程中，等待第一次运行完成

**Q: 如何禁用内存FS？**
```bash
python train.py --no_cache  # 完全禁用
```

**Q: 内存FS和内存缓存可以同时使用吗？**
- 不建议同时使用。内存FS已经很快，不需要额外的内存缓存
- 系统会自动禁用内存缓存：`use_cache=False` 当 `use_memory_fs=True`

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
