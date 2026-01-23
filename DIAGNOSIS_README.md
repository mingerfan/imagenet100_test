# 训练问题诊断工具

当遇到"训练准确率很高但验证准确率很低"的问题时，使用这些工具快速定位根本原因。

## 问题描述

- **训练准确率**: 93%
- **验证准确率**: 54%
- **差距**: 39个百分点 ❌

这是典型的过拟合或数据问题，需要系统性诊断。

## 诊断工具

### 1. `quick_diagnosis.py` - 快速诊断（10分钟）

**最推荐先运行这个！** 检查3个最关键的问题：

```bash
python quick_diagnosis.py
```

**检查内容**:
- ✓ 验证集前16个样本的标签是否正确
- ✓ train/val的class_to_idx是否一致（**最致命**）
- ✓ 数据归一化参数是否正确

### 2. `diagnose_with_model.py` - 带模型诊断

检查"训练集用验证transforms的真实准确率"：

```bash
python diagnose_with_model.py \
    --model_path nas_results/trained_models/best/model_xxx/best_model.pth \
    --config_path nas_results/best_models/model_xxx.json
```

**检查内容**:
- ✓ 训练集用验证transforms的准确率（验证93%是否真实）
- ✓ 训练/验证差距分析
- ✓ 过拟合程度诊断

### 3. `diagnose_training_issue.py` - 完整诊断

最全面的诊断（需要更多时间）：

```bash
python diagnose_training_issue.py
```

## GPT建议的诊断步骤

### 检查1: 验证模式和Transforms ✓

**问题**: 验证时是否用了正确的模式和transforms？

**验证**:
- model.eval() ✓ 已确认
- torch.no_grad() ✓ 已确认
- 验证transforms: Resize(256) + CenterCrop(224) + Normalize ✓
- 训练/验证的normalize参数一致 ✓

**结论**: 这个没问题

### 检查2: 训练准确率93%是否可信 ⚠️

**问题**: 报告的93%可能是统计口径问题（如使用了mixup/cutmix）

**快速验证**:
```bash
python diagnose_with_model.py --model_path <your_checkpoint>
```

**判断标准**:
- 如果train_eval_acc > 85%: 93%可信，问题在别处
- 如果train_eval_acc < 70%: 93%不可信，统计有问题

### 检查3: 类别索引映射 ❌ **最致命**

**问题**: ImageNet最容易踩的坑 - train/val的class_to_idx不一致

**快速验证**:
```bash
python quick_diagnosis.py
```

**常见原因**:
- ImageNet目录结构不标准
- train/val使用了不同的类别排序
- 手动整理数据集时弄乱了映射

**判断标准**:
- 如果class_to_idx不一致 → **这就是问题根源！**
- 症状: 训练能到90%+，但验证像"随机错位"，卡在很低水平

### 检查4: 数据增强问题 ✓ 已修复

**问题**: 数据增强过强导致训练/验证分布差异大

**已修复**:
- ✓ 将RandomResizedCrop从scale=(0.8, 1.0)改为(0.08, 1.0)
- ✓ 移除了RandomRotation（对ImageNet效果不好）
- ✓ 减弱了ColorJitter强度

## 可能的根本原因（按概率排序）

### 1. 类别索引映射不一致 (70%概率) ❌❌❌

**症状**: 训练很高，验证很低且不提升

**诊断**: 
```bash
python quick_diagnosis.py
# 查看 "检查 2: 类别索引映射一致性"
```

**修复**: 确保train/val目录结构完全一致

### 2. 数据增强过强 (20%概率) ✓ 已修复

**症状**: 训练高，验证低但会慢慢提升

**修复**: 已调整数据增强参数

### 3. 过拟合 (10%概率) ⚠️

**症状**: 训练/验证都在合理范围，但差距大

**修复方案**:
- 增强正则化（增加weight decay）
- 减少训练epoch（25-30即可）
- 添加Dropout
- 使用Label Smoothing

## 推荐行动步骤

### 立即执行（10分钟内）

1. **运行快速诊断**:
   ```bash
   python quick_diagnosis.py
   ```

2. **重点查看**:
   - 检查2的结果（class_to_idx是否一致）
   - 检查1的结果（前16个样本标签是否正确）

3. **如果class_to_idx不一致**:
   - 这就是问题根源！
   - 需要重新整理数据集或统一映射

### 如果有模型（额外5分钟）

4. **运行带模型诊断**:
   ```bash
   python diagnose_with_model.py \
       --model_path <your_checkpoint> \
       --config_path <your_config>
   ```

5. **查看train_eval_acc**:
   - 如果 > 85%: 数据问题（class_to_idx）
   - 如果 < 70%: 统计问题（报告的93%不准）

### 重新训练（如果数据增强是问题）

6. **使用修复后的代码重新训练**:
   ```bash
   python train_nas_architectures_multigpu.py \
       --epochs 25 \
       --learning_rate 0.0005
   ```

## 预期正常水平

对于ImageNet100:

- **训练准确率**: 75-85%
- **验证准确率**: 70-80%
- **差距**: 5-10个百分点

**54%的验证准确率明显不正常！**

## 常见错误和修复

### 错误1: class_to_idx不一致

```python
# 错误示例
train: {'n01440764': 0, 'n01443537': 1, ...}
val:   {'n01440764': 5, 'n01443537': 3, ...}  # ❌ 索引不同
```

**修复**: 确保使用ImageFolder时目录结构完全一致

### 错误2: 验证集没有Normalize

```python
# 错误
train_transform = Compose([..., Normalize(...)])
val_transform = Compose([..., ToTensor()])  # ❌ 忘了Normalize
```

**修复**: 已确认两者都有Normalize ✓

### 错误3: 数据增强过强

```python
# 错误（旧代码）
RandomResizedCrop(224, scale=(0.8, 1.0))  # ❌ 太窄
RandomRotation(degrees=15)  # ❌ 对ImageNet不好

# 正确（新代码）✓
RandomResizedCrop(224, scale=(0.08, 1.0))  # ✓ 标准范围
# 移除RandomRotation
```

## 总结

**最可能的原因**: class_to_idx不一致（ImageNet最常见问题）

**快速验证**: 运行 `python quick_diagnosis.py`

**如果确认是数据增强问题**: 代码已修复，重新训练即可

**如果是类别映射问题**: 需要重新整理数据集
