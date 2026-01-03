# 正则匹配配置功能使用指南

## 功能说明

从现在开始，你的训练系统支持两种方式配置模型：

1. **显式指定**：在 `models` 列表中逐个添加模型配置
2. **正则匹配**：使用正则表达式自动匹配已注册的模型

## 工作原理

系统会：
1. 从 `models.MODEL_REGISTRY` 获取所有已注册的模型名称
2. 读取 `configs/models_list.yaml` 配置文件
3. 对于 `model_patterns` 中的每个正则模式，匹配已注册的模型
4. 为匹配到的模型自动生成配置（使用模式指定的参数）
5. 显式指定的模型优先级最高，会覆盖正则匹配

## 使用方法

### 1. 基本用法

在 `configs/models_list.yaml` 中添加 `model_patterns` 部分：

```yaml
model_patterns:
  - pattern: "^resnet-.*"
    description: "所有 ResNet 变体"
    epochs: 60
    batch_size: 64
    learning_rate: 0.001
    num_workers: 16
    params:
      num_classes: 100
      pretrained: false
```

### 2. 常见正则模式示例

#### 匹配所有以 "resnet-" 开头的模型
```yaml
- pattern: "^resnet-.*"
```

#### 只匹配 basic_self_gated 类型的模型
```yaml
- pattern: "^resnet-basic_self_gated-.*"
```

#### 匹配所有使用 learnablerelu 的模型
```yaml
- pattern: ".*learnablerelu.*"
```

#### 匹配所有使用 gated 的模型（包括 basic_self_gated 和 bottleneck_self_gated）
```yaml
- pattern: ".*self_gated.*"
```

#### 匹配特定激活函数的模型
```yaml
- pattern: ".*stablepoly4.*"
```

### 3. 多个模式组合

你可以定义多个模式，每个模式使用不同的配置：

```yaml
model_patterns:
  # Self-gated 模型使用较小的 batch size
  - pattern: ".*self_gated.*"
    epochs: 50
    batch_size: 32
    learning_rate: 0.01
  
  # 普通 ResNet 变体
  - pattern: "^resnet-.*"
    epochs: 60
    batch_size: 64
    learning_rate: 0.001
```

### 4. 优先级规则

1. **显式指定的模型**（在 `models` 列表中）优先级最高
2. 正则匹配按配置文件中的顺序依次进行
3. 如果一个模型被多个模式匹配，使用第一个匹配到的模式配置

示例：

```yaml
models:
  # resnet18 使用显式配置（batch_size=128）
  - name: "resnet18"
    batch_size: 128
    # ...其他配置

model_patterns:
  # 这个模式不会匹配 resnet18（因为它已在 models 中）
  - pattern: "^resnet.*"
    batch_size: 64  # 这个配置不会应用到 resnet18
```

### 5. 配置继承

正则匹配的配置会继承全局设置：

```yaml
global:
  num_classes: 100
  default_epochs: 60
  default_batch_size: 128
  default_learning_rate: 0.001

model_patterns:
  - pattern: "^resnet-.*"
    # 只需要覆盖需要修改的参数
    batch_size: 64  # epochs 和 learning_rate 会使用全局默认值
```

## 你的模型

当前 `models/gate_net.py` 中注册了 16 个模型变体：

- `resnet-basic-*-layer1block1` (4个)
- `resnet-basic_self_gated-*-layer1block1` (4个)
- `resnet-bottleneck-*-layer1block1` (4个)
- `resnet-bottleneck_self_gated-*-layer1block1` (4个)

其中 `*` 可以是：`relu`, `learnableswish`, `learnablerelu`, `stablepoly4`

默认配置已经设置为匹配所有这些模型（`^resnet-.*`），你无需逐个添加！

## 运行训练

### 训练所有配置的模型（包括正则匹配的）
```bash
uv run python train.py
```

### 只训练特定的模型（即使有正则匹配）
```bash
uv run python train.py --models resnet18 resnet-basic-relu-layer1block1
```

### 查看将要训练的模型列表（不实际训练）
运行时会显示所有将要训练的模型及其配置

## 注意事项

1. 确保模型已经在 `models/__init__.py` 中导入并注册
2. 正则表达式必须是有效的 Python 正则表达式
3. 如果正则表达式有误，系统会显示警告并跳过该模式
4. 建议先用测试脚本验证配置：
   ```bash
   uv run python test/test_regex_matching.py
   ```

## 常见问题

**Q: 我新注册了一个模型，但训练时没看到它？**
A: 检查：
1. 模型是否在 `models/__init__.py` 中导入
2. 模型名称是否匹配 `model_patterns` 中的正则表达式
3. 运行 `uv run python test/test_regex_matching.py` 查看匹配结果

**Q: 我想让某个模型使用不同的配置，但正则匹配覆盖了我的设置？**
A: 在 `models` 列表中显式指定该模型，显式配置优先级最高。

**Q: 如何临时禁用正则匹配？**
A: 注释掉 `model_patterns` 部分或删除相关配置。