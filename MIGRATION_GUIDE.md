# ImageNet100 迁移指南

## 📋 总体流程

```
源服务器: 提取清单 → 复制文本到剪贴板
   ↓
目标服务器: 粘贴清单 → 指定ImageNet1K路径 → 重建ImageNet100
```

---

## 🔧 第一步：在源服务器上提取文件清单

### 方法1：自动提取（推荐）

```bash
python extract_filelist.py /path/to/imagenet100
```

这将生成两个文件：
- `imagenet100_filelist.txt` - 完整列表（用于参考）
- `imagenet100_filelist_compact.json` - 紧凑格式（用于传输）

### 生成的文件示例

`imagenet100_filelist_compact.json` 格式：
```json
{
  "n01440764": ["n01440764_18.JPEG", "n01440764_11.JPEG", ...],
  "n01614925": ["n01614925_2674.JPEG", "n01614925_45.JPEG", ...],
  ...
}
```

---

## 📋 第二步：传输清单到目标服务器

### 方式1：通过剪贴板复制（最简洁）

```bash
# 源服务器 - 复制JSON文件内容
cat imagenet100_filelist_compact.json | clip  # Windows
cat imagenet100_filelist_compact.json | xclip -selection clipboard  # Linux

# 目标服务器 - 粘贴
# 使用编辑器粘贴内容到 imagenet100_filelist_compact.json
```

### 方式2：直接文件传输

```bash
# 如果可以使用USB/网盘等方式直接传输
scp imagenet100_filelist_compact.json user@target-server:/path/to/
```

---

## 🔨 第三步：在目标服务器上重建ImageNet100

### 前置条件

确认目标服务器有完整的ImageNet1K数据集，例如：
```
/mnt/data/imagenet1k/
├── n01440764/
├── n01614925/
├── ...
└── n15075141/
```

### 执行重建

```bash
# 方式1：使用符号链接（推荐，节省最多空间）
python rebuild_from_filelist.py \
  imagenet100_filelist_compact.json \
  /mnt/data/imagenet1k \
  /mnt/data/imagenet100

# 方式2：使用硬链接（节省空间但需同一文件系统）
python rebuild_from_filelist.py \
  imagenet100_filelist_compact.json \
  /mnt/data/imagenet1k \
  /mnt/data/imagenet100 \
  --hardlink

# 方式3：复制文件（最安全，占用完整空间）
python rebuild_from_filelist.py \
  imagenet100_filelist_compact.json \
  /mnt/data/imagenet1k \
  /mnt/data/imagenet100 \
  --copy
```

### 预期输出

```
📋 读取清单: 100 个类
📁 源ImageNet1K: /mnt/data/imagenet1k
📁 输出目录: /mnt/data/imagenet100
🔗 链接方式: 符号链接

==================================================
📊 重建完成统计
==================================================
✓ 成功处理的文件: 131200
⚠  缺失的文件: 0
⚠  缺失的类: 0
📁 总类数: 100
📄 总文件数: 131200

✓ 成功率: 100.0%
```

---

## ✅ 验证

### 快速验证

```bash
# 检查类数
ls /mnt/data/imagenet100 | wc -l  # 应该是 100

# 检查文件数
find /mnt/data/imagenet100 -type f | wc -l  # 应该匹配清单中的总数

# 随机检查一个文件
ls /mnt/data/imagenet100/n01440764 | head -5
```

### 用Python验证

```python
import json
from pathlib import Path

# 读取清单
with open("imagenet100_filelist_compact.json") as f:
    config = json.load(f)

# 验证
root = Path("/mnt/data/imagenet100")
missing = []

for class_name, files in config.items():
    for filename in files:
        filepath = root / class_name / filename
        if not filepath.exists():
            missing.append(f"{class_name}/{filename}")

if missing:
    print(f"❌ 缺失 {len(missing)} 个文件")
    for f in missing[:10]:
        print(f"  - {f}")
else:
    print("✅ 所有文件都已正确迁移！")
```

---

## 📊 空间对比

| 方式 | 目标服务器占用空间 | 速度 | 可用性 |
|------|------------------|------|-------|
| 符号链接 | ~0 字节 | ⚡⚡⚡ 快 | ✅ 完全可用 |
| 硬链接 | ~0 字节 | ⚡⚡ 中 | ✅ 完全可用 |
| 复制文件 | ~100GB | ⚡ 慢 | ✅ 完全可用 |

---

## 🐛 故障排除

### 问题1：缺少某些类

```
⚠  类不存在: n01234567
```

**原因**：目标服务器的ImageNet1K不完整

**解决**：检查源ImageNet1K中是否存在该类
```bash
ls /mnt/data/imagenet1k/n01234567
```

### 问题2：符号链接跨文件系统失败

```
❌ 处理失败: n01440764/file.JPEG
   错误: Invalid cross-device link
```

**原因**：源和目标在不同文件系统

**解决**：使用 `--hardlink` 或 `--copy` 选项

### 问题3：权限不足

```
❌ 处理失败: Permission denied
```

**解决**：确保有读取ImageNet1K和写入输出目录的权限
```bash
chmod 755 /mnt/data/imagenet100
```

---

## 💡 高级用法

### 创建测试清单（仅前3张图片）

```python
from rebuild_from_filelist import create_test_filelist
create_test_filelist("/path/to/imagenet100")
```

### 自定义选择某些类

编辑 `imagenet100_filelist_compact.json`，只保留需要的类：

```json
{
  "n01440764": ["file1.JPEG", "file2.JPEG", ...],
  "n01614925": ["file1.JPEG", "file2.JPEG", ...]
}
```

---

## 🔐 安全检查清单

- [ ] 验证源ImageNet100的完整性
- [ ] 确认目标服务器有足够的ImageNet1K数据
- [ ] 备份文件清单JSON
- [ ] 重建后进行验证
- [ ] 确认数据集可正常被训练脚本读取

---

## 📝 笔记

- 符号链接在大多数情况下是最好的选择
- 如需修改清单，直接编辑JSON文件即可
- 两个服务器的ImageNet1K路径可以不同，只需在重建时指定正确路径
