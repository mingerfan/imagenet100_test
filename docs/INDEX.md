# 项目文档索引

欢迎来到 ImageNet-100 多模型训练系统的文档中心。

## 📚 主要文档

### 核心文档
- **[README.md](../README.md)** - 项目主文档
  - 系统架构介绍
  - 快速开始指南
  - 训练配置说明
  - 内存文件系统优化

- **[test/README.md](../test/README.md)** - 测试文档
  - 测试脚本说明
  - 使用方法
  - 测试模型列表

- **[fhe_statistics/README.md](../fhe_statistics/README.md)** - FHE 统计模块文档
  - 快速开始
  - 模块概览
  - 开发者指南
  - 重构说明

---

## 📖 特性文档

### 项目重构与改进
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - 项目重构总结
  - 完成的模块列表
  - 项目结构
  - 配置文件说明
  - 重构成果统计

- **[OPTIMIZER_IMPROVEMENTS.md](OPTIMIZER_IMPROVEMENTS.md)** - 优化器改进
  - 智能优化器选择
  - 梯度裁剪策略
  - 学习率调度器优化
  - 使用示例

- **[REGEX_MATCHING_GUIDE.md](REGEX_MATCHING_GUIDE.md)** - 正则匹配配置指南
  - 正则匹配功能说明
  - 配置示例
  - 高级用法
  - 最佳实践

---

## 🗂️ 文档组织结构

```
project/
├── README.md                      # 主项目文档
├── docs/                          # 文档目录
│   ├── INDEX.md                   # 本文档（索引）
│   ├── PROJECT_SUMMARY.md         # 项目总结
│   ├── OPTIMIZER_IMPROVEMENTS.md  # 优化器改进
│   └── REGEX_MATCHING_GUIDE.md    # 正则匹配指南
├── test/
│   └── README.md                  # 测试文档
└── fhe_statistics/
    └── README.md                  # FHE 统计模块文档
```

---

## 🎯 快速导航

### 我想...

#### 开始使用项目
→ 阅读 [主 README.md](../README.md)

#### 了解如何测试模型
→ 阅读 [test/README.md](../test/README.md)

#### 使用 FHE 统计功能
→ 阅读 [fhe_statistics/README.md](../fhe_statistics/README.md)

#### 了解项目架构和重构历史
→ 阅读 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

#### 配置训练优化器
→ 阅读 [OPTIMIZER_IMPROVEMENTS.md](OPTIMIZER_IMPROVEMENTS.md)

#### 使用正则表达式匹配模型
→ 阅读 [REGEX_MATCHING_GUIDE.md](REGEX_MATCHING_GUIDE.md)

---

## 📝 文档维护

### 文档更新日志

- **2026-01-10**: 整理项目文档结构
  - 创建 `docs/` 目录
  - 合并 `fhe_statistics/` 下的4个重构文档为1个
  - 移动特性文档到 `docs/`
  - 创建文档索引

### 贡献文档

如果你想为文档做贡献：

1. 确保文档使用 Markdown 格式
2. 遵循现有的文档结构
3. 在文档中包含代码示例
4. 更新本索引文件

---

## 🔗 相关链接

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch FX 文档](https://pytorch.org/docs/stable/fx.html)
- [Hydra 配置框架](https://hydra.cc/)
