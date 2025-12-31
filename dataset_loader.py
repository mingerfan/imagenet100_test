"""
ImageNet-100 数据加载器
使用ResNet18和完全内存缓存优化数据加载性能
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import time
import os


class CachedImageFolder(datasets.ImageFolder):
    """
    带内存缓存的ImageFolder
    将所有图片预加载到内存中，大幅提升数据读取速度
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        print(f"\n正在将数据集缓存到内存: {root}")
        print(f"总图片数: {len(self.samples)}")
        
        self.cache = {}
        self.targets = []
        
        # 统计信息
        total_images = len(self.samples)
        cache_start = time.time()
        
        # 预加载所有图片到内存
        for idx, (path, target) in enumerate(self.samples):
            # 加载图片
            img = Image.open(path).convert('RGB')
            self.cache[idx] = img
            self.targets.append(target)
            
            # 显示进度
            if (idx + 1) % 1000 == 0 or (idx + 1) == total_images:
                progress = (idx + 1) / total_images * 100
                elapsed = time.time() - cache_start
                print(f"  进度: {idx + 1}/{total_images} ({progress:.1f}%) - 已用时间: {elapsed:.1f}s")
        
        self.targets = torch.tensor(self.targets)
        
        cache_time = time.time() - cache_start
        print(f"✓ 数据集缓存完成! 耗时: {cache_time:.2f} 秒")
        print(f"✓ 估计内存占用: ~{total_images * 0.5:.1f} MB (假设每张图片0.5MB)")
    
    def __getitem__(self, index):
        """
        获取缓存的图片并应用transform
        """
        # 从缓存获取原始图片
        img = self.cache[index]
        
        # 应用transform
        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.targets[index]
    
    def __len__(self):
        return len(self.samples)


def load_class_mapping(label_file):
    """
    读取synset到类别名称的映射
    
    Args:
        label_file: LOC_synset_mapping.txt文件路径
    
    Returns:
        idx_to_class: 类别索引到类别名称的字典
    """
    synset_map = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            synset_id = parts[0]
            class_name = parts[1] if len(parts) > 1 else synset_id
            synset_map[synset_id] = class_name
    return synset_map


def create_dataloaders(
    train_dir,
    val_dir,
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    use_cache=True
):
    """
    创建训练和验证数据加载器
    
    Args:
        train_dir: 训练集目录
        val_dir: 验证集目录
        batch_size: 批次大小
        num_workers: 数据加载的worker数量
        pin_memory: 是否使用内存固定（GPU训练时设为True）
        use_cache: 是否使用内存缓存
    
    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    print("=" * 60)
    print("ImageNet-100 数据加载器配置")
    print("=" * 60)
    
    # 数据预处理
    print("\n[1] 配置数据预处理...")
    
    # 训练集预处理（包含数据增强）
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 验证集预处理（不包含数据增强）
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    print("  ✓ 训练集: RandomResizedCrop + RandomHorizontalFlip + RandomRotation + ColorJitter")
    print("  ✓ 验证集: Resize + CenterCrop")
    
    # 创建数据集
    print("\n[2] 加载数据集...")
    
    if use_cache:
        print("  使用完全内存缓存模式...")
        train_dataset = CachedImageFolder(train_dir, transform=train_transform)
        val_dataset = CachedImageFolder(val_dir, transform=val_transform)
    else:
        print("  使用标准磁盘读取模式...")
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    # 打印数据集信息
    print(f"\n  训练集大小: {len(train_dataset):,} 张图片")
    print(f"  验证集大小: {len(val_dataset):,} 张图片")
    print(f"  类别数量: {len(train_dataset.classes)} 个类别")
    
    # 创建DataLoader
    print("\n[3] 创建DataLoader...")
    print(f"  批次大小: {batch_size}")
    print(f"  Worker数量: {num_workers}")
    print(f"  内存固定: {pin_memory}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )
    
    print(f"\n  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    
    return train_loader, val_loader, train_dataset, val_dataset


def create_resnet18(num_classes=100, pretrained=True):
    """
    创建ResNet18模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    
    Returns:
        model: ResNet18模型
    """
    print("\n[4] 创建ResNet18模型...")
    
    model = models.resnet18(pretrained=pretrained)
    
    # 修改最后一层全连接层
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    print(f"  ✓ 使用预训练权重: {pretrained}")
    print(f"  ✓ 输出类别数: {num_classes}")
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ 总参数量: {total_params:,}")
    print(f"  ✓ 可训练参数量: {trainable_params:,}")
    
    return model


def test_dataloader_speed(train_loader, num_batches=10, device='cuda'):
    """
    测试数据加载器速度
    
    Args:
        train_loader: 训练数据加载器
        num_batches: 测试的批次数
        device: 设备类型
    """
    print("\n[5] 测试数据加载速度...")
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("  ⚠ CUDA不可用，使用CPU")
        device = 'cpu'
    
    device = torch.device(device)
    
    # 预热
    print("  预热中...")
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if batch_idx >= 5:
            break
    
    # 正式测试
    print(f"  测试 {num_batches} 个批次...")
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if batch_idx >= num_batches - 1:
            break
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    batch_size = images.size(0)
    total_images = num_batches * batch_size
    throughput = total_images / elapsed
    avg_time_per_batch = elapsed / num_batches
    
    print(f"\n  数据加载性能结果:")
    print(f"  - 加载图片数: {total_images:,}")
    print(f"  - 总耗时: {elapsed:.2f} 秒")
    print(f"  - 吞吐量: {throughput:.2f} 图片/秒")
    print(f"  - 每批次平均耗时: {avg_time_per_batch:.4f} 秒")


def main():
    """主函数"""
    # 配置参数
    DATA_ROOT = '/home/xuming/Documents/dataset/ImageNet_100'
    TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
    VAL_DIR = os.path.join(DATA_ROOT, 'val')
    LABEL_FILE = '/home/xuming/Documents/dataset/label/LOC_synset_mapping.txt'
    
    # 训练参数
    BATCH_SIZE = 128  # 根据GPU内存调整
    NUM_WORKERS = 16   # 根据CPU核心数调整
    PIN_MEMORY = True # GPU训练时设为True
    USE_CACHE = True  # 使用内存缓存
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        use_cache=USE_CACHE
    )
    
    # 创建模型
    model = create_resnet18(num_classes=100, pretrained=True)
    model = model.to(device)
    
    # 测试数据加载速度
    test_dataloader_speed(train_loader, num_batches=50, device=device)
    
    # 打印类别映射示例
    synset_map = load_class_mapping(LABEL_FILE)
    idx_to_class = {idx: synset_map.get(cls_name, cls_name) 
                    for idx, cls_name in enumerate(train_dataset.classes)}
    
    print("\n" + "=" * 60)
    print("类别映射示例 (前10个):")
    for i in range(min(10, len(idx_to_class))):
        print(f"  {i}: {train_dataset.classes[i]} -> {idx_to_class[i]}")
    print("=" * 60)
    
    print("\n✓ 数据加载器配置完成!")
    print("\n使用示例:")
    print("  for images, labels in train_loader:")
    print("      images, labels = images.to(device), labels.to(device)")
    print("      # 训练代码...")


if __name__ == '__main__':
    main()