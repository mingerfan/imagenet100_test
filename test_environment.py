"""
环境测试脚本
快速验证系统配置和依赖是否正常
"""

import sys
import os


def test_python_version():
    """测试Python版本"""
    print("\n" + "=" * 60)
    print("[1] 检查Python版本")
    print("=" * 60)
    
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 7:
        print("✓ Python版本符合要求 (>= 3.7)")
        return True
    else:
        print("✗ Python版本过低，需要 >= 3.7")
        return False


def test_imports():
    """测试必要的库是否安装"""
    print("\n" + "=" * 60)
    print("[2] 检查必要的库")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'torchvision',
        'tqdm': 'tqdm (进度条)',
        'numpy': 'numpy',
        'PIL': 'Pillow (图像处理)'
    }
    
    all_installed = True
    
    for module, name in required_packages.items():
        try:
            if module == 'PIL':
                from PIL import Image
                print(f"✓ {name}: 已安装")
            else:
                __import__(module)
                print(f"✓ {name}: 已安装")
        except ImportError:
            print(f"✗ {name}: 未安装")
            all_installed = False
    
    if all_installed:
        print("\n✓ 所有依赖库已安装")
    else:
        print("\n✗ 请安装缺失的库:")
        print("  pip install torch torchvision tqdm numpy pillow")
    
    return all_installed


def test_cuda():
    """测试CUDA是否可用"""
    print("\n" + "=" * 60)
    print("[3] 检查CUDA")
    print("=" * 60)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✓ CUDA可用")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        else:
            print("✗ CUDA不可用，将使用CPU训练")
        
        return cuda_available
    except:
        print("✗ 无法检测CUDA")
        return False


def test_dataset_paths():
    """测试数据集路径是否存在"""
    print("\n" + "=" * 60)
    print("[4] 检查数据集路径")
    print("=" * 60)
    
    paths = {
        '训练集': '/home/xuming/Documents/dataset/ImageNet_100/train',
        '验证集': '/home/xuming/Documents/dataset/ImageNet_100/val',
        '标签文件': '/home/xuming/Documents/dataset/label/LOC_synset_mapping.txt'
    }
    
    all_exist = True
    
    for name, path in paths.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                # 统计文件夹数量
                num_classes = len([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
                print(f"✓ {name}: {path}")
                print(f"  包含 {num_classes} 个类别")
            else:
                # 文件
                size = os.path.getsize(path)
                print(f"✓ {name}: {path}")
                print(f"  文件大小: {size / 1024:.2f} KB")
        else:
            print(f"✗ {name}: {path}")
            all_exist = False
    
    if all_exist:
        print("\n✓ 所有数据集路径有效")
    else:
        print("\n✗ 部分数据集路径不存在，请检查")
    
    return all_exist


def test_memory():
    """测试内存信息"""
    print("\n" + "=" * 60)
    print("[5] 系统内存信息")
    print("=" * 60)
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        total = mem.total / 1024**3
        available = mem.available / 1024**3
        used = mem.used / 1024**3
        
        print(f"总内存: {total:.2f} GB")
        print(f"已用内存: {used:.2f} GB ({mem.percent}%)")
        print(f"可用内存: {available:.2f} GB")
        
        if available >= 100:
            print(f"✓ 可用内存充足 ({available:.2f} GB >= 100 GB)")
            return True
        else:
            print(f"⚠ 可用内存不足 ({available:.2f} GB < 100 GB)")
            print("  建议关闭内存缓存或释放内存")
            return False
    except ImportError:
        print("⚠ 无法检测内存信息 (需要安装 psutil)")
        print("  安装命令: pip install psutil")
        return None


def test_gpu_memory():
    """测试GPU内存"""
    print("\n" + "=" * 60)
    print("[6] GPU显存信息")
    print("=" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  总显存: {total:.2f} GB")
                print(f"  已分配: {allocated:.2f} GB")
                print(f"  已缓存: {cached:.2f} GB")
                print(f"  可用: {total - cached:.2f} GB")
                
                if total - cached >= 4:
                    print(f"  ✓ 显存充足")
                else:
                    print(f"  ⚠ 显存不足，建议减小batch_size")
            
            return True
        else:
            print("⚠ CUDA不可用")
            return False
    except:
        print("✗ 无法检测GPU显存")
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 60)
    print("[7] 测试数据加载")
    print("=" * 60)
    
    try:
        from torchvision import datasets, transforms
        import time
        
        train_dir = '/home/xuming/Documents/dataset/ImageNet_100/train'
        
        # 创建简单的transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        print("正在加载训练集...")
        start = time.time()
        dataset = datasets.ImageFolder(train_dir, transform=transform)
        elapsed = time.time() - start
        
        print(f"✓ 加载完成")
        print(f"  图片数量: {len(dataset):,}")
        print(f"  类别数量: {len(dataset.classes)}")
        print(f"  加载时间: {elapsed:.2f} 秒")
        
        # 测试加载一张图片
        print("\n测试加载单张图片...")
        start = time.time()
        img, label = dataset[0]
        elapsed = time.time() - start
        print(f"✓ 图片加载成功")
        print(f"  图片形状: {img.shape}")
        print(f"  标签: {label}")
        print(f"  加载时间: {elapsed:.4f} 秒")
        
        return True
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("ImageNet-100 环境测试")
    print("=" * 60)
    
    results = []
    
    # 运行所有测试
    results.append(('Python版本', test_python_version()))
    results.append(('依赖库', test_imports()))
    results.append(('CUDA', test_cuda()))
    results.append(('数据集路径', test_dataset_paths()))
    results.append(('系统内存', test_memory()))
    results.append(('GPU显存', test_gpu_memory()))
    results.append(('数据加载', test_data_loading()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    passed = 0
    failed = 0
    warnings = 0
    
    for test_name, result in results:
        if result is True:
            print(f"✓ {test_name}: 通过")
            passed += 1
        elif result is False:
            print(f"✗ {test_name}: 失败")
            failed += 1
        else:
            print(f"⚠ {test_name}: 跳过")
            warnings += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败, {warnings} 跳过")
    
    if failed == 0:
        print("\n✓ 环境配置完成！可以开始训练了。")
        print("\n下一步:")
        print("  1. 测试数据加载器: python dataset_loader.py")
        print("  2. 开始训练: python train.py")
    else:
        print("\n✗ 环境配置存在问题，请先解决上述错误。")
        sys.exit(1)


if __name__ == '__main__':
    main()