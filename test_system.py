"""
测试脚本 - 验证训练系统是否正常工作
"""

import torch
from models import get_model, MODEL_REGISTRY
from trainers import MultiGPUManager
from utils import load_config, get_model_configs
import sys


def test_model_registry():
    """测试模型注册器"""
    print("=" * 60)
    print("测试1: 模型注册器")
    print("=" * 60)
    
    # 列出所有已注册的模型
    models = MODEL_REGISTRY.list_models()
    print(f"已注册的模型: {models}")
    
    if len(models) == 0:
        print("❌ 没有找到已注册的模型")
        return False
    
    # 测试创建模型
    for model_name in models:
        try:
            model = get_model(model_name, num_classes=100, pretrained=False)
            print(f"✓ 成功创建模型: {model_name}")
            
            # 统计参数
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  参数量: {total_params:,}")
        except Exception as e:
            print(f"❌ 创建模型失败 {model_name}: {e}")
            return False
    
    print("✓ 模型注册器测试通过\n")
    return True


def test_config_loading():
    """测试配置加载"""
    print("=" * 60)
    print("测试2: 配置加载")
    print("=" * 60)
    
    config_path = 'configs/models_list.yaml'
    
    try:
        config = load_config(config_path)
        print(f"✓ 成功加载配置文件: {config_path}")
        
        # 获取模型配置
        model_configs = get_model_configs(config)
        print(f"✓ 找到 {len(model_configs)} 个模型配置")
        
        for i, model_config in enumerate(model_configs, 1):
            print(f"\n  {i}. {model_config['name']}")
            print(f"     Epochs: {model_config.get('epochs', 60)}")
            print(f"     Batch Size: {model_config.get('batch_size', 128)}")
            print(f"     Learning Rate: {model_config.get('learning_rate', 0.001)}")
        
        print("\n✓ 配置加载测试通过\n")
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False


def test_gpu_availability():
    """测试GPU可用性"""
    print("=" * 60)
    print("测试3: GPU可用性")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠ CUDA不可用，将使用CPU")
        return True
    
    num_gpus = torch.cuda.device_count()
    print(f"✓ 检测到 {num_gpus} 个GPU")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    内存: {props.total_memory / 1024**3:.1f} GB")
        print(f"    计算能力: {props.major}.{props.minor}")
    
    print("✓ GPU可用性测试通过\n")
    return True


def test_multi_gpu_manager():
    """测试多GPU管理器初始化"""
    print("=" * 60)
    print("测试4: 多GPU管理器")
    print("=" * 60)
    
    try:
        manager = MultiGPUManager(
            train_dir='/home/xuming/Documents/dataset/ImageNet_100/train',
            val_dir='/home/xuming/Documents/dataset/ImageNet_100/val',
            result_dir='./results',
            gpus=[0, 1, 2, 3],
            num_classes=100,
            use_memory_fs=False  # 测试时不使用内存FS
        )
        
        print(f"✓ 多GPU管理器创建成功")
        print(f"  可用GPU: {manager.available_gpus}")
        
        # 测试配置加载
        config = load_config('configs/models_list.yaml')
        model_configs = get_model_configs(config)
        
        print(f"\n✓ 检查模型训练状态:")
        for model_config in model_configs[:2]:  # 只检查前两个
            is_trained = manager.is_model_trained(model_config['name'])
            status = "已训练" if is_trained else "未训练"
            print(f"  {model_config['name']}: {status}")
        
        print("\n✓ 多GPU管理器测试通过\n")
        return True
        
    except Exception as e:
        print(f"❌ 多GPU管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("ImageNet-100 训练系统测试")
    print("=" * 60 + "\n")
    
    tests = [
        ("模型注册器", test_model_registry),
        ("配置加载", test_config_loading),
        ("GPU可用性", test_gpu_availability),
        ("多GPU管理器", test_multi_gpu_manager),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 测试 '{test_name}' 发生异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 打印总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统准备就绪。")
        print("\n您现在可以运行:")
        print("  python train.py                    # 训练所有模型")
        print("  python train.py --models resnet18  # 只训练resnet18")
        print("  python train.py --gpus 0 1         # 使用GPU 0和1")
        return 0
    else:
        print(f"\n⚠ 有 {total - passed} 个测试失败，请检查错误信息")
        return 1


if __name__ == '__main__':
    sys.exit(main())