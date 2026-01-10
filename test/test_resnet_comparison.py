"""测试ResNet不同变体的比较"""
import sys
import torch
import torchvision
from fhe_statistics.statistics_fn import FheInfo, compare_networks

def main():
    """比较ResNet18、ResNet34、ResNet50"""
    print("\n" + "="*60)
    print("ResNet变体横向比较")
    print("="*60)
    
    models = [
        ("ResNet18", torchvision.models.resnet18()),
        ("ResNet34", torchvision.models.resnet34()),
        ("ResNet50", torchvision.models.resnet50()),
    ]
    
    # 使用compare_networks函数
    network_data = compare_networks(models, plot_folder=".", input_shape=(1, 3, 224, 224))
    
    print(f"\n比较完成，共 {len(network_data)} 个网络")
    for net_name, net_stats in network_data.items():
        total = sum(net_stats.values())
        print(f"  {net_name}: 总耗时 = {total:.0f}")
    
    print("\n" + "="*60)
    print("测试完成！✓")
    print("="*60)

if __name__ == "__main__":
    main()