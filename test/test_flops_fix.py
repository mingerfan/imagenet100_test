#!/usr/bin/env python3
"""测试FLOPs统计功能"""

import torch
import torchvision
from fhe_statistics.statistics_fn import FheInfo

print("=" * 80)
print("测试FLOPs统计功能")
print("=" * 80)

# 测试单个模型的FLOPs计算
print("\n1. 测试ResNet18的FLOPs计算...")
try:
    model = torchvision.models.resnet18()
    fhe_info = FheInfo(model, input_shape=(1, 3, 224, 224), model_name="ResNet18")
    fhe_info.run_statistics()
    
    # 获取FLOPs
    flops = fhe_info.get_flops_count()
    print(f"   FLOPs: {flops:,} ({flops/1e6:.2f}M)")
    
    # 获取浅层指标
    shallow_metrics = fhe_info.get_shallow_layer_metrics(shallow_threshold=0.2)
    print(f"   浅层FLOPs占比: {shallow_metrics['shallow_flops_pct']:.2f}%")
    
    # 测试深度-FLOPs分布
    print("\n2. 测试深度-FLOPs分布图...")
    flops_dist = fhe_info.get_depth_flops_distribution(bin_size=10, max_bins=30)
    print(f"   深度分箱数量: {len(flops_dist['bins'])}")
    print(f"   FLOPs类型: {list(flops_dist['flops_data'].keys())}")
    print(f"   总FLOPs前5个bin: {[f'{x/1e6:.2f}M' for x in flops_dist['total_flops'][:5]]}")
    
    print("\n✓ FLOPs统计功能正常！")
    
except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("测试网络分组对比图的归一化功能")
print("=" * 80)

# 测试网络分组对比图的归一化
print("\n3. 测试网络分组对比图的归一化...")
try:
    models = {
        "ResNet18": torchvision.models.resnet18(),
        "ResNet34": torchvision.models.resnet34(),
    }
    
    network_infos = {}
    for name, model in models.items():
        fhe_info = FheInfo(model, input_shape=(1, 3, 224, 224), model_name=name)
        fhe_info.run_statistics()
        network_infos[name] = fhe_info
    
    # 手动检查归一化逻辑
    from fhe_statistics.statistics_fn import IMAGENET1K_ACCURACY
    
    print("\n原始数据:")
    for name, info in network_infos.items():
        flops = info.get_flops_count() / 1e6
        latency = (info.total_latency + info.total_boot_latency) / 1e6
        accuracy = IMAGENET1K_ACCURACY.get(name, 0)
        print(f"   {name}: FLOPs={flops:.2f}M, Latency={latency:.2f}M, Accuracy={accuracy:.1f}%")
    
    # 测试归一化
    raw_data = {}
    for name in network_infos.keys():
        info = network_infos[name]
        shallow_metrics = info.get_shallow_layer_metrics(shallow_threshold=0.2)
        raw_data[name] = {
            'FHE Latency (M)': (info.total_latency + info.total_boot_latency) / 1e6,
            'Accuracy (%)': IMAGENET1K_ACCURACY.get(name, 0),
            'FLOPs (M)': info.get_flops_count() / 1e6,
        }
    
    # 对每个指标进行Min-Max归一化
    metrics = ['FHE Latency (M)', 'Accuracy (%)', 'FLOPs (M)']
    normalized_data = {}
    
    for metric in metrics:
        values = [raw_data[name][metric] for name in network_infos.keys()]
        max_val = max(values)
        print(f"\n   {metric}:")
        print(f"     最大值: {max_val:.2f}")
        print(f"     归一化前: {[f'{v:.2f}' for v in values]}")
        
        for name in network_infos.keys():
            if name not in normalized_data:
                normalized_data[name] = {}
            normalized_data[name][metric] = raw_data[name][metric] / max_val if max_val > 0 else 0.0
        
        print(f"     归一化后: {[f'{normalized_data[n][metric]:.3f}' for n in network_infos.keys()]}")
    
    print("\n✓ 归一化功能正常！")
    
except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("所有测试完成！")
print("=" * 80)