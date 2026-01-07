"""测试新增的FHE统计功能"""
import sys
import torch
import torchvision
from fhe_statistics.statistics_fn import FheInfo, analyze_model, compare_networks

def test_single_model():
    """测试单个模型的完整统计功能"""
    print("\n" + "="*60)
    print("测试1: 单个模型完整统计")
    print("="*60)
    
    model = torchvision.models.resnet18()
    fhe_info = FheInfo(model)
    fhe_info.run_statistics()
    
    # 测试数据准备函数
    print("\n--- 测试 get_operator_breakdown_data() ---")
    operator_data = fhe_info.get_operator_breakdown_data()
    print(f"算子类型数量: {len(operator_data)}")
    for op_type, data in list(operator_data.items())[:3]:
        print(f"  {op_type}: rotation={data['rotation']:.0f}, mul_single={data['mul_single']:.0f}, boot={data['total_boot_latency']:.0f}")
    
    print("\n--- 测试 get_depth_histogram_data() ---")
    depth_data = fhe_info.get_depth_histogram_data(bin_size=10)
    print(f"深度区间数量: {len(depth_data['bins'])}")
    print(f"深度区间: {depth_data['bins'][:5]}")
    
    print("\n--- 测试 get_network_comparison_data() ---")
    network_data = fhe_info.get_network_comparison_data()
    print(f"算子类型数量: {len(network_data)}")
    for op_type, latency in list(network_data.items())[:3]:
        print(f"  {op_type}: {latency:.0f}")
    
    # 测试可视化函数（不显示，只保存）
    print("\n--- 测试可视化函数 ---")
    fhe_info.plot_operator_stack(plot_folder=".", show=False)
    print("✓ 算子堆栈图生成成功")
    
    fhe_info.plot_depth_histogram(bin_size=10, plot_folder=".", show=False)
    print("✓ 深度直方图生成成功")
    
    # 测试带max_bins的深度直方图
    fhe_info.plot_depth_histogram(bin_size=10, max_bins=30, plot_folder=".", show=False)
    print("✓ 深度直方图（限制30个区间）生成成功")
    
    fhe_info._plot_basic_statistics(plot_folder=".", show=False)
    print("✓ 基础统计图生成成功")
    
    # 测试plot_statistics
    fhe_info.plot_statistics(plot_folder=".", show=False, plot_types=['all'])
    print("✓ 所有图表生成成功")
    
    return True

def test_network_comparison():
    """测试多个网络的横向比较"""
    print("\n" + "="*60)
    print("测试2: 多个网络横向比较")
    print("="*60)
    
    models = [
        ("ResNet18", torchvision.models.resnet18()),
        ("MobileNetV2", torchvision.models.mobilenet_v2()),
    ]
    
    # 使用compare_networks函数
    network_data = compare_networks(models, plot_folder=".", input_shape=(1, 3, 224, 224))
    
    print(f"\n比较完成，共 {len(network_data)} 个网络")
    for net_name, net_stats in network_data.items():
        total = sum(net_stats.values())
        print(f"  {net_name}: 总耗时 = {total:.0f}")
    
    return True

def test_plot_types():
    """测试不同的plot_types参数"""
    print("\n" + "="*60)
    print("测试3: 不同plot_types参数")
    print("="*60)
    
    model = torchvision.models.resnet18()
    fhe_info = FheInfo(model)
    fhe_info.run_statistics()
    
    # 测试basic
    print("\n--- 测试 plot_types=['basic'] ---")
    fhe_info.plot_statistics(plot_folder=".", show=False, plot_types=['basic'])
    print("✓ basic类型图表生成成功")
    
    # 测试operator_stack
    print("\n--- 测试 plot_types=['operator_stack'] ---")
    fhe_info.plot_statistics(plot_folder=".", show=False, plot_types=['operator_stack'])
    print("✓ operator_stack类型图表生成成功")
    
    # 测试depth_histogram
    print("\n--- 测试 plot_types=['depth_histogram'] ---")
    fhe_info.plot_statistics(plot_folder=".", show=False, plot_types=['depth_histogram'])
    print("✓ depth_histogram类型图表生成成功")
    
    return True

def main():
    """运行所有测试"""
    print("\n开始测试新增的FHE统计功能...")
    
    try:
        # 测试1: 单个模型
        test_single_model()
        
        # 测试2: 网络比较
        test_network_comparison()
        
        # 测试3: 不同plot_types
        test_plot_types()
        
        print("\n" + "="*60)
        print("所有测试通过！✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)