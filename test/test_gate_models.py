import torch
from models.registry import MODEL_REGISTRY


def test_all_models():
    """测试所有注册的 gate_net 模型"""
    
    print("=" * 70)
    print("Testing All ResNet-18 Gate Variants")
    print("=" * 70)
    
    # 获取所有已注册的模型
    all_models = MODEL_REGISTRY.list_models()
    
    # 筛选出 gate_net 相关的模型
    gate_models = [name for name in all_models if 'layer1block1' in name]
    
    print(f"\nFound {len(gate_models)} gate_net models to test\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    results = {}
    
    for model_name in sorted(gate_models):
        try:
            print(f"Testing: {model_name}")
            print("-" * 70)
            
            # 创建模型
            model = MODEL_REGISTRY.get(model_name, num_classes=100)
            model.to(device)
            model.eval()
            
            # 统计参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 测试前向传播
            test_input = torch.randn(2, 3, 224, 224).to(device)
            with torch.no_grad():
                output = model(test_input)
            
            # 验证输出形状
            expected_shape = (2, 100)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            
            print(f"  ✓ Model created successfully")
            print(f"  ✓ Forward pass successful: {test_input.shape} -> {output.shape}")
            print(f"  ✓ Total parameters: {total_params:,}")
            print(f"  ✓ Trainable parameters: {trainable_params:,}")
            
            results[model_name] = {
                'status': 'success',
                'output_shape': output.shape,
                'total_params': total_params,
                'trainable_params': trainable_params
            }
            
            print()
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            print()
            results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # 打印总结
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    failed_count = sum(1 for r in results.values() if r['status'] == 'failed')
    
    print(f"\nTotal models tested: {len(gate_models)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    
    if failed_count > 0:
        print("\nFailed models:")
        for name, result in results.items():
            if result['status'] == 'failed':
                print(f"  - {name}: {result['error']}")
    
    # 打印参数量统计
    if success_count > 0:
        print("\nParameter Statistics:")
        params = [r['total_params'] for r in results.values() if r['status'] == 'success']
        print(f"  Min params: {min(params):,}")
        print(f"  Max params: {max(params):,}")
        print(f"  Avg params: {sum(params)//len(params):,}")
    
    print("\n" + "=" * 70)
    
    if failed_count == 0:
        print("All tests passed! ✓")
    else:
        print(f"{failed_count} test(s) failed! ✗")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    test_all_models()