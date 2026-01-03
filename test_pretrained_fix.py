"""
测试脚本：验证所有模型都能接受 pretrained 参数
"""

import torch
from models import get_model, MODEL_REGISTRY

def test_all_models():
    """测试所有已注册的模型"""
    print("=" * 80)
    print("测试所有模型是否能接受 pretrained 参数")
    print("=" * 80)
    
    registered_models = list(MODEL_REGISTRY._registry.keys())
    print(f"\n已注册模型总数: {len(registered_models)}")
    
    success_count = 0
    failed_models = []
    
    for model_name in registered_models:
        print(f"\n测试模型: {model_name}")
        try:
            # 测试不带 pretrained 参数
            model1 = get_model(model_name, num_classes=100)
            print(f"  ✓ 不带 pretrained 参数: 成功")
            
            # 测试带 pretrained=False
            model2 = get_model(model_name, num_classes=100, pretrained=False)
            print(f"  ✓ 带 pretrained=False: 成功")
            
            # 测试带 pretrained=True（对于标准模型应该工作，自定义模型会忽略）
            model3 = get_model(model_name, num_classes=100, pretrained=True)
            print(f"  ✓ 带 pretrained=True: 成功")
            
            # 验证模型结构
            if model1 is None or model2 is None or model3 is None:
                raise Exception("模型创建返回None")
            
            # 简单的前向传播测试
            dummy_input = torch.randn(1, 3, 224, 224)
            output = model1(dummy_input)
            if output.shape != (1, 100):
                raise Exception(f"输出形状错误: {output.shape}, 期望: (1, 100)")
            
            print(f"  ✓ 前向传播测试通过")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            failed_models.append((model_name, str(e)))
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"\n✅ 成功: {success_count}/{len(registered_models)}")
    
    if failed_models:
        print(f"\n❌ 失败的模型:")
        for model_name, error in failed_models:
            print(f"  - {model_name}: {error}")
        return False
    else:
        print(f"\n🎉 所有模型测试通过！")
        return True

if __name__ == "__main__":
    success = test_all_models()
    exit(0 if success else 1)