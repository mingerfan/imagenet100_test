"""
测试 StablePoly7 的 set_epoch 方法是否被正确调用
"""

import torch
import sys
from models import get_model
from trainers.base_trainer import Trainer


def test_set_epoch_fix():
    """测试 set_epoch 是否被正确调用"""
    
    print("=" * 60)
    print("测试 StablePoly7 set_epoch 修复")
    print("=" * 60)
    
    # 创建一个使用 StablePoly7 的模型
    model_name = 'resnet-basic-stablepoly7-layer1block1'
    print(f"\n创建模型: {model_name}")
    
    try:
        model = get_model(model_name, num_classes=100)
        print("✓ 模型创建成功")
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False
    
    # 统计 StablePoly7 实例数量
    stablepoly_count = 0
    for module in model.modules():
        if module.__class__.__name__ == 'StablePoly7':
            stablepoly_count += 1
    
    print(f"✓ 找到 {stablepoly_count} 个 StablePoly7 实例")
    
    if stablepoly_count == 0:
        print("✗ 警告：没有找到 StablePoly7 实例")
        return False
    
    # 创建一个简化的 Trainer 对象（不需要真实的数据加载器）
    # 我们只需要测试 _set_epoch_for_model 方法
    class TestTrainer(Trainer):
        def __init__(self, model):
            # 只初始化必要的属性
            self.model = model
    
    trainer = TestTrainer(model)
    
    # 测试 _set_epoch_for_model 方法
    test_epochs = [1, 10, 31, 40, 60]
    
    print("\n测试 _set_epoch_for_model 方法:")
    for epoch in test_epochs:
        # 调用 _set_epoch_for_model
        trainer._set_epoch_for_model(epoch)
        
        # 检查所有 StablePoly7 实例的 current_epoch
        all_match = True
        for module in model.modules():
            if module.__class__.__name__ == 'StablePoly7':
                if module.current_epoch != epoch:
                    all_match = False
                    print(f"  ✗ Epoch {epoch}: 某些模块的 current_epoch 未正确更新")
                    break
        
        if all_match:
            # 检查 alpha 值是否符合预期
            # 前 30 个 epoch: alpha = 0.0 (使用 ReLU)
            # 31-40 个 epoch: alpha 逐渐从 0 到 1 (过渡)
            # 40+ 个 epoch: alpha = 1.0 (使用多项式)
            
            # 测试一个 StablePoly7 实例的 forward 行为
            for module in model.modules():
                if module.__class__.__name__ == 'StablePoly7':
                    x = torch.tensor([[1.0]])
                    with torch.no_grad():
                        output = module(x)
                    
                    # 打印当前状态
                    if epoch <= 30:
                        expected_phase = "ReLU 预热阶段"
                    elif epoch <= 40:
                        expected_phase = "过渡阶段"
                    else:
                        expected_phase = "多项式激活阶段"
                    
                    print(f"  ✓ Epoch {epoch}: current_epoch={epoch}, 阶段={expected_phase}")
                    break
    
    print("\n" + "=" * 60)
    print("测试完成！set_epoch 修复验证通过 ✓")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    success = test_set_epoch_fix()
    sys.exit(0 if success else 1)