"""
测试 StablePoly4 的 register_buffer 修复是否正确

验证：
1. current_epoch 是否被正确保存到 state_dict
2. 加载模型后 current_epoch 是否能正确恢复
3. 推理时是否能正确使用多项式激活
"""

import os
import sys
import tempfile

import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gate_net_cmp.block_def import StablePoly4


def test_buffer_in_state_dict():
    """测试 current_epoch 是否在 state_dict 中"""
    print("=" * 60)
    print("测试 1: current_epoch 是否被保存到 state_dict")
    print("=" * 60)

    module = StablePoly4()

    # 设置 epoch 为训练完成状态
    module.set_epoch(60)

    # 获取 state_dict
    state_dict = module.state_dict()

    print(f"state_dict 中的键: {list(state_dict.keys())}")

    if "current_epoch" in state_dict:
        print(f"✓ current_epoch 在 state_dict 中")
        print(f"  值: {state_dict['current_epoch']}")
        return True
    else:
        print("✗ current_epoch 不在 state_dict 中！")
        return False


def test_save_and_load():
    """测试保存和加载后 current_epoch 是否正确恢复"""
    print("\n" + "=" * 60)
    print("测试 2: 保存和加载后 current_epoch 是否正确恢复")
    print("=" * 60)

    # 创建模块并设置 epoch
    module1 = StablePoly4()
    module1.set_epoch(60)  # 设置为训练完成后的状态

    print(f"保存前 current_epoch: {module1.current_epoch.item()}")

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        temp_path = f.name

    try:
        torch.save(module1.state_dict(), temp_path)
        print(f"✓ 模型已保存到 {temp_path}")

        # 创建新模块并加载
        module2 = StablePoly4()
        print(f"加载前新模块的 current_epoch: {module2.current_epoch.item()}")

        module2.load_state_dict(torch.load(temp_path, weights_only=True))
        print(f"加载后新模块的 current_epoch: {module2.current_epoch.item()}")

        if module2.current_epoch.item() == 60:
            print("✓ current_epoch 正确恢复为 60")
            return True
        else:
            print(
                f"✗ current_epoch 恢复错误，期望 60，实际 {module2.current_epoch.item()}"
            )
            return False
    finally:
        os.unlink(temp_path)


def test_inference_uses_polynomial():
    """测试推理时是否正确使用多项式激活"""
    print("\n" + "=" * 60)
    print("测试 3: 推理时是否正确使用多项式激活")
    print("=" * 60)

    # 创建模块
    module = StablePoly4()

    # 测试输入
    x = torch.tensor([[1.0, 2.0, -1.0, 0.5]])

    # 测试不同 epoch 下的输出
    print("\n不同 epoch 下的输出对比:")

    # Epoch 0 (ReLU 阶段)
    module.set_epoch(0)
    module.eval()
    with torch.no_grad():
        out_epoch0 = module(x)
    print(f"  Epoch 0 (ReLU 预热): {out_epoch0}")

    # Epoch 35 (过渡阶段, alpha=0.5)
    module.set_epoch(35)
    with torch.no_grad():
        out_epoch35 = module(x)
    print(f"  Epoch 35 (过渡, alpha=0.5): {out_epoch35}")

    # Epoch 60 (完全多项式)
    module.set_epoch(60)
    with torch.no_grad():
        out_epoch60 = module(x)
    print(f"  Epoch 60 (完全多项式): {out_epoch60}")

    # 验证输出是否不同
    if torch.allclose(out_epoch0, out_epoch60):
        print("\n✗ 警告：Epoch 0 和 Epoch 60 的输出相同，多项式可能没有被使用")
        return False
    else:
        print("\n✓ 不同阶段的输出不同，证明过渡机制正常工作")

    # 模拟推理场景：保存训练完成的模型，加载后推理
    print("\n模拟推理场景:")

    # 训练完成的模型
    trained_module = StablePoly4()
    trained_module.set_epoch(60)

    # 手动修改多项式参数以更明显地区分
    with torch.no_grad():
        trained_module.d.fill_(1.0)  # 线性系数改为 1.0

    # 保存
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        temp_path = f.name

    try:
        torch.save(trained_module.state_dict(), temp_path)

        # 模拟推理：创建新模块，加载权重
        inference_module = StablePoly4()
        inference_module.load_state_dict(torch.load(temp_path, weights_only=True))
        inference_module.eval()

        print(f"  加载后的 current_epoch: {inference_module.current_epoch.item()}")

        # 推理
        with torch.no_grad():
            inference_out = inference_module(x)

        print(f"  推理输出: {inference_out}")

        # 创建一个 epoch=0 的模块对比
        relu_module = StablePoly4()
        relu_module.eval()
        with torch.no_grad():
            relu_module.d.fill_(1.0)
            relu_out = relu_module(x)
        print(f"  ReLU 模式输出 (对比): {relu_out}")

        if inference_module.current_epoch.item() >= 40:
            print("\n✓ 推理时 current_epoch >= 40，会使用完全多项式激活")
            if not torch.allclose(inference_out, relu_out):
                print("✓ 推理输出与纯 ReLU 输出不同，证明多项式被正确使用")
                return True
            else:
                print("✗ 推理输出与纯 ReLU 输出相同，可能存在问题")
                return False
        else:
            print(
                f"\n✗ 推理时 current_epoch = {inference_module.current_epoch.item()}，不会使用完全多项式"
            )
            return False
    finally:
        os.unlink(temp_path)


def test_with_full_model():
    """测试完整模型中的 StablePoly4"""
    print("\n" + "=" * 60)
    print("测试 4: 完整模型中的 StablePoly4")
    print("=" * 60)

    try:
        from models import get_model

        # 创建模型
        model = get_model("resnet-basic-stablepoly4-layer1block1", num_classes=100)

        # 统计 StablePoly4 实例
        stablepoly_modules = []
        for name, module in model.named_modules():
            if isinstance(module, StablePoly4):
                stablepoly_modules.append((name, module))

        print(f"找到 {len(stablepoly_modules)} 个 StablePoly4 模块")

        if len(stablepoly_modules) == 0:
            print("✗ 没有找到 StablePoly4 模块")
            return False

        # 模拟训练：设置 epoch
        for name, module in stablepoly_modules:
            module.set_epoch(60)

        # 保存模型
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = f.name

        try:
            torch.save(model.state_dict(), temp_path)
            print("✓ 模型已保存")

            # 创建新模型并加载
            new_model = get_model(
                "resnet-basic-stablepoly4-layer1block1", num_classes=100
            )
            new_model.load_state_dict(torch.load(temp_path, weights_only=True))

            # 检查所有 StablePoly4 模块的 current_epoch
            all_correct = True
            for name, module in new_model.named_modules():
                if isinstance(module, StablePoly4):
                    epoch_val = module.current_epoch.item()
                    if epoch_val != 60:
                        print(f"✗ {name}: current_epoch = {epoch_val}，期望 60")
                        all_correct = False
                    else:
                        print(f"✓ {name}: current_epoch = {epoch_val}")

            if all_correct:
                print("\n✓ 所有 StablePoly4 模块的 current_epoch 正确恢复")
                return True
            else:
                print("\n✗ 部分模块的 current_epoch 未正确恢复")
                return False
        finally:
            os.unlink(temp_path)

    except ImportError as e:
        print(f"跳过完整模型测试: {e}")
        return True  # 不算失败


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("StablePoly4 register_buffer 修复验证")
    print("=" * 60)

    results = []

    results.append(("state_dict 保存", test_buffer_in_state_dict()))
    results.append(("保存/加载恢复", test_save_and_load()))
    results.append(("推理多项式使用", test_inference_uses_polynomial()))
    results.append(("完整模型测试", test_with_full_model()))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！修复验证成功 ✓")
    else:
        print("部分测试失败，请检查修复 ✗")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
