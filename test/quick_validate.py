"""
快速验证脚本 - 在正式训练前测试所有模型

功能：
1. 创建所有配置的模型
2. 运行一次前向传播和反向传播
3. 验证数据加载是否正常
4. 验证GPU内存是否充足
5. 如果任何步骤失败，立即停止并报错

使用方法：
    uv run python quick_validate.py
    uv run python quick_validate.py --models resnet18 resnet34
"""

import argparse
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from models import get_model, MODEL_REGISTRY
from utils import load_config, get_model_configs, set_random_seed
from data import create_dataloaders


def check_tensor_finite(tensor, name="tensor"):
    """
    检查张量是否包含NaN或Inf值
    
    Args:
        tensor: 要检查的张量
        name: 张量名称（用于错误消息）
    
    Returns:
        (是否有限, 错误信息)
    """
    if tensor is None:
        return True, None
    
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        
        error_msg = (
            f"{name} 包含无效值: "
            f"NaN={nan_count}, Inf={inf_count}, "
            f"形状={tensor.shape}, "
            f"值范围=[{tensor.min().item():.4f}, {tensor.max().item():.4f}]"
        )
        return False, error_msg
    
    return True, None


def check_gradients_finite(model, verbose=False):
    """
    检查模型的所有梯度是否有限
    
    Args:
        model: 要检查的模型
        verbose: 是否打印详细梯度信息
    
    Returns:
        (是否有限, 错误信息)
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            is_finite, error_msg = check_tensor_finite(param.grad, f"梯度[{name}]")
            if not is_finite:
                return False, error_msg
            
            if verbose:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1000:  # 警告：梯度可能过大
                    print(f"    ⚠️  警告: {name} 梯度范数过大: {grad_norm:.2f}")
    
    return True, None


def check_parameters_finite(model, verbose=False):
    """
    检查模型的所有参数是否有限
    
    Args:
        model: 要检查的模型
        verbose: 是否打印详细参数信息
    
    Returns:
        (是否有限, 错误信息)
    """
    for name, param in model.named_parameters():
        is_finite, error_msg = check_tensor_finite(param, f"参数[{name}]")
        if not is_finite:
            return False, error_msg
        
        if verbose:
            param_norm = param.norm().item()
            if param_norm > 1000:  # 警告：参数可能过大
                print(f"    ⚠️  警告: {name} 参数范数过大: {param_norm:.2f}")
    
    return True, None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="快速验证模型结构")

    parser.add_argument(
        "--config", type=str, default="configs/models_list.yaml", help="模型配置文件"
    )

    parser.add_argument(
        "--train_dir",
        type=str,
        default="/home/xuming/Documents/dataset/ImageNet_100/train",
        help="训练集目录",
    )

    parser.add_argument(
        "--val_dir",
        type=str,
        default="/home/xuming/Documents/dataset/ImageNet_100/val",
        help="验证集目录",
    )

    parser.add_argument(
        "--gpus", type=int, nargs="+", default=[0], help="使用的GPU列表"
    )

    parser.add_argument("--batch_size", type=int, default=32, help="验证批次大小")

    parser.add_argument(
        "--num_batches", type=int, default=5, help="每个模型测试的批次数"
    )

    parser.add_argument(
        "--models", type=str, nargs="+", default=None, help="只验证指定的模型"
    )

    parser.add_argument("--use_cache", action="store_true", help="使用内存缓存数据集")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def validate_model(model, dataloader, device, num_batches=5):
    """
    验证单个模型

    Args:
        model: 待验证的模型
        dataloader: 数据加载器
        device: 设备
        num_batches: 测试的批次数

    Returns:
        (是否成功, 错误信息)
    """
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    try:
        print(f"    ✓ 模型已移动到设备: {device}")

        # 测试前向传播和反向传播
        for batch_idx, (images, labels) in enumerate(
            tqdm(dataloader, desc="    测试", leave=False)
        ):
            if batch_idx >= num_batches:
                break

            # 移动数据到设备
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 检查输出是否包含NaN/Inf
            is_finite, error_msg = check_tensor_finite(outputs, "模型输出")
            if not is_finite:
                return False, f"前向传播失败 - {error_msg}"
            
            # 检查损失值是否有效
            is_finite, error_msg = check_tensor_finite(loss, "损失值")
            if not is_finite:
                return False, f"损失值无效 - {error_msg}"
            
            # 打印损失值用于监控
            if batch_idx == 0:
                print(f"    ✓ 初始损失值: {loss.item():.4f}")
            
            # 检查损失值是否过大（可能是梯度爆炸的早期迹象）
            if loss.item() > 1000:
                return False, f"损失值过大: {loss.item():.4f}, 可能导致梯度爆炸"

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 检查梯度是否包含NaN/Inf
            is_finite, error_msg = check_gradients_finite(model, verbose=(batch_idx == 0))
            if not is_finite:
                return False, f"反向传播失败 - {error_msg}"
            
            optimizer.step()
            
            # 检查更新后的参数是否有效
            is_finite, error_msg = check_parameters_finite(model, verbose=(batch_idx == 0))
            if not is_finite:
                return False, f"参数更新失败 - {error_msg}"

            # 验证输出形状
            batch_size, num_classes = outputs.shape
            if num_classes != 100:
                return False, f"输出类别数错误: {num_classes}, 期望: 100"

        # 测试推理模式
        model.eval()
        with torch.no_grad():
            images, labels = next(iter(dataloader))
            images = images.to(device)
            outputs = model(images)

            # 测试预测
            _, predicted = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, dim=1)

        print(f"    ✓ 成功完成 {num_batches} 批次训练")
        print(f"    ✓ 推理模式测试通过")

        # GPU内存使用情况
        if device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(
                f"    ✓ GPU内存使用: {memory_allocated:.2f}GB (已分配) / {memory_reserved:.2f}GB (已保留)"
            )

        return True, None

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            return False, f"GPU内存不足: {str(e)}"
        return False, f"运行时错误: {str(e)}"
    except Exception as e:
        return False, f"未知错误: {str(e)}"

    finally:
        # 清理GPU内存
        if device.type == "cuda":
            torch.cuda.empty_cache()


def quick_validate(args):
    """快速验证所有模型"""
    print("=" * 80)
    print("快速验证 - 模型结构和数据加载测试")
    print("=" * 80)

    # 加载配置
    print("\n[1/5] 加载配置文件...")
    try:
        config = load_config(args.config)
        
        # 获取已注册的模型列表（用于正则匹配）
        registered_models = list(MODEL_REGISTRY._registry.keys())
        print(f"  已注册模型数量: {len(registered_models)}")
        
        # 使用已注册模型列表进行匹配
        model_configs = get_model_configs(config, registered_models)

        # 过滤指定模型
        if args.models:
            model_configs = [m for m in model_configs if m["name"] in args.models]
            if not model_configs:
                print(f"❌ 未找到指定的模型: {args.models}")
                return False

        print(f"✓ 配置加载成功，共 {len(model_configs)} 个模型待验证")
        for i, m in enumerate(model_configs, 1):
            print(f"  {i}. {m['name']}")

    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

    # 检查GPU
    print("\n[2/5] 检查GPU...")
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False

    device = torch.device(f"cuda:{args.gpus[0]}")
    props = torch.cuda.get_device_properties(args.gpus[0])
    print(f"✓ GPU {args.gpus[0]}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    # 创建数据加载器
    print("\n[3/5] 创建数据加载器...")
    try:
        train_loader, val_loader, _, _ = create_dataloaders(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            seed=args.seed,
        )
        print(f"✓ 训练集加载器创建成功")
        print(f"  批次大小: {args.batch_size}")
        print(f"  训练集批次数: {len(train_loader)}")
        print(f"  验证集批次数: {len(val_loader)}")

        # 测试数据加载
        print("\n  测试数据加载...")
        images, labels = next(iter(train_loader))
        print(f"  ✓ 数据形状: {images.shape}, 标签形状: {labels.shape}")
        print(f"  ✓ 数据类型: {images.dtype}, 标签类型: {labels.dtype}")
        print(f"  ✓ 数据范围: [{images.min():.2f}, {images.max():.2f}]")

    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 验证所有模型
    print("\n[4/5] 验证所有模型...")
    print("=" * 80)

    success_count = 0
    failed_models = []

    for i, model_config in enumerate(model_configs, 1):
        model_name = model_config["name"]
        print(f"\n[{i}/{len(model_configs)}] 验证模型: {model_name}")
        print("-" * 80)

        try:
            # 创建模型
            model_params = model_config.get("params", {})
            model_params["num_classes"] = model_params.get("num_classes", 100)
            model_params["pretrained"] = False  # 验证时不使用预训练权重以加快速度

            model = get_model(model_name, **model_params)

            # 统计参数
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            print(f"  总参数量: {total_params:,}")
            print(f"  可训练参数: {trainable_params:,}")

            # 验证模型
            success, error_msg = validate_model(
                model, train_loader, device, num_batches=args.num_batches
            )

            if success:
                print(f"\n  ✅ {model_name} 验证通过")
                success_count += 1
            else:
                print(f"\n  ❌ {model_name} 验证失败: {error_msg}")
                print(f"  🔴 快速失败！停止验证")
                failed_models.append((model_name, error_msg))
                break

        except Exception as e:
            print(f"\n  ❌ {model_name} 验证异常: {e}")
            import traceback

            traceback.print_exc()
            failed_models.append((model_name, str(e)))
            break

    # 清理GPU内存
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 总结
    print("\n[5/5] 验证总结")
    print("=" * 80)

    if failed_models:
        print(f"\n❌ 验证失败！")
        for model_name, error in failed_models:
            print(f"  - {model_name}: {error}")
        print(f"\n🔴 带病上路风险！请修复上述错误后再进行训练")
        return False

    print(f"\n✅ 所有模型验证通过！({success_count}/{success_count})")
    print(f"\n🎉 系统健康，可以开始训练")
    print("\n下一步:")
    print("  uv run python train.py")
    print("  uv run python train.py --models resnet18  # 只训练resnet18")

    return True


def main():
    """主函数"""
    args = parse_args()
    set_random_seed(args.seed)

    success = quick_validate(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
