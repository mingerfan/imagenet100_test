"""
测试内存文件系统功能
验证 /dev/shm 是否可用，以及数据集是否可以正常复制
"""

from data.memory_fs import MemoryFSManager
from pathlib import Path

def test_memory_fs():
    """测试内存文件系统"""
    print("=" * 60)
    print("内存文件系统测试")
    print("=" * 60)
    
    # 数据集路径
    dataset_path = "/home/xuming/Documents/dataset/ImageNet_100"
    
    if not Path(dataset_path).exists():
        print(f"\n❌ 数据集不存在: {dataset_path}")
        return False
    
    # 创建管理器
    print("\n[1] 创建内存FS管理器...")
    manager = MemoryFSManager(
        source_path=dataset_path,
        shm_name="imagenet100_test",
        auto_copy=False  # 不自动复制，先检查
    )
    
    # 检查可用性
    print("\n[2] 检查 /dev/shm 可用性...")
    available, reason = manager.check_shm_available()
    if not available:
        print(f"  ❌ {reason}")
        return False
    print(f"  ✅ {reason}")
    
    # 检查容量
    print("\n[3] 检查容量...")
    enough, reason = manager.check_capacity()
    if not enough:
        print(f"  ❌ {reason}")
        return False
    print(f"  ✅ {reason}")
    
    # 检查是否已复制
    print("\n[4] 检查是否已复制...")
    if manager.is_copied():
        print("  ✅ 数据已在内存FS中")
    else:
        print("  ℹ️  数据未复制到内存FS")
        
        # 询问是否复制
        response = input("\n  是否现在复制数据集到内存FS？(y/n): ")
        if response.lower() == 'y':
            manager.copy_to_shm()
        else:
            print("  跳过复制")
            return True
    
    # 获取有效路径
    print("\n[5] 获取有效路径...")
    manager.auto_copy = False  # 已手动复制，不再自动
    effective_path = manager.get_effective_path()
    print(f"  ✅ 有效路径: {effective_path}")
    
    # 验证路径
    print("\n[6] 验证路径...")
    train_path = effective_path / "train"
    val_path = effective_path / "val"
    
    if train_path.exists() and val_path.exists():
        print(f"  ✅ 训练集: {train_path}")
        print(f"  ✅ 验证集: {val_path}")
        
        # 统计文件
        train_files = sum(1 for _ in train_path.rglob('*') if _.is_file())
        val_files = sum(1 for _ in val_path.rglob('*') if _.is_file())
        print(f"  ✅ 训练集文件数: {train_files:,}")
        print(f"  ✅ 验证集文件数: {val_files:,}")
    else:
        print("  ❌ 路径验证失败")
        return False
    
    print(f"\n{'=' * 60}")
    print("✅ 所有测试通过！")
    print(f"{'=' * 60}")
    print("\n提示: 现在可以运行训练:")
    print("  python train.py --use_memory_fs")
    print("\n或者清理内存FS:")
    print(f"  python -c \"from data.memory_fs import MemoryFSManager; m = MemoryFSManager('{dataset_path}', 'imagenet100_test', False); m.cleanup()\"")
    
    return True


if __name__ == '__main__':
    try:
        success = test_memory_fs()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试已取消")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
