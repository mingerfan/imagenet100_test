"""
内存文件系统管理器
自动将数据集复制到 /dev/shm 以加速数据加载
包含进程间锁机制以防止并发复制问题
"""

import os
import shutil
import time
import fcntl
from pathlib import Path
from typing import Optional, Tuple


class MemoryFSManager:
    """内存文件系统管理器"""
    
    def __init__(self, 
                 source_path: str,
                 shm_name: str = "imagenet100",
                 auto_copy: bool = True):
        """
        初始化内存文件系统管理器
        
        Args:
            source_path: 原始数据集路径
            shm_name: 在/dev/shm中的名称
            auto_copy: 是否自动复制到内存FS
        """
        self.source_path = Path(source_path).resolve()
        self.shm_name = shm_name
        self.shm_path = Path(f"/dev/shm/{shm_name}")
        self.auto_copy = auto_copy
        
        # 锁文件路径，用于防止并发复制
        self.lock_file_path = Path(f"/dev/shm/.{shm_name}.lock")
        self.lock_file = None
        
        print(f"[MemoryFS] 原始路径: {self.source_path}")
        print(f"[MemoryFS] 内存FS路径: {self.shm_path}")
        print(f"[MemoryFS] 锁文件路径: {self.lock_file_path}")
    
    def _acquire_lock(self):
        """
        获取进程间锁，防止并发复制
        
        使用fcntl.flock实现跨进程的文件锁机制
        """
        try:
            # 创建锁文件（如果不存在）
            self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.lock_file = open(self.lock_file_path, 'w')
            
            # 尝试获取排他锁（非阻塞模式）
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                print(f"[MemoryFS] ✓ 获取锁成功，开始复制...")
                return True
            except (IOError, BlockingIOError):
                # 锁已被其他进程持有，等待
                print(f"[MemoryFS] 检测到其他进程正在复制数据，等待完成...")
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)  # 阻塞等待
                print(f"[MemoryFS] ✓ 等待完成，数据已复制")
                self._release_lock()
                return False
        except Exception as e:
            print(f"[MemoryFS] ⚠ 锁机制异常: {e}")
            if self.lock_file:
                self.lock_file.close()
                self.lock_file = None
            return True  # 出错时继续执行
    
    def _release_lock(self):
        """
        释放锁
        """
        if self.lock_file:
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                self.lock_file.close()
            except Exception as e:
                print(f"[MemoryFS] ⚠ 释放锁异常: {e}")
            finally:
                self.lock_file = None
    
    def check_shm_available(self) -> Tuple[bool, str]:
        """
        检查内存文件系统是否可用
        
        Returns:
            (是否可用, 原因/信息)
        """
        # 1. 检查 /dev/shm 是否存在
        if not Path("/dev/shm").exists():
            return False, "/dev/shm 不存在"
        
        # 2. 检查是否可写
        test_path = Path("/dev/shm/.test_write")
        try:
            test_path.touch()
            test_path.unlink()
        except Exception as e:
            return False, f"无法写入 /dev/shm: {e}"
        
        return True, "内存文件系统可用"
    
    def get_shm_size(self) -> int:
        """
        获取 /dev/shm 的总大小（字节）
        
        Returns:
            大小（字节）
        """
        stat = os.statvfs("/dev/shm")
        return stat.f_bsize * stat.f_blocks
    
    def get_dataset_size(self, path: Optional[Path] = None) -> int:
        """
        获取数据集大小（字节）
        
        Args:
            path: 要计算大小的路径，默认为source_path
        
        Returns:
            大小（字节）
        """
        target_path = path if path is not None else self.source_path
        total_size = 0
        
        for dirpath, _, filenames in os.walk(target_path):
            for filename in filenames:
                filepath = Path(dirpath) / filename
                if filepath.is_file():
                    try:
                        total_size += filepath.stat().st_size
                    except OSError:
                        pass
        
        return total_size
    
    def check_capacity(self) -> Tuple[bool, str]:
        """
        检查 /dev/shm 容量是否足够
        
        Returns:
            (是否足够, 原因/信息)
        """
        shm_size = self.get_shm_size()
        dataset_size = self.get_dataset_size()
        usage = dataset_size / shm_size * 100
        
        print(f"[MemoryFS] 内存FS容量: {shm_size / 1024**3:.2f} GB")
        print(f"[MemoryFS] 数据集大小: {dataset_size / 1024**3:.2f} GB")
        print(f"[MemoryFS] 预估占用: {usage:.1f}%")
        
        if dataset_size > shm_size * 0.9:  # 留10%余量
            return False, f"数据集({dataset_size/1024**3:.2f}GB) " \
                       f"超过内存FS容量({shm_size/1024**3:.2f}GB)的90%"
        
        return True, f"容量充足 (占用{usage:.1f}%)"
    
    def is_copied(self) -> bool:
        """
        检查数据是否已复制到内存FS
        
        Returns:
            是否已复制
        """
        if not self.shm_path.exists():
            return False
        
        # 检查关键目录是否存在
        src_subdirs = []
        for item in self.source_path.iterdir():
            if item.is_dir():
                src_subdirs.append(item.name)
        
        # 检查所有子目录是否都已复制
        for subdir in src_subdirs:
            if not (self.shm_path / subdir).exists():
                return False
        
        return True
    
    def copy_to_shm(self, show_progress: bool = True):
        """
        复制数据集到内存文件系统（带进程间锁保护）
        
        Args:
            show_progress: 是否显示复制进度
        """
        # 获取锁，防止并发复制
        if not self._acquire_lock():
            # 其他进程正在复制，已等待完成，直接返回
            return
        
        try:
            print(f"\n[MemoryFS] 开始复制数据集到内存文件系统...")
            print(f"[MemoryFS] 源: {self.source_path}")
            print(f"[MemoryFS] 目标: {self.shm_path}")
            
            start_time = time.time()
            
            # 创建目标目录
            self.shm_path.mkdir(parents=True, exist_ok=True)
            
            # 复制所有子目录
            for src_item in self.source_path.iterdir():
                if src_item.is_dir():
                    dst_item = self.shm_path / src_item.name
                    
                    # 如果目标已存在，跳过
                    if dst_item.exists():
                        print(f"[MemoryFS] 目标已存在: {src_item.name}，跳过")
                        continue
                    
                    self._copy_dir(src_item, dst_item, show_progress)
            
            elapsed = time.time() - start_time
            print(f"[MemoryFS] ✓ 复制完成! 耗时: {elapsed:.2f} 秒")
            
            # 显示统计信息
            self._print_stats()
        finally:
            # 确保释放锁
            self._release_lock()
    
    def _copy_dir(self, src: Path, dst: Path, show_progress: bool):
        """
        复制目录
        
        Args:
            src: 源目录
            dst: 目标目录
            show_progress: 是否显示进度
        """
        print(f"[MemoryFS] 复制 {src.name}...")
        
        # 统计文件数量
        file_count = sum(1 for _ in src.rglob('*') if _.is_file())
        print(f"[MemoryFS]   文件数量: {file_count:,}")
        
        # 执行复制
        if show_progress:
            # 使用 rsync 显示进度（如果可用）
            try:
                import subprocess
                result = subprocess.run([
                    'rsync', '-av', '--progress',
                    str(src) + '/', str(dst)
                ], check=True, capture_output=True, text=True)
                
                # 显示最后几行进度信息
                lines = result.stderr.split('\n')
                for line in lines[-5:]:
                    if line:
                        print(f"[MemoryFS]   {line}")
            except (FileNotFoundError, subprocess.CalledProcessError):
                # rsync不可用，使用shutil
                shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=False)
        else:
            shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=False)
    
    def _print_stats(self):
        """打印统计信息"""
        dataset_size = self.get_dataset_size(self.shm_path)
        
        # 计算文件数量
        file_count = sum(1 for _ in self.shm_path.rglob('*') if _.is_file())
        
        print(f"[MemoryFS] 文件总数: {file_count:,}")
        print(f"[MemoryFS] 总大小: {dataset_size / 1024**3:.2f} GB")
    
    def get_effective_path(self) -> Path:
        """
        获取实际使用的数据路径
        
        Returns:
            优先返回内存FS路径（如果可用且已复制），否则返回原始路径
        """
        print(f"\n[MemoryFS] 检查是否使用内存文件系统...")
        
        # 1. 检查内存FS是否可用
        available, reason = self.check_shm_available()
        if not available:
            print(f"[MemoryFS] ✗ 使用原始路径: {reason}")
            return self.source_path
        
        # 2. 检查容量
        enough, reason = self.check_capacity()
        if not enough:
            print(f"[MemoryFS] ✗ 使用原始路径: {reason}")
            return self.source_path
        
        # 3. 自动复制（如果启用且未复制）
        if self.auto_copy and not self.is_copied():
            self.copy_to_shm()
        elif not self.is_copied():
            print(f"[MemoryFS] 使用原始路径: auto_copy=False")
            return self.source_path
        
        # 4. 检查是否已复制
        if self.is_copied():
            print(f"[MemoryFS] ✓ 使用内存FS路径: {self.shm_path}")
            return self.shm_path
        else:
            print(f"[MemoryFS] 使用原始路径: 数据未复制到内存FS")
            return self.source_path
    
    def cleanup(self):
        """清理内存文件系统中的数据"""
        if self.shm_path.exists():
            print(f"[MemoryFS] 清理内存FS: {self.shm_path}")
            shutil.rmtree(self.shm_path)
        else:
            print(f"[MemoryFS] 内存FS中无数据: {self.shm_path}")


def create_memory_fs_manager(train_dir: str,
                              val_dir: str,
                              use_memory_fs: bool = True,
                              shm_name: Optional[str] = None) -> Optional[MemoryFSManager]:
    """
    创建内存文件系统管理器
    
    Args:
        train_dir: 训练集目录
        val_dir: 验证集目录
        use_memory_fs: 是否使用内存文件系统
        shm_name: /dev/shm 中的名称（为空时使用默认值）
    
    Returns:
        MemoryFSManager实例或None
    """
    if not use_memory_fs:
        print("[MemoryFS] 内存文件系统已禁用")
        return None
    
    # 确定数据集根目录
    train_path = Path(train_dir).resolve()
    val_path = Path(val_dir).resolve()
    
    # 尝试找到共同的父目录
    common_parent = None
    for parent in train_path.parents:
        if val_path.is_relative_to(parent):
            common_parent = parent
            break
    
    if common_parent is None:
        print("[MemoryFS] 训练集和验证集不在同一目录，无法使用内存FS")
        return None
    
    # 创建管理器
    manager = MemoryFSManager(
        source_path=str(common_parent),
        shm_name=shm_name or "imagenet100",
        auto_copy=True
    )
    
    return manager
