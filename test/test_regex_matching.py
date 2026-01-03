"""
测试正则匹配配置功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_config, get_model_configs
from models import MODEL_REGISTRY


def test_regex_matching():
    """测试正则匹配功能"""
    print("=" * 60)
    print("测试正则匹配配置功能")
    print("=" * 60)
    
    # 加载配置
    config_path = 'configs/models_list.yaml'
    if not os.path.exists(config_path):
        print(f"⚠ 配置文件不存在: {config_path}")
        return False
    
    config = load_config(config_path)
    print(f"\n✓ 加载配置文件: {config_path}")
    
    # 获取已注册的模型
    registered_models = MODEL_REGISTRY.list_models()
    print(f"\n✓ 已注册的模型: {len(registered_models)} 个")
    
    # 获取模型配置（包括正则匹配）
    model_configs = get_model_configs(config, registered_models)
    print(f"\n✓ 生成的模型配置: {len(model_configs)} 个")
    
    # 分类显示
    explicit_models = [m for m in model_configs if m['name'] in [
        'resnet18', 'resnet34', 'resnet50'
    ]]
    matched_models = [m for m in model_configs if m['name'] not in [
        'resnet18', 'resnet34', 'resnet50'
    ]]
    
    print(f"\n  - 显式指定的模型: {len(explicit_models)} 个")
    for m in explicit_models:
        print(f"    {m['name']}")
    
    print(f"\n  - 正则匹配的模型: {len(matched_models)} 个")
    for m in matched_models:
        print(f"    {m['name']}")
        print(f"      Epochs: {m.get('epochs', 60)}, "
              f"Batch: {m.get('batch_size', 128)}, "
              f"LR: {m.get('learning_rate', 0.001)}")
    
    # 验证
    if len(matched_models) > 0:
        print("\n✓ 正则匹配功能正常工作！")
        return True
    else:
        print("\n⚠ 没有匹配到任何模型")
        return False


if __name__ == '__main__':
    success = test_regex_matching()
    print("\n" + "=" * 60)
    if success:
        print("测试通过！✓")
    else:
        print("测试失败！✗")
    print("=" * 60)
    sys.exit(0 if success else 1)