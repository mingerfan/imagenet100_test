"""
配置文件管理
"""

import yaml
import os
import re
from typing import Dict, List


def load_config(config_path: str) -> Dict:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict, save_path: str):
    """
    保存配置到YAML文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def get_model_configs(config: Dict, registered_models: List[str] = None) -> List[Dict]:
    """
    从配置中获取模型列表，支持正则匹配
    
    Args:
        config: 配置字典
        registered_models: 已注册的模型名称列表（用于正则匹配）
    
    Returns:
        模型配置列表
    """
    models = config.get('models', [])
    patterns = config.get('model_patterns', [])
    
    # 如果没有提供已注册模型列表或没有模式配置，直接返回显式模型列表
    if not patterns or registered_models is None:
        return models
    
    # 获取全局设置作为默认值
    global_config = config.get('global', {})
    default_config = {
        'num_classes': global_config.get('num_classes', 100),
        'pretrained': False,
        'epochs': global_config.get('default_epochs', 60),
        'batch_size': global_config.get('default_batch_size', 128),
        'learning_rate': global_config.get('default_learning_rate', 0.001),
        'num_workers': global_config.get('default_num_workers', 16)
    }
    
    # 收集所有显式指定的模型名称
    explicit_model_names = set(m['name'] for m in models)
    
    # 对每个正则模式进行匹配
    for pattern_config in patterns:
        pattern = pattern_config['pattern']
        
        try:
            regex = re.compile(pattern)
        except re.error as e:
            print(f"⚠ 无效的正则表达式 '{pattern}': {e}")
            continue
        
        # 匹配已注册的模型
        matched_models = []
        for model_name in registered_models:
            # 跳过已经在显式列表中的模型
            if model_name in explicit_model_names:
                continue
            
            # 检查是否匹配正则表达式
            if regex.match(model_name):
                matched_models.append(model_name)
        
        # 为匹配的模型创建配置
        for model_name in matched_models:
            # 合并默认配置、模式配置和模型特定配置
            model_config = {
                'name': model_name,
                'class': model_name,  # 默认使用模型名称作为类名
                **default_config,
                **{k: v for k, v in pattern_config.items() if k != 'pattern'}  # 排除 pattern 字段
            }
            
            # 如果有 params，合并默认参数
            if 'params' not in model_config:
                model_config['params'] = {
                    'num_classes': default_config['num_classes'],
                    'pretrained': default_config['pretrained']
                }
            else:
                model_config['params'].setdefault('num_classes', default_config['num_classes'])
                model_config['params'].setdefault('pretrained', default_config['pretrained'])
            
            models.append(model_config)
            explicit_model_names.add(model_name)
    
    return models