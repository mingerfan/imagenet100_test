"""
配置文件管理
"""

import yaml
import os
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


def get_model_configs(config: Dict) -> List[Dict]:
    """
    从配置中获取模型列表
    
    Args:
        config: 配置字典
    
    Returns:
        模型配置列表
    """
    return config.get('models', [])