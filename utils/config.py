"""
配置文件管理
"""

import yaml
import os
import re
from typing import Dict, List
from pathlib import Path


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


def _get_default_config(config: Dict) -> Dict:
    global_config = config.get('global', {})
    return {
        'num_classes': global_config.get('num_classes', 100),
        'pretrained': False,
        'epochs': global_config.get('default_epochs', 60),
        'batch_size': global_config.get('default_batch_size', 128),
        'learning_rate': global_config.get('default_learning_rate', 0.001),
        'num_workers': global_config.get('default_num_workers', 16),
        'save_freq': global_config.get('default_save_freq', 10),
    }


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
    default_config = _get_default_config(config)
    
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


def _collect_json_paths(json_root: str) -> List[Path]:
    root = Path(json_root)
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    return [p for p in root.rglob("*.json") if p.is_file()]


def get_json_model_configs(config: Dict) -> List[Dict]:
    """Build model configs from JSON files (explicit list or regex patterns)."""
    default_config = _get_default_config(config)
    json_models = config.get('json_models', []) or []
    json_patterns = config.get('json_model_patterns', []) or []
    models: List[Dict] = []

    # Explicit JSON list
    for item in json_models:
        json_path = item.get('json_path') or item.get('path')
        if not json_path:
            print("Warning: json_models entry missing json_path")
            continue
        model_name = item.get('name') or Path(json_path).stem
        model_config = {
            'name': model_name,
            'class': item.get('class', 'nas-json'),
            **default_config,
            **{k: v for k, v in item.items() if k not in ('json_path', 'path', 'class', 'name')}
        }
        params = dict(model_config.get('params', {}) or {})
        params.setdefault('num_classes', default_config['num_classes'])
        params['json_path'] = json_path
        model_config['params'] = params
        models.append(model_config)

    # Regex-based JSON patterns
    for pattern_config in json_patterns:
        pattern = pattern_config.get('pattern')
        if not pattern:
            print("Warning: json_model_patterns entry missing pattern")
            continue
        try:
            regex = re.compile(pattern)
        except re.error as e:
            print(f"Warning: invalid regex '{pattern}': {e}")
            continue

        json_root = pattern_config.get('json_root') or pattern_config.get('json_dir')
        if not json_root:
            print(f"Warning: json_model_patterns '{pattern}' missing json_root")
            continue

        json_paths = _collect_json_paths(json_root)
        if not json_paths:
            print(f"Warning: no json files found in {json_root}")
            continue

        name_prefix = pattern_config.get('name_prefix', '')
        for path in json_paths:
            rel = str(path.relative_to(json_root).as_posix())
            if not (regex.search(path.name) or regex.search(rel)):
                continue
            model_name = f"{name_prefix}{path.stem}" if name_prefix else path.stem
            model_config = {
                'name': model_name,
                'class': pattern_config.get('class', 'nas-json'),
                **default_config,
                **{k: v for k, v in pattern_config.items() if k not in ('pattern', 'json_root', 'json_dir', 'name_prefix', 'class')}
            }
            params = dict(model_config.get('params', {}) or {})
            params.setdefault('num_classes', default_config['num_classes'])
            params['json_path'] = str(path)
            model_config['params'] = params
            models.append(model_config)

    return models
