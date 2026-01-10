#!/usr/bin/env python
"""
快速示例：使用批量分析工具

这个脚本展示了如何快速使用批量FHE统计分析工具。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fhe_statistics.batch_analyzer import BatchAnalyzer


def example_1_list_models():
    """示例1：列出配置文件中的所有模型"""
    print("\n" + "="*80)
    print("示例1：列出所有模型")
    print("="*80)

    analyzer = BatchAnalyzer("fhe_statistics/batch_analysis_config.yaml")
    analyzer.list_models()


def example_2_analyze_specific_models():
    """示例2：只分析特定的几个模型"""
    print("\n" + "="*80)
    print("示例2：分析特定模型")
    print("="*80)

    analyzer = BatchAnalyzer("fhe_statistics/batch_analysis_config.yaml")

    # 只分析 ResNet18 和 MobileNetV2
    analyzer.run(specific_models=["ResNet18", "MobileNetV2"])


def example_3_programmatic_config():
    """示例3：通过代码创建配置并分析"""
    print("\n" + "="*80)
    print("示例3：通过代码配置")
    print("="*80)

    # 创建临时配置
    import yaml
    import tempfile
    import os

    config = {
        'global': {
            'output_folder': 'fhe_statistics/results',
            'plot_folder': 'fhe_statistics/plots',
            'default_input_shape': [1, 3, 224, 224],
            'print_detailed': False,  # 为了快速演示，禁用详细输出
            'generate_plots': True,
            'generate_comparison': True,
        },
        'models': [
            {
                'name': 'ResNet18',
                'source': 'torchvision',
                'model_class': 'resnet18',
                'input_shape': [1, 3, 224, 224],
                'enabled': True,
            },
            {
                'name': 'ResNet18-96x96',
                'source': 'torchvision',
                'model_class': 'resnet18',
                'input_shape': [1, 3, 96, 96],
                'enabled': True,
            },
        ],
        'comparison': {
            'plot_types': ['network_comparison', 'comprehensive_comparison'],
        }
    }

    # 保存临时配置
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_config_path = f.name

    try:
        # 使用临时配置运行分析
        analyzer = BatchAnalyzer(temp_config_path)
        analyzer.run()
    finally:
        # 清理临时文件
        os.unlink(temp_config_path)


def example_4_compare_resolutions():
    """示例4：比较不同输入分辨率的影响"""
    print("\n" + "="*80)
    print("示例4：比较不同输入分辨率")
    print("="*80)

    import yaml
    import tempfile
    import os

    # 创建配置：同一个模型，不同分辨率
    config = {
        'global': {
            'output_folder': 'fhe_statistics/results',
            'plot_folder': 'fhe_statistics/plots',
            'default_input_shape': [1, 3, 224, 224],
            'print_detailed': False,
            'generate_plots': True,
            'generate_comparison': True,
        },
        'models': [
            {
                'name': 'MobileNetV2-64x64',
                'source': 'torchvision',
                'model_class': 'mobilenet_v2',
                'input_shape': [1, 3, 64, 64],
                'enabled': True,
            },
            {
                'name': 'MobileNetV2-96x96',
                'source': 'torchvision',
                'model_class': 'mobilenet_v2',
                'input_shape': [1, 3, 96, 96],
                'enabled': True,
            },
            {
                'name': 'MobileNetV2-128x128',
                'source': 'torchvision',
                'model_class': 'mobilenet_v2',
                'input_shape': [1, 3, 128, 128],
                'enabled': True,
            },
            {
                'name': 'MobileNetV2-224x224',
                'source': 'torchvision',
                'model_class': 'mobilenet_v2',
                'input_shape': [1, 3, 224, 224],
                'enabled': True,
            },
        ],
        'comparison': {
            'plot_types': ['network_comparison', 'comprehensive_comparison', 'grouped_comparison'],
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_config_path = f.name

    try:
        analyzer = BatchAnalyzer(temp_config_path)
        analyzer.run()
    finally:
        os.unlink(temp_config_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='FHE批量分析工具示例')
    parser.add_argument('--example', type=int, choices=[1, 2, 3, 4], default=1,
                       help='运行哪个示例 (1-4)')

    args = parser.parse_args()

    if args.example == 1:
        example_1_list_models()
    elif args.example == 2:
        example_2_analyze_specific_models()
    elif args.example == 3:
        example_3_programmatic_config()
    elif args.example == 4:
        example_4_compare_resolutions()
