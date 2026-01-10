"""
批量FHE统计分析工具

从配置文件读取模型列表，批量进行FHE统计分析，并生成综合比较报告。

Usage:
    python batch_analyzer.py --config fhe_statistics/batch_analysis_config.yaml
    python batch_analyzer.py --config myconfig.yaml --models ResNet18 ResNet34  # 只分析指定模型
    python batch_analyzer.py --list  # 列出配置文件中的所有模型
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml
import re

import torch
import torch.nn as nn
import torchvision.models

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fhe_statistics.statistics_fn import FheInfo, analyze_model


class BatchAnalyzer:
    """批量FHE统计分析器"""

    def __init__(self, config_path: str):
        """初始化批量分析器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.results: Dict[str, FheInfo] = {}

    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _get_global_setting(self, key: str, default: Any = None) -> Any:
        """获取全局设置"""
        return self.config.get('global', {}).get(key, default)

    def _load_torchvision_model(self, model_class: str, params: Optional[Dict] = None) -> nn.Module:
        """加载TorchVision预训练模型

        Args:
            model_class: 模型类名（如 'resnet18'）
            params: 模型参数（可选）

        Returns:
            加载的模型
        """
        if not hasattr(torchvision.models, model_class):
            raise ValueError(f"TorchVision does not have model: {model_class}")

        model_fn = getattr(torchvision.models, model_class)

        # 处理参数
        if params is None:
            params = {}

        # 创建模型
        model = model_fn(**params)
        return model

    def _load_custom_model(self, module_path: str, model_class: str,
                          params: Optional[Dict] = None) -> nn.Module:
        """加载自定义模型

        Args:
            module_path: 模块路径（如 'models.gate_net'）
            model_class: 模型类名（如 'resnet18'）
            params: 模型参数

        Returns:
            加载的模型
        """
        # 动态导入模块
        import importlib
        module = importlib.import_module(module_path)

        if not hasattr(module, model_class):
            raise ValueError(f"Module {module_path} does not have class: {model_class}")

        model_fn = getattr(module, model_class)

        # 处理参数
        if params is None:
            params = {}

        # 创建模型
        model = model_fn(**params)
        return model

    def _load_checkpoint_model(self, module_path: str, model_class: str,
                              checkpoint_path: str, params: Optional[Dict] = None) -> nn.Module:
        """从checkpoint加载模型

        Args:
            module_path: 模块路径
            model_class: 模型类名
            checkpoint_path: checkpoint文件路径
            params: 模型参数

        Returns:
            加载的模型
        """
        # 先创建模型
        model = self._load_custom_model(module_path, model_class, params)

        # 加载checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 根据checkpoint结构加载权重
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        return model

    def _load_model(self, model_config: Dict) -> Tuple[str, nn.Module, Tuple[int, ...]]:
        """根据配置加载模型

        Args:
            model_config: 模型配置字典

        Returns:
            (模型名称, 模型对象, 输入形状)
        """
        name = model_config['name']
        source = model_config['source']
        input_shape = tuple(model_config.get('input_shape', self._get_global_setting('default_input_shape', [1, 3, 224, 224])))
        params = model_config.get('params', None)

        print(f"Loading model: {name} (source: {source})")

        if source == 'torchvision':
            model = self._load_torchvision_model(model_config['model_class'], params)
        elif source == 'custom':
            model = self._load_custom_model(
                model_config['module_path'],
                model_config['model_class'],
                params
            )
        elif source == 'checkpoint':
            model = self._load_checkpoint_model(
                model_config['module_path'],
                model_config['model_class'],
                model_config['checkpoint_path'],
                params
            )
        else:
            raise ValueError(f"Unknown model source: {source}")

        return name, model, input_shape

    def list_models(self):
        """列出配置文件中的所有模型"""
        print(f"\n{'='*80}")
        print(f"Models in configuration file: {self.config_path}")
        print(f"{'='*80}\n")

        models = self.config.get('models', [])

        # 按enabled状态分组
        enabled_models = []
        disabled_models = []

        for i, model_config in enumerate(models):
            name = model_config.get('name', f'Model_{i}')
            source = model_config.get('source', 'unknown')
            enabled = model_config.get('enabled', False)
            optional = model_config.get('optional', False)
            description = model_config.get('description', 'No description')
            input_shape = model_config.get('input_shape', self._get_global_setting('default_input_shape'))

            model_info = {
                'name': name,
                'source': source,
                'enabled': enabled,
                'optional': optional,
                'description': description,
                'input_shape': input_shape
            }

            if enabled:
                enabled_models.append(model_info)
            else:
                disabled_models.append(model_info)

        # 打印已启用的模型
        if enabled_models:
            print(f"ENABLED MODELS ({len(enabled_models)}):")
            print(f"{'-'*80}")
            for info in enabled_models:
                optional_tag = " [OPTIONAL]" if info['optional'] else ""
                print(f"  • {info['name']}{optional_tag}")
                print(f"    Source: {info['source']}")
                print(f"    Input Shape: {info['input_shape']}")
                print(f"    Description: {info['description']}")
                print()

        # 打印已禁用的模型
        if disabled_models:
            print(f"\nDISABLED MODELS ({len(disabled_models)}):")
            print(f"{'-'*80}")
            for info in disabled_models:
                print(f"  • {info['name']}")
                print(f"    Source: {info['source']}")
                print(f"    Description: {info['description']}")
                print()

        print(f"{'='*80}\n")

    def analyze_models(self, specific_models: Optional[List[str]] = None):
        """批量分析模型

        Args:
            specific_models: 要分析的特定模型名称列表（如果为None，则分析所有enabled的模型）
        """
        models_config = self.config.get('models', [])

        # 筛选要分析的模型
        models_to_analyze = []
        for model_config in models_config:
            name = model_config.get('name')
            enabled = model_config.get('enabled', False)
            optional = model_config.get('optional', False)

            # 如果指定了特定模型，只分析这些模型
            if specific_models:
                if name in specific_models:
                    models_to_analyze.append(model_config)
            # 否则分析所有enabled的模型
            elif enabled:
                models_to_analyze.append(model_config)

        if not models_to_analyze:
            print("No models to analyze!")
            return

        print(f"\n{'='*80}")
        print(f"Analyzing {len(models_to_analyze)} models...")
        print(f"{'='*80}\n")

        # 获取全局设置
        output_folder = self._get_global_setting('output_folder')
        plot_folder = self._get_global_setting('plot_folder')
        print_detailed = self._get_global_setting('print_detailed', True)
        generate_plots = self._get_global_setting('generate_plots', True)

        # 确保输出目录存在
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        if generate_plots and plot_folder:
            os.makedirs(plot_folder, exist_ok=True)

        # 覆盖FHE参数（如果配置中有指定）
        fhe_params = self.config.get('fhe_params', {})

        # 逐个分析模型
        for i, model_config in enumerate(models_to_analyze):
            name = model_config.get('name', f'Model_{i}')
            optional = model_config.get('optional', False)

            print(f"\n{'='*80}")
            print(f"[{i+1}/{len(models_to_analyze)}] Analyzing: {name}")
            print(f"{'='*80}\n")

            try:
                # 加载模型
                model_name, model, input_shape = self._load_model(model_config)

                # 创建FheInfo对象
                fhe_info = FheInfo(model, input_shape, model_name)

                # 覆盖FHE参数（如果有）
                if fhe_params:
                    for param_name, param_value in fhe_params.items():
                        if hasattr(fhe_info, param_name):
                            setattr(fhe_info, param_name, param_value)

                # 运行统计
                fhe_info.run_statistics()

                # 打印统计结果
                fhe_info.print_statistics(output_folder)

                # 打印详细统计
                if print_detailed:
                    fhe_info.print_detailed_statistics(output_folder)

                # 生成图表
                if generate_plots and plot_folder:
                    fhe_info.plot_statistics(plot_folder=plot_folder, show=False)

                # 保存结果
                self.results[model_name] = fhe_info

                print(f"\n✓ Successfully analyzed {name}\n")

            except Exception as e:
                if optional:
                    print(f"\n⚠ Skipping optional model {name}: {e}\n")
                else:
                    print(f"\n✗ Error analyzing {name}: {e}\n")
                    import traceback
                    traceback.print_exc()

        print(f"\n{'='*80}")
        print(f"Analysis complete! Analyzed {len(self.results)} models successfully.")
        print(f"{'='*80}\n")

    def generate_comparison_plots(self):
        """生成综合比较图"""
        if not self.results:
            print("No analysis results to compare!")
            return

        generate_comparison = self._get_global_setting('generate_comparison', True)
        if not generate_comparison:
            print("Comparison plots disabled in config.")
            return

        plot_folder = self._get_global_setting('plot_folder')
        if not plot_folder:
            print("Plot folder not specified in config.")
            return

        comparison_config = self.config.get('comparison', {})
        plot_types = comparison_config.get('plot_types', ['network_comparison'])

        print(f"\n{'='*80}")
        print(f"Generating comparison plots...")
        print(f"{'='*80}\n")

        # 准备网络比较数据
        network_data = {}
        for name, fhe_info in self.results.items():
            network_data[name] = fhe_info.get_network_comparison_data()

        # 生成各种比较图
        for plot_type in plot_types:
            print(f"Generating {plot_type}...")

            if plot_type == 'network_comparison':
                FheInfo.plot_network_comparison(network_data, plot_folder=plot_folder, show=False)

            elif plot_type == 'comprehensive_comparison':
                FheInfo.plot_network_comprehensive_comparison(self.results, plot_folder=plot_folder, show=False)

            elif plot_type == 'grouped_comparison':
                FheInfo.plot_network_grouped_comparison(self.results, plot_folder=plot_folder, show=False)

            else:
                print(f"  Unknown plot type: {plot_type}")

        print(f"\n✓ Comparison plots generated successfully!\n")

    def generate_summary_report(self):
        """生成汇总报告"""
        if not self.results:
            print("No analysis results to summarize!")
            return

        output_folder = self._get_global_setting('output_folder')
        if not output_folder:
            print("Output folder not specified.")
            return

        print(f"\n{'='*80}")
        print(f"Generating summary report...")
        print(f"{'='*80}\n")

        # 生成汇总报告
        lines = []
        lines.append("="*120)
        lines.append("FHE STATISTICS SUMMARY REPORT")
        lines.append("="*120)
        lines.append("")

        # 表头
        header = f"{'Model Name':<30} {'FHE Latency':>15} {'Boot Latency':>15} {'Total':>15} {'Max Depth':>12} {'Params(M)':>12} {'FLOPs(M)':>12}"
        lines.append(header)
        lines.append("-"*120)

        # 各模型汇总
        for name, fhe_info in sorted(self.results.items()):
            fhe_latency = fhe_info.total_latency
            boot_latency = fhe_info.total_boot_latency
            total_latency = fhe_latency + boot_latency
            max_depth = fhe_info.get_max_depth()
            params = fhe_info.get_parameter_count() / 1e6
            flops = fhe_info.get_flops_count() / 1e6

            line = f"{name:<30} {fhe_latency:>15.2f} {boot_latency:>15.2f} {total_latency:>15.2f} {max_depth:>12} {params:>12.2f} {flops:>12.2f}"
            lines.append(line)

        lines.append("="*120)

        # 保存报告
        from fhe_statistics.statistics_fn import generate_unique_filename
        report_path = generate_unique_filename("summary_report", "txt", output_folder)

        report_text = "\n".join(lines)
        print(report_text)

        with open(report_path, 'w') as f:
            f.write(report_text)

        print(f"\n✓ Summary report saved to: {report_path}\n")

    def run(self, specific_models: Optional[List[str]] = None):
        """运行完整的批量分析流程

        Args:
            specific_models: 要分析的特定模型名称列表
        """
        # 分析模型
        self.analyze_models(specific_models)

        # 生成比较图
        if len(self.results) > 1:
            self.generate_comparison_plots()

        # 生成汇总报告
        self.generate_summary_report()

        print(f"\n{'='*80}")
        print(f"ALL DONE! Results saved to:")
        print(f"  • Statistics: {self._get_global_setting('output_folder')}")
        print(f"  • Plots: {self._get_global_setting('plot_folder')}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Batch FHE Statistics Analyzer')
    parser.add_argument('--config', type=str,
                       default='fhe_statistics/batch_analysis_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--list', action='store_true',
                       help='List all models in configuration file')
    parser.add_argument('--models', nargs='+', type=str,
                       help='Specific models to analyze (by name)')

    args = parser.parse_args()

    # 创建分析器
    analyzer = BatchAnalyzer(args.config)

    # 如果是列出模型模式
    if args.list:
        analyzer.list_models()
        return

    # 运行批量分析
    analyzer.run(specific_models=args.models)


if __name__ == '__main__':
    main()
