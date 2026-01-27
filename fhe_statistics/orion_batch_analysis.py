#!/usr/bin/env python
"""
Batch Orion latency analysis for common networks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision

# add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fhe_statistics.orion_statistics_fn import analyze_model_orion
from models import resnet20


def build_models(
    resnet20_num_classes: int,
    imagenet_num_classes: int,
    pretrained: bool,
) -> List[Tuple[str, torch.nn.Module, tuple]]:
    models: List[Tuple[str, torch.nn.Module, tuple]] = []

    models.append((
        "ResNet20",
        resnet20(num_classes=resnet20_num_classes, pretrained=False),
        (1, 3, 32, 32),
    ))

    models.append((
        "ResNet18",
        torchvision.models.resnet18(pretrained=pretrained),
        (1, 3, 224, 224),
    ))
    models.append((
        "ResNet34",
        torchvision.models.resnet34(pretrained=pretrained),
        (1, 3, 224, 224),
    ))
    models.append((
        "ResNet50",
        torchvision.models.resnet50(pretrained=pretrained),
        (1, 3, 224, 224),
    ))
    models.append((
        "EfficientNetB0",
        torchvision.models.efficientnet_b0(pretrained=pretrained),
        (1, 3, 224, 224),
    ))
    models.append((
        "MobileNetV2",
        torchvision.models.mobilenet_v2(pretrained=pretrained),
        (1, 3, 224, 224),
    ))

    # adjust classifier heads for class counts
    for name, model, _ in models:
        if name == "EfficientNetB0":
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_features, imagenet_num_classes)
        elif name == "MobileNetV2":
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_features, imagenet_num_classes)
        elif name.startswith("ResNet") and name != "ResNet20":
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, imagenet_num_classes)

    return models


def main() -> int:
    parser = argparse.ArgumentParser(description="Orion latency batch analysis")
    parser.add_argument("--output-folder", type=str, default=None)
    parser.add_argument("--plot-folder", type=str, default=None)
    parser.add_argument("--no-boot-opt", action="store_true")
    parser.add_argument("--l-eff", type=int, default=None)
    parser.add_argument("--embedding-method", type=str, default="hybrid")
    parser.add_argument("--bsgs-ratio", type=float, default=2.0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resnet20-classes", type=int, default=100)
    parser.add_argument("--imagenet-classes", type=int, default=1000)
    args = parser.parse_args()

    output_folder = args.output_folder
    plot_folder = args.plot_folder

    if output_folder:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
    if plot_folder:
        Path(plot_folder).mkdir(parents=True, exist_ok=True)

    models = build_models(
        resnet20_num_classes=args.resnet20_classes,
        imagenet_num_classes=args.imagenet_classes,
        pretrained=args.pretrained,
    )

    print("\nModel".ljust(16), "Latency(M)".rjust(12), "Boots".rjust(8), "MaxDepth".rjust(10))
    print("-" * 50)

    for name, model, input_shape in models:
        info = analyze_model_orion(
            model=model,
            model_name=name,
            output_folder=output_folder,
            plot_folder=plot_folder,
            input_shape=input_shape,
            print_detailed=False,
            optimize_boot=not args.no_boot_opt,
            l_eff=args.l_eff,
            embedding_method=args.embedding_method,
            bsgs_ratio=args.bsgs_ratio,
        )
        total = (info.total_latency + info.total_boot_latency) / 1e6
        print(f"{name.ljust(16)} {total:>12.2f} {info.total_boot_count:>8} {info.get_max_depth():>10}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
