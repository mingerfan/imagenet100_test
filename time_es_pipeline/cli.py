from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from latency_estimator import LatencyEstimateResult, estimate_latency
from model_registry import ModelRequest, list_registered_models, parse_key_value_items
from onnx_export import OnnxExportConfig, export_onnx, parse_shape, result_to_json
from paths import DEFAULT_CKKS_CONFIG, DEFAULT_HECATE_DIR, DEFAULT_MLIR_DIR, DEFAULT_ONNX_DIR
from pooling_compare import DEFAULT_COMPARE_MODELS, compare_pooling_modes
from tools import (
    compile_with_onnx_mlir,
    default_hecate_args_for_onnx,
    default_onnx_mlir_args_for_hecate,
    find_onnx_mlir_output,
    find_tool,
    optimize_with_hecate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="time_es_pipeline",
        description="Export project models to ONNX and run local ONNX-MLIR/Hecate tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-models", help="List registered models from ../models.")
    subparsers.add_parser("tool-status", help="Show where onnx-mlir and hecate-opt are found.")

    export_parser = subparsers.add_parser("export-onnx", help="Export a registered model to ONNX.")
    add_export_args(export_parser)

    compile_parser = subparsers.add_parser("compile-onnx", help="Run onnx-mlir on an ONNX file.")
    compile_parser.add_argument("--onnx", type=Path, required=True, help="Input ONNX file.")
    compile_parser.add_argument("--output-dir", type=Path, default=DEFAULT_MLIR_DIR)
    compile_parser.add_argument("--emit", default="EmitONNXIR", help="onnx-mlir emit option without leading --.")
    compile_parser.add_argument("--tool", default=None, help="Explicit onnx-mlir path or directory.")
    compile_parser.add_argument("--arg", action="append", default=[], help="Extra argument for onnx-mlir.")

    hecate_parser = subparsers.add_parser("optimize-mlir", help="Run hecate-opt on an MLIR file.")
    hecate_parser.add_argument("--input", type=Path, required=True, help="Input MLIR file.")
    hecate_parser.add_argument("--output", type=Path, default=None, help="Output MLIR file.")
    hecate_parser.add_argument("--ckks-config", type=Path, default=DEFAULT_CKKS_CONFIG)
    hecate_parser.add_argument("--pipeline", default="trytry", help="Hecate pass or pipeline name.")
    hecate_parser.add_argument("--tool", default=None, help="Explicit hecate-opt path or directory.")
    hecate_parser.add_argument("--arg", action="append", default=[], help="Extra argument for hecate-opt.")

    latency_parser = subparsers.add_parser(
        "estimate-latency",
        help="Export a model, run ONNX-MLIR and Hecate --onnx, and parse latency.",
    )
    add_export_args(latency_parser)
    latency_parser.add_argument("--onnx-mlir-tool", default=None)
    latency_parser.add_argument(
        "--onnx-mlir-arg",
        action="append",
        default=None,
        help="Extra onnx-mlir arg. Defaults to --enable-conv-opt-pass=false.",
    )
    latency_parser.add_argument("--hecate-tool", default=None)
    latency_parser.add_argument("--ckks-config", type=Path, default=DEFAULT_CKKS_CONFIG)
    latency_parser.add_argument("--hecate-pipeline", default="onnx")
    latency_parser.add_argument(
        "--hecate-arg",
        action="append",
        default=None,
        help="Extra hecate-opt arg. Defaults to --waterline=46 and --allow-unregistered-dialect.",
    )
    latency_parser.add_argument(
        "--prefer-tmp-mlir",
        action="store_true",
        help="Prefer onnx-mlir .tmp output instead of .onnx.mlir.",
    )
    latency_parser.add_argument("--timeout", type=int, default=None, help="Seconds before killing hecate-opt.")
    latency_parser.add_argument("--log-dir", type=Path, default=None)
    latency_parser.add_argument(
        "--ld-library-path-prepend",
        action="append",
        default=[],
        help="Directory to prepend to LD_LIBRARY_PATH for hecate-opt.",
    )

    compare_parser = subparsers.add_parser(
        "compare-pooling",
        help="Compare ONNX-MLIR pooling lowering modes and Hecate --onnx behavior.",
    )
    compare_parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=f"Model to test. Defaults to: {', '.join(DEFAULT_COMPARE_MODELS)}.",
    )
    compare_parser.add_argument("--num-classes", type=int, default=100)
    compare_parser.add_argument("--pretrained", action="store_true")
    compare_parser.add_argument("--input-shape", default="1,3,224,224")
    compare_parser.add_argument("--timeout", type=int, default=90, help="Seconds per Hecate invocation.")
    compare_parser.add_argument("--ckks-config", type=Path, default=DEFAULT_CKKS_CONFIG)
    compare_parser.add_argument(
        "--skip-trytry-control",
        action="store_true",
        help="Skip the separate Hecate trytry control run.",
    )

    run_parser = subparsers.add_parser("run", help="Export ONNX and optionally run downstream tools.")
    add_export_args(run_parser)
    run_parser.add_argument("--with-onnx-mlir", action="store_true", help="Run onnx-mlir after export.")
    run_parser.add_argument("--with-hecate-opt", action="store_true", help="Run hecate-opt after onnx-mlir.")
    run_parser.add_argument("--onnx-mlir-emit", default="EmitONNXIR")
    run_parser.add_argument("--onnx-mlir-tool", default=None)
    run_parser.add_argument("--onnx-mlir-arg", action="append", default=[])
    run_parser.add_argument("--hecate-tool", default=None)
    run_parser.add_argument("--ckks-config", type=Path, default=DEFAULT_CKKS_CONFIG)
    run_parser.add_argument("--hecate-pipeline", default="trytry")
    run_parser.add_argument("--hecate-arg", action="append", default=[])
    run_parser.add_argument(
        "--prefer-tmp-mlir",
        action="store_true",
        help="Prefer onnx-mlir .tmp output. By default Hecate --onnx uses .onnx.mlir.",
    )
    run_parser.add_argument(
        "--require-tools",
        action="store_true",
        help="Fail if requested tools are missing. Otherwise missing tools are skipped.",
    )
    return parser


def add_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="resnet20", help="Registered model name.")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--pretrained", action="store_true", help="Pass pretrained=True to model constructor.")
    parser.add_argument(
        "--model-arg",
        action="append",
        default=[],
        help="Extra model constructor argument as KEY=VALUE.",
    )
    parser.add_argument("--input-shape", default="1,3,224,224", help="Comma-separated input tensor shape.")
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--output-name", default="output")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--output-stem", default=None, help="Override output file stem.")
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--no-verify", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "list-models":
            for model_name in list_registered_models():
                print(model_name)
            return 0

        if args.command == "tool-status":
            for name in ("onnx-mlir", "hecate-opt"):
                lookup = find_tool(name)
                if lookup.path:
                    print(f"{name}: {lookup.path} ({lookup.source})")
                else:
                    print(f"{name}: not found")
            return 0

        if args.command == "export-onnx":
            result = export_onnx(export_config_from_args(args))
            print(result_to_json(result))
            return 0

        if args.command == "compile-onnx":
            result = compile_with_onnx_mlir(
                onnx_path=args.onnx,
                output_dir=args.output_dir,
                emit=args.emit,
                tool_path=args.tool,
                extra_args=args.arg,
            )
            print_command_result(result)
            return 0 if result.ok else result.returncode

        if args.command == "optimize-mlir":
            result = optimize_with_hecate(
                input_path=args.input,
                output_path=args.output,
                ckks_config=args.ckks_config,
                pipeline=args.pipeline,
                tool_path=args.tool,
                extra_args=args.arg,
            )
            print_command_result(result)
            return 0 if result.ok else result.returncode

        if args.command == "estimate-latency":
            result = estimate_latency(
                export_config=export_config_from_args(args),
                onnx_mlir_tool=args.onnx_mlir_tool,
                onnx_mlir_args=args.onnx_mlir_arg,
                hecate_tool=args.hecate_tool,
                ckks_config=args.ckks_config,
                hecate_pipeline=args.hecate_pipeline,
                hecate_args=args.hecate_arg,
                prefer_tmp_mlir=args.prefer_tmp_mlir,
                timeout_seconds=args.timeout,
                log_dir=args.log_dir,
                ld_library_path_prepend=args.ld_library_path_prepend,
            )
            print_latency_estimate(result)
            return 0 if result.ok else 1

        if args.command == "compare-pooling":
            summary_path, rows = compare_pooling_modes(
                models=args.model or None,
                input_shape=parse_shape(args.input_shape),
                num_classes=args.num_classes,
                pretrained=args.pretrained,
                timeout_seconds=args.timeout,
                ckks_config=args.ckks_config,
                run_trytry_control=not args.skip_trytry_control,
            )
            print_pooling_compare(summary_path, rows)
            return 0

        if args.command == "run":
            return run_pipeline(args)

    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unhandled command: {args.command}")
    return 2


def export_config_from_args(args: argparse.Namespace) -> OnnxExportConfig:
    return OnnxExportConfig(
        model=ModelRequest(
            name=args.model,
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            extra_args=parse_key_value_items(args.model_arg),
        ),
        input_shape=parse_shape(args.input_shape),
        input_name=args.input_name,
        output_name=args.output_name,
        opset=args.opset,
        output_dir=args.output_dir,
        output_name_override=args.output_stem,
        dynamic_batch=args.dynamic_batch,
        verify=not args.no_verify,
    )


def run_pipeline(args: argparse.Namespace) -> int:
    export_result = export_onnx(export_config_from_args(args))
    print(result_to_json(export_result))

    if not args.with_onnx_mlir:
        return 0

    onnx_mlir_args = list(args.onnx_mlir_arg)
    if args.hecate_pipeline == "onnx" and not onnx_mlir_args:
        onnx_mlir_args = default_onnx_mlir_args_for_hecate()

    try:
        mlir_result = compile_with_onnx_mlir(
            onnx_path=export_result.onnx_path,
            output_dir=DEFAULT_MLIR_DIR,
            emit=args.onnx_mlir_emit,
            tool_path=args.onnx_mlir_tool,
            extra_args=onnx_mlir_args,
        )
    except FileNotFoundError as exc:
        if args.require_tools:
            raise
        print(f"skip onnx-mlir: {exc}", file=sys.stderr)
        return 0

    print_command_result(mlir_result)
    if not mlir_result.ok:
        return mlir_result.returncode

    if not args.with_hecate_opt:
        return 0

    mlir_input = find_onnx_mlir_output(
        export_result.onnx_path,
        DEFAULT_MLIR_DIR,
        prefer_tmp=args.prefer_tmp_mlir,
        prefer_onnx_mlir=args.hecate_pipeline == "onnx" and not args.prefer_tmp_mlir,
    )
    if mlir_input is None:
        message = f"expected MLIR output not found for: {export_result.onnx_path}"
        if args.require_tools:
            raise FileNotFoundError(message)
        print(f"skip hecate-opt: {message}", file=sys.stderr)
        return 0

    try:
        hecate_args = list(args.hecate_arg)
        if args.hecate_pipeline == "onnx" and not hecate_args:
            hecate_args = default_hecate_args_for_onnx()

        hecate_result = optimize_with_hecate(
            input_path=mlir_input,
            ckks_config=args.ckks_config,
            pipeline=args.hecate_pipeline,
            tool_path=args.hecate_tool,
            extra_args=hecate_args,
        )
    except FileNotFoundError as exc:
        if args.require_tools:
            raise
        print(f"skip hecate-opt: {exc}", file=sys.stderr)
        return 0

    print_command_result(hecate_result)
    return 0 if hecate_result.ok else hecate_result.returncode


def print_command_result(result) -> None:
    print(f"$ {result.command_string()}")
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)


def print_latency_estimate(result: LatencyEstimateResult) -> None:
    payload = {
        "status": result.status,
        "model": result.model,
        "latency": result.latency,
        "latency_line": result.latency_line,
        "elapsed_seconds": result.elapsed_seconds,
        "summary_path": result.summary_path,
        "onnx_path": result.onnx_path,
        "mlir_path": result.mlir_path,
        "hecate_output_path": result.hecate_output_path,
        "hecate_log": result.hecate_log,
        "error": result.error,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def print_pooling_compare(summary_path, rows) -> None:
    print(f"summary: {summary_path}")
    header = (
        f"{'model':<10} {'variant':<13} {'input':<10} {'onnx pools':<28} "
        f"{'mlir pools':<34} {'--onnx':<14} {'trytry':<14}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        onnx_pools = ",".join(row.onnx_pool_ops) or "-"
        mlir_pools = ",".join(row.mlir_pool_ops) or "-"
        hecate_input = Path(row.hecate_input).suffix if row.hecate_input else "-"
        if row.hecate_input and row.hecate_input.endswith(".onnx.mlir"):
            hecate_input = ".onnx.mlir"
        print(
            f"{row.model:<10} {row.variant:<13} {hecate_input:<10} {onnx_pools:<28} "
            f"{mlir_pools:<34} {row.hecate_onnx_status:<14} {row.hecate_trytry_status:<14}"
        )
        if row.hecate_onnx_excerpt:
            print(f"  --onnx: {row.hecate_onnx_excerpt}")
