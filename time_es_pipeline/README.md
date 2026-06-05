# time_es_pipeline

`time_es_pipeline` is a small runner for exporting project models to ONNX and, once
the local tools are available, passing generated intermediates through
`onnx-mlir` and `hecate-opt`.

The pipeline is intentionally local to this directory:

- tool binaries or build trees go under `./build/`
- ONNX files and intermediate artifacts go under `./temp/`
- both directories are ignored by git
- `compiler_config.json` is copied locally from the upstream pipeline and used as
  the default CKKS config
- `configapollo1.json` is kept locally as a legacy/statistics-enabled config for
  comparison

Run commands from this directory with the parent project environment:

```bash
uv run python main.py list-models
uv run python main.py export-onnx --model resnet20
```

The default export uses:

- model: `resnet20`
- input shape: `1,3,224,224`
- classes: `100`
- pretrained weights: disabled
- ONNX opset: `18`

Examples:

```bash
# Export a model to ./temp/onnx/resnet20.onnx
uv run python main.py export-onnx --model resnet20

# Export another registered model
uv run python main.py export-onnx --model efficientnet-b0

# Check where the pipeline will look for compiler tools
uv run python main.py tool-status

# Emit lowered MLIR.
uv run python main.py compile-onnx --onnx temp/onnx/resnet20.onnx

# Emit ONNX dialect MLIR for Hecate.
uv run python main.py compile-onnx --onnx temp/onnx/resnet20.onnx --emit EmitONNXIR

# Run Hecate's trytry pass on ONNX dialect MLIR.
uv run python main.py optimize-mlir --input temp/mlir/resnet20.tmp --pipeline trytry

# Compare ONNX-MLIR pooling lowering modes against Hecate --onnx.
uv run python main.py compare-pooling

# Estimate latency and parse Hecate output into a summary JSON.
uv run python main.py estimate-latency --model resnet18 --output-stem resnet18_latency

# Run export -> onnx-mlir -> hecate-opt in one command using Hecate's trytry pass.
uv run python main.py run --model resnet20 --with-onnx-mlir --with-hecate-opt --require-tools

# Run the upstream-style Hecate --onnx path. This uses .onnx.mlir, adds
# --enable-conv-opt-pass=false for onnx-mlir, and adds Hecate's --waterline=46
# --allow-unregistered-dialect unless explicit args are provided.
uv run python main.py run \
  --model resnet20 \
  --with-onnx-mlir \
  --with-hecate-opt \
  --hecate-pipeline onnx \
  --require-tools
```

To compare against the old Hecate binary:

```bash
uv run python main.py estimate-latency \
  --model resnet18 \
  --output-stem resnet18_latency_old \
  --hecate-tool "/mnt/zhitai/KnowledgeLib/Research/EfficientNet/experiment data/time-es-pipline 12.13/hecate-opt" \
  --ld-library-path-prepend /nix/store/l0x1xn0fddp1k0g6swbs3cyfvr57ixsg-openmp-16.0.6/lib
```

Extra model constructor arguments can be passed as `KEY=VALUE` pairs:

```bash
uv run python main.py export-onnx \
  --model nas-json \
  --model-arg json_path=../best_models/Model\ A.json
```
