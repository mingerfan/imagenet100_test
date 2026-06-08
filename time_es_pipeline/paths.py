from __future__ import annotations

from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent
DEFAULT_TEMP_DIR = PIPELINE_ROOT / "temp"
DEFAULT_BUILD_DIR = PIPELINE_ROOT / "build"
DEFAULT_ONNX_DIR = DEFAULT_TEMP_DIR / "onnx"
DEFAULT_MLIR_DIR = DEFAULT_TEMP_DIR / "mlir"
DEFAULT_HECATE_DIR = DEFAULT_TEMP_DIR / "hecate"
DEFAULT_CKKS_CONFIG = PIPELINE_ROOT / "compiler_config.json"
LEGACY_CKKS_CONFIG = PIPELINE_ROOT / "configapollo1.json"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
