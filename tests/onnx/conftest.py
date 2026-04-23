"""Path constants and tolerances for Phase 2 ONNX export tests.

Intentionally separate from tests/reference/conftest.py because that module is torch 1.9 /
mirai-py38-scoped. Phase 2 runs under mirai-export (torch 2.2); we import numpy/onnx/ort
but NOT torch here.
"""

from __future__ import annotations

import pathlib

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "reference" / "fixtures"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGE_ENCODER_ONNX = MODELS_DIR / "image_encoder.onnx"
RISK_MODEL_ONNX = MODELS_DIR / "risk_model.onnx"

# PyTorch wrapper reproduces the captured fixture bit-for-bit (same torch graph).
ATOL_TORCH = 0.0

# ORT uses MLAS kernels; empirically ULP-level differences vs the torch-captured
# fixture (~9.5e-6 pydicom, ~1.05e-5 dcmtk — a single outlier element). Set the
# tolerance at 2e-5 to cover both paths with headroom. Independent of graph
# optimization level and constant folding — an intrinsic property of the
# cross-framework bridge. See PHASE_2_REPORT.md for measurements.
ATOL_ORT = 2e-5
RTOL = 0.0

MAX_FILE_SIZE_MB = 200
