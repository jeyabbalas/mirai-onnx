"""Phase 2: Export the Mirai image encoder to ONNX.

Produces models/image_encoder.onnx: input (N, 3, 2048, 1664) fp32 -> output (N, 512) fp32.

Contract source: docs/architecture.md §9.1, §11 (R6-R8), §12, §13.
Run inside the mirai-export conda env (torch >= 2.1 required for the ONNX exporter).

Usage:
    conda activate mirai-export
    python scripts/export_image_encoder.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MIRAI_CONFIG = PROJECT_ROOT / "external" / "Mirai" / "onconet" / "configs" / "mirai_trained.json"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "reference" / "fixtures"
OUT_DIR = PROJECT_ROOT / "models"
OUT_PATH = OUT_DIR / "image_encoder.onnx"

EXPECTED_RF_DIM = 100
# Torch-internal parity is bit-exact (the wrapper is a view + slice over the same graph
# that produced Phase 0's image_encoder_out.npy). ORT is a separate kernel implementation
# (MLAS) and produces ULP-level differences from torch's ATen; empirically ~1e-5 on
# this graph, independent of graph-optimization level and constant-folding. See
# PHASE_2_REPORT.md for measurements.
ATOL_TORCH = 0.0
ATOL_ORT = 2e-5
OPSET = 17


class ImageEncoderExport(nn.Module):
    """Per-image encoder wrapper for ONNX export.

    Accepts (N, 3, 2048, 1664) instead of MiraiFull's (B, C, N, H, W) layout;
    the caller reshapes upstream. Applies the [:, :img_repr_dim] slice from
    MiraiFull.forward line 59.
    """

    def __init__(self, mirai_full):
        super().__init__()
        self.image_encoder = mirai_full.image_encoder
        self.image_repr_dim = int(mirai_full.image_repr_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden, _ = self.image_encoder(x, None, None)
        return hidden[:, : self.image_repr_dim]


def _load_mirai_full():
    """Steps 1-5 of architecture.md §13."""
    from onconet.predict import _load_config
    from onconet.models.mirai_full import MiraiFull

    config = _load_config(str(MIRAI_CONFIG), threads=1)
    config.cuda = False
    config.use_precomputed_hiddens = False

    model = MiraiFull(config)
    if isinstance(model, nn.DataParallel):
        model = model.module
    model = model.cpu()
    model.eval()
    return model, config


def _load_tracer_stack() -> np.ndarray:
    """Stack the four pydicom preproc tensors in batch_order.json slot order."""
    order = json.loads((FIXTURES_DIR / "batch_order.json").read_text())
    per_view: List[np.ndarray] = []
    for entry in order:
        name = f"{entry['view_str']}_{entry['side_str']}.npy"
        per_view.append(np.load(FIXTURES_DIR / "preproc_tensor" / name))
    stack = np.stack(per_view).astype(np.float32, copy=False)
    assert stack.shape == (4, 3, 2048, 1664), f"unexpected stack shape {stack.shape}"
    return stack


def main() -> int:
    torch.manual_seed(0)
    torch.set_num_threads(1)

    print(f"[export] loading MiraiFull from {MIRAI_CONFIG}")
    model, config = _load_mirai_full()

    rf_dim = int(model.transformer.pool.length_risk_factor_vector)
    assert rf_dim == EXPECTED_RF_DIM, (
        f"rf_dim={rf_dim} but expected {EXPECTED_RF_DIM}; snapshot may have drifted"
    )
    print(f"[export] rf_dim={rf_dim}, image_repr_dim={model.image_repr_dim}")

    wrapper = ImageEncoderExport(model)
    wrapper.eval()

    stack = _load_tracer_stack()
    dummy = torch.from_numpy(stack)  # (4, 3, 2048, 1664) fp32
    print(f"[export] tracer input shape={tuple(dummy.shape)}, dtype={dummy.dtype}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference pytorch forward before export, for comparison.
    with torch.no_grad():
        torch_out = wrapper(dummy).numpy()
    gold = np.load(FIXTURES_DIR / "image_encoder_out.npy").reshape(4, -1)
    torch_max_diff = float(np.abs(torch_out - gold).max())
    print(f"[export] pytorch wrapper vs fixture max abs diff: {torch_max_diff:.3e}")
    np.testing.assert_allclose(torch_out, gold, atol=ATOL_TORCH, rtol=0.0)
    print(f"[export] pytorch wrapper parity: exact (atol={ATOL_TORCH})")

    print(f"[export] torch.onnx.export -> {OUT_PATH} (opset {OPSET})")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(OUT_PATH),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
            opset_version=OPSET,
            do_constant_folding=True,
        )
    print(f"[export] export took {time.time() - t0:.1f}s")

    print(f"[export] file size: {OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")

    proto = onnx.load(str(OUT_PATH))
    onnx.checker.check_model(proto)
    print("[export] onnx.checker.check_model: OK")

    sess = ort.InferenceSession(str(OUT_PATH), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": stack})[0]
    ort_max_diff = float(np.abs(ort_out - gold).max())
    print(f"[export] onnxruntime vs fixture max abs diff: {ort_max_diff:.3e}")
    np.testing.assert_allclose(ort_out, gold, atol=ATOL_ORT, rtol=0.0)
    print(f"[export] onnxruntime parity OK at atol={ATOL_ORT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
