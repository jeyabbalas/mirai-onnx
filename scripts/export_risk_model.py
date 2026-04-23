"""Phase 3: Export the Mirai risk model (transformer + RF pool + hazard head) to ONNX.

Produces models/risk_model.onnx with the following I/O:

    Inputs:
        img_feats       (B, 4, 512) fp32   per-view image features (from image_encoder.onnx)
        view_seq        (B, 4)     int64   per-slot view code  (CC=0, MLO=1, PAD=2)
        side_seq        (B, 4)     int64   per-slot side code  (R=0,  L=1,   PAD=2)
        time_seq        (B, 4)     int64   per-slot time index (0 for single exam)
        rf_vector       (B, 100)   fp32    user-supplied RF vector (zeros to invoke model-predicted)
        rf_known_mask   (B, 100)   fp32    elementwise 1 where rf_vector is real, 0 where unknown

    Outputs:
        logit             (B, 5)    fp32   pre-sigmoid 5-year cumulative-hazard logits (== raw_logit.npy)
        hidden_pre_hazard (B, 612)  fp32   the XAI embedding (post-ReLU; == xai_hidden.npy)

The wrapper reimplements just the parts of AllImageTransformer.forward / RiskFactorPool.forward
that need to change for export:
  * skip mask_input (identity in eval mode by inspection of hiddens_transfomer.py:55-66)
  * call each per-key {key}_fc EXACTLY ONCE (no get_pred_rf_loss try/except second call)
  * replace the Python branches in RiskFactorPool with a tensor-level blend on rf_known_mask
  * use out-of-place F.relu instead of the upstream nn.ReLU(inplace=True)

Contract source: docs/architecture.md §9.2, §11 (R1-R8); mirai-migration-plan.md §4.
Run inside the mirai-export conda env (torch >= 2.1 required for the ONNX exporter).

Usage:
    conda activate mirai-export
    python scripts/export_risk_model.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MIRAI_CONFIG = PROJECT_ROOT / "external" / "Mirai" / "onconet" / "configs" / "mirai_trained.json"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "reference" / "fixtures"
OUT_DIR = PROJECT_ROOT / "models"
OUT_PATH = OUT_DIR / "risk_model.onnx"

EXPECTED_RF_DIM = 100
EXPECTED_IMG_REPR_DIM = 512
EXPECTED_NUM_VIEWS = 4
EXPECTED_MAX_FOLLOWUP = 5

# Two-track tolerance scheme (docs/architecture.md §1, PHASE_2_REPORT.md). Note that
# Phase 3 cannot inherit Phase 2's `ATOL_TORCH=0.0` bound: the risk-model graph is
# attention + LayerNorm + Linear + LinearLU rather than Conv + BN + ReLU, and the ATen
# kernel-reduction order for these ops drifted between torch 1.9.0 (the snapshot/fixture
# capture env, mirai-py38) and torch 2.2.2 (the exporter env, mirai-export). The
# resulting torch-vs-torch divergence is at the fp32 ULP floor (observed ~2.4e-7,
# matching the value-magnitude × 2^-22 for hidden activations near 0.7-1.0). We pin at
# 1e-6 to match Phase 0's ATOL_FP32 floor, with headroom over the observed maximum.
# See PHASE_3_REPORT.md for measurements and rationale.
ATOL_TORCH = 1e-6
ATOL_ORT = 2e-5
OPSET = 17


class RiskModelExport(nn.Module):
    """Risk-model wrapper for ONNX export.

    Holds references to the loaded MiraiFull submodules; reimplements the forward
    path with the export-time changes documented in docs/architecture.md §11.
    All weights remain shared with the input MiraiFull instance.
    """

    def __init__(self, mirai_full):
        super().__init__()
        transformer = mirai_full.transformer
        pool = transformer.pool

        self.projection_layer = transformer.projection_layer
        self.transformer = transformer.transformer  # the inner Transformer (attention stack)
        self.internal_pool = pool.internal_pool
        self.prob_of_failure_layer = transformer.prob_of_failure_layer

        # Per-key FC modules. Register them on self so torch.onnx.export can find
        # the parameters and so .eval() / state_dict() behave correctly.
        self.risk_factor_keys: List[str] = list(transformer.args.risk_factor_keys)
        self.num_class: dict = dict(transformer.args.risk_factor_key_to_num_class)
        for key in self.risk_factor_keys:
            fc = getattr(pool, f"{key}_fc")
            self.add_module(f"{key}_fc", fc)

        self.rf_dim = int(pool.length_risk_factor_vector)
        assert self.rf_dim == EXPECTED_RF_DIM, (
            f"length_risk_factor_vector={self.rf_dim} but expected {EXPECTED_RF_DIM}; "
            "snapshot may have drifted"
        )
        self.image_repr_dim = int(mirai_full.image_repr_dim)
        assert self.image_repr_dim == EXPECTED_IMG_REPR_DIM, (
            f"image_repr_dim={self.image_repr_dim} but expected {EXPECTED_IMG_REPR_DIM}"
        )
        # transformer.args.hidden_dim has been mutated to img_only_dim + rf_dim by the
        # RiskFactorPool constructor; the projection_layer maps the precomputed image
        # hidden to args.transfomer_hidden_dim (equal to image_repr_dim for this snapshot).

    def forward(
        self,
        img_feats: torch.Tensor,        # (B, 4, 512) fp32
        view_seq: torch.Tensor,         # (B, 4) int64
        side_seq: torch.Tensor,         # (B, 4) int64
        time_seq: torch.Tensor,         # (B, 4) int64
        rf_vector: torch.Tensor,        # (B, 100) fp32
        rf_known_mask: torch.Tensor,    # (B, 100) fp32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # mask_input is identity in eval mode (mask_prob=0 -> mask_embedding(ones) is
        # zero by padding_idx=1, so masked_x == x). Skip the bernoulli node entirely.
        projected = self.projection_layer(img_feats)
        transformer_hidden = self.transformer(projected, time_seq, view_seq, side_seq)

        img_like_hidden = transformer_hidden.transpose(1, 2).unsqueeze(-1)
        _, image_hidden = self.internal_pool(img_like_hidden)

        # Each per-key FC is invoked exactly once (matches Phase 0's first-call capture
        # via the `if k not in captured` guard in tests/reference/_hooks.py).
        per_key_probs = []
        for key in self.risk_factor_keys:
            fc = getattr(self, f"{key}_fc")
            key_logit = fc(image_hidden)
            if self.num_class[key] == 1:
                key_probs = torch.sigmoid(key_logit)
            else:
                key_probs = F.softmax(key_logit, dim=-1)
            per_key_probs.append(key_probs)
        rf_predicted = torch.cat(per_key_probs, dim=1)

        rf_used = rf_known_mask * rf_vector + (1.0 - rf_known_mask) * rf_predicted
        pool_hidden = torch.cat([image_hidden, rf_used], dim=1)
        # Out-of-place ReLU; upstream uses inplace=True which the Phase 0 hooks defend
        # against with .copy(). Out-of-place is cleaner for the trace.
        hidden_pre_hazard = F.relu(pool_hidden)
        logit = self.prob_of_failure_layer(hidden_pre_hazard)
        return logit, hidden_pre_hazard


def _load_mirai_full():
    """Steps 1-5 of architecture.md §13; identical to scripts/export_image_encoder.py."""
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


def _slot_seqs_from_batch_order(suffix: str = "") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (view_seq, side_seq, time_seq) at B=1 from batch_order{suffix}.json."""
    order = json.loads((FIXTURES_DIR / f"batch_order{suffix}.json").read_text())
    assert len(order) == EXPECTED_NUM_VIEWS, f"expected {EXPECTED_NUM_VIEWS} slots, got {len(order)}"
    views = [int(e["view"]) for e in order]
    sides = [int(e["side"]) for e in order]
    view_seq = torch.tensor([views], dtype=torch.int64)
    side_seq = torch.tensor([sides], dtype=torch.int64)
    time_seq = torch.zeros((1, EXPECTED_NUM_VIEWS), dtype=torch.int64)
    return view_seq, side_seq, time_seq


def _load_dummy_inputs():
    """Build the demo (pydicom) input tuple for export-tracing and parity check."""
    img_feats_np = np.load(FIXTURES_DIR / "image_encoder_out.npy").astype(np.float32, copy=False)
    assert img_feats_np.shape == (1, 4, 512), img_feats_np.shape
    img_feats = torch.from_numpy(img_feats_np)
    view_seq, side_seq, time_seq = _slot_seqs_from_batch_order()
    rf_vector = torch.zeros((1, EXPECTED_RF_DIM), dtype=torch.float32)
    rf_known_mask = torch.zeros((1, EXPECTED_RF_DIM), dtype=torch.float32)
    return img_feats, view_seq, side_seq, time_seq, rf_vector, rf_known_mask


def _gold_outputs() -> Tuple[np.ndarray, np.ndarray]:
    logit = np.load(FIXTURES_DIR / "raw_logit.npy")
    hidden = np.load(FIXTURES_DIR / "xai_hidden.npy")
    assert logit.shape == (1, EXPECTED_MAX_FOLLOWUP), logit.shape
    assert hidden.shape == (1, EXPECTED_IMG_REPR_DIM + EXPECTED_RF_DIM), hidden.shape
    return logit, hidden


def main() -> int:
    torch.manual_seed(0)
    torch.set_num_threads(1)

    print(f"[export] loading MiraiFull from {MIRAI_CONFIG}")
    model, config = _load_mirai_full()
    rf_dim = int(model.transformer.pool.length_risk_factor_vector)
    print(f"[export] rf_dim={rf_dim}, image_repr_dim={model.image_repr_dim}")

    wrapper = RiskModelExport(model)
    wrapper.eval()

    img_feats, view_seq, side_seq, time_seq, rf_vector, rf_known_mask = _load_dummy_inputs()
    print(f"[export] dummy shapes: img_feats={tuple(img_feats.shape)} "
          f"view_seq={tuple(view_seq.shape)} rf_vector={tuple(rf_vector.shape)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gold_logit, gold_hidden = _gold_outputs()

    with torch.no_grad():
        torch_logit, torch_hidden = wrapper(
            img_feats, view_seq, side_seq, time_seq, rf_vector, rf_known_mask
        )
    torch_logit_np = torch_logit.numpy()
    torch_hidden_np = torch_hidden.numpy()
    torch_logit_diff = float(np.abs(torch_logit_np - gold_logit).max())
    torch_hidden_diff = float(np.abs(torch_hidden_np - gold_hidden).max())
    print(f"[export] pytorch wrapper vs fixture: "
          f"logit max abs diff={torch_logit_diff:.3e}, "
          f"hidden max abs diff={torch_hidden_diff:.3e}")
    np.testing.assert_allclose(torch_logit_np, gold_logit, atol=ATOL_TORCH, rtol=0.0,
                                err_msg="pytorch wrapper logit diverges from raw_logit.npy")
    np.testing.assert_allclose(torch_hidden_np, gold_hidden, atol=ATOL_TORCH, rtol=0.0,
                                err_msg="pytorch wrapper hidden diverges from xai_hidden.npy")
    print(f"[export] pytorch wrapper parity OK at atol={ATOL_TORCH}")

    print(f"[export] torch.onnx.export -> {OUT_PATH} (opset {OPSET})")
    t0 = time.time()
    dynamic_axes = {
        "img_feats": {0: "B"},
        "view_seq": {0: "B"},
        "side_seq": {0: "B"},
        "time_seq": {0: "B"},
        "rf_vector": {0: "B"},
        "rf_known_mask": {0: "B"},
        "logit": {0: "B"},
        "hidden_pre_hazard": {0: "B"},
    }
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (img_feats, view_seq, side_seq, time_seq, rf_vector, rf_known_mask),
            str(OUT_PATH),
            input_names=["img_feats", "view_seq", "side_seq", "time_seq", "rf_vector", "rf_known_mask"],
            output_names=["logit", "hidden_pre_hazard"],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            do_constant_folding=True,
        )
    print(f"[export] export took {time.time() - t0:.1f}s")

    print(f"[export] file size: {OUT_PATH.stat().st_size / 1024 / 1024:.2f} MB")

    proto = onnx.load(str(OUT_PATH))
    onnx.checker.check_model(proto)
    print("[export] onnx.checker.check_model: OK")

    sess = ort.InferenceSession(str(OUT_PATH), providers=["CPUExecutionProvider"])
    ort_logit, ort_hidden = sess.run(
        None,
        {
            "img_feats": img_feats.numpy(),
            "view_seq": view_seq.numpy(),
            "side_seq": side_seq.numpy(),
            "time_seq": time_seq.numpy(),
            "rf_vector": rf_vector.numpy(),
            "rf_known_mask": rf_known_mask.numpy(),
        },
    )
    ort_logit_diff = float(np.abs(ort_logit - gold_logit).max())
    ort_hidden_diff = float(np.abs(ort_hidden - gold_hidden).max())
    print(f"[export] onnxruntime vs fixture: "
          f"logit max abs diff={ort_logit_diff:.3e}, "
          f"hidden max abs diff={ort_hidden_diff:.3e}")
    np.testing.assert_allclose(ort_logit, gold_logit, atol=ATOL_ORT, rtol=0.0,
                                err_msg="onnxruntime logit diverges from raw_logit.npy")
    np.testing.assert_allclose(ort_hidden, gold_hidden, atol=ATOL_ORT, rtol=0.0,
                                err_msg="onnxruntime hidden diverges from xai_hidden.npy")
    print(f"[export] onnxruntime parity OK at atol={ATOL_ORT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
