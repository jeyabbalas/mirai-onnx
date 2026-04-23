"""Phase 5: Compose image_encoder.onnx + risk_model.onnx + calibrator.json end-to-end.

Reads Phase 0 preproc tensors (bit-exact outputs of upstream Mirai's DICOM preprocessor,
pinned by SHA in tests/reference/fixtures/MANIFEST.json) and runs them through the two
ONNX sessions + the JSON calibrator to reproduce the demo Mirai prediction.

Writes into artifacts/phase_5/ (gitignored, reproducible):
    onnx_prediction{,_dcmtk}.json   -- {"predictions": {"Year 1": ..., ...}, "modelVersion": "0.14.1"}
    onnx_embedding{,_dcmtk}.npy     -- (1, 612) fp32 hidden_pre_hazard (XAI embedding)

Exit non-zero if any 4-decimal-rounded prediction disagrees with the pinned baseline.

Contract source: mirai-migration-plan.md §6; docs/architecture.md §1, §9, §10.
Run inside the mirai-export conda env (onnxruntime >= 1.17).

Usage:
    conda activate mirai-export
    python scripts/run_onnx_pipeline.py                   # both decode paths
    python scripts/run_onnx_pipeline.py --decode pydicom  # single path
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MODELS_DIR = PROJECT_ROOT / "models"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "reference" / "fixtures"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "phase_5"

IMAGE_ENCODER_ONNX = MODELS_DIR / "image_encoder.onnx"
RISK_MODEL_ONNX = MODELS_DIR / "risk_model.onnx"
CALIBRATOR_JSON = MODELS_DIR / "calibrator.json"

# Baseline from mirai-migration-plan.md §1 (4-decimal rounded); every Phase 5 run
# reproduces these bit-for-bit or exits non-zero.
PINNED_PREDICTIONS = {
    "pydicom": {"Year 1": 0.0314, "Year 2": 0.0505, "Year 3": 0.0711, "Year 4": 0.0935, "Year 5": 0.1052},
    "dcmtk":   {"Year 1": 0.0298, "Year 2": 0.0483, "Year 3": 0.0684, "Year 4": 0.09,   "Year 5": 0.1016},
}

# Phase 0's predictions.json carries this tag; keep it in sync.
MODEL_VERSION = "0.14.1"

sys.path.insert(0, str(HERE))
from calibrator_from_json import load_calibrator, apply_calibrator_json  # noqa: E402


def _load_preproc_stack(decode: str) -> Tuple[np.ndarray, List[dict]]:
    """Return (stack, batch_order) for the given decode path.

    stack:       (4, 3, 2048, 1664) fp32 -- Phase 0 preproc tensors, slot-ordered per batch_order.json.
    batch_order: list of {slot, view, view_str, side, side_str}.
    """
    suffix = "_dcmtk" if decode == "dcmtk" else ""
    order_path = FIXTURES_DIR / f"batch_order{suffix}.json"
    stack_dir = FIXTURES_DIR / f"preproc_tensor{suffix}"
    if not order_path.exists():
        raise FileNotFoundError(f"missing {order_path}; run Phase 0 first")
    if not stack_dir.is_dir():
        raise FileNotFoundError(f"missing {stack_dir}; run Phase 0 first")

    order = json.loads(order_path.read_text())
    tensors = []
    for entry in order:
        p = stack_dir / f"{entry['view_str']}_{entry['side_str']}.npy"
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run Phase 0 first")
        tensors.append(np.load(p))
    stack = np.stack(tensors).astype(np.float32, copy=False)
    if stack.shape != (4, 3, 2048, 1664):
        raise ValueError(f"unexpected preproc stack shape {stack.shape}, want (4, 3, 2048, 1664)")
    return stack, order


def _build_risk_inputs(img_feats_flat: np.ndarray, order: List[dict]) -> Dict[str, np.ndarray]:
    """Assemble the 6-input dict for risk_model.onnx from encoder output + slot order."""
    if img_feats_flat.shape != (4, 512):
        raise ValueError(f"encoder output must be (4, 512), got {img_feats_flat.shape}")
    img_feats = img_feats_flat[None, ...].astype(np.float32, copy=False)  # (1, 4, 512)
    view_seq = np.array([[int(e["view"]) for e in order]], dtype=np.int64)
    side_seq = np.array([[int(e["side"]) for e in order]], dtype=np.int64)
    time_seq = np.zeros((1, len(order)), dtype=np.int64)
    # zeros in both rf inputs -> invoke model-predicted-RF path (matches the demo).
    rf_vector = np.zeros((1, 100), dtype=np.float32)
    rf_known_mask = np.zeros((1, 100), dtype=np.float32)
    return {
        "img_feats": img_feats,
        "view_seq": view_seq,
        "side_seq": side_seq,
        "time_seq": time_seq,
        "rf_vector": rf_vector,
        "rf_known_mask": rf_known_mask,
    }


def run_pipeline(
    decode: str,
    encoder_session: ort.InferenceSession,
    risk_session: ort.InferenceSession,
    calibrator,
) -> Dict[str, object]:
    """Run preproc -> encoder ORT -> risk ORT -> sigmoid -> calibrator -> round(4).

    Returns a dict with all intermediate tensors (useful for the test harness).
    """
    stack, order = _load_preproc_stack(decode)
    img_feats_flat = encoder_session.run(None, {"input": stack})[0]          # (4, 512) fp32

    risk_inputs = _build_risk_inputs(img_feats_flat, order)
    logit, hidden = risk_session.run(None, risk_inputs)                      # (1, 5), (1, 612)
    if logit.shape != (1, 5):
        raise ValueError(f"logit shape {logit.shape}, want (1, 5)")
    if hidden.shape != (1, 612):
        raise ValueError(f"hidden shape {hidden.shape}, want (1, 612)")

    sigmoid = 1.0 / (1.0 + np.exp(-logit.astype(np.float32)))                # (1, 5) fp32
    calibrated = apply_calibrator_json(calibrator, sigmoid)                  # (5,) fp64

    predictions = {f"Year {i+1}": round(float(calibrated[i]), 4) for i in range(5)}
    return {
        "decode": decode,
        "order": order,
        "img_feats_flat": img_feats_flat,
        "logit": logit,
        "sigmoid": sigmoid,
        "calibrated": calibrated,
        "hidden": hidden,
        "predictions": predictions,
    }


def _write_outputs(result: Dict[str, object], output_dir: pathlib.Path) -> Dict[str, pathlib.Path]:
    """Write onnx_prediction*.json and onnx_embedding*.npy for this decode path."""
    suffix = "_dcmtk" if result["decode"] == "dcmtk" else ""
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_dir / f"onnx_prediction{suffix}.json"
    emb_path = output_dir / f"onnx_embedding{suffix}.npy"

    pred_obj = {"predictions": result["predictions"], "modelVersion": MODEL_VERSION}
    pred_path.write_text(json.dumps(pred_obj, indent=2) + "\n")
    np.save(emb_path, result["hidden"])

    return {"prediction": pred_path, "embedding": emb_path}


def _check_pinned(result: Dict[str, object]) -> List[str]:
    """Return a list of mismatch strings (empty means all years matched)."""
    decode = result["decode"]
    expected = PINNED_PREDICTIONS[decode]
    got = result["predictions"]
    mismatches: List[str] = []
    for k in ("Year 1", "Year 2", "Year 3", "Year 4", "Year 5"):
        if got[k] != expected[k]:
            mismatches.append(f"  {decode}/{k}: got {got[k]}, expected {expected[k]}")
    return mismatches


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--decode", choices=["pydicom", "dcmtk", "both"], default="both",
        help="which DICOM decode path(s) to run (selects batch_order + preproc_tensor fixtures)",
    )
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR,
        help=f"where to write onnx_prediction*.json and onnx_embedding*.npy (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args(argv)

    for p in (IMAGE_ENCODER_ONNX, RISK_MODEL_ONNX, CALIBRATOR_JSON):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    decodes = ["pydicom", "dcmtk"] if args.decode == "both" else [args.decode]

    encoder_session = ort.InferenceSession(
        str(IMAGE_ENCODER_ONNX), providers=["CPUExecutionProvider"]
    )
    risk_session = ort.InferenceSession(
        str(RISK_MODEL_ONNX), providers=["CPUExecutionProvider"]
    )
    calibrator = load_calibrator(CALIBRATOR_JSON)

    any_mismatch = False
    for decode in decodes:
        result = run_pipeline(decode, encoder_session, risk_session, calibrator)
        written = _write_outputs(result, args.output_dir)

        print(f"[{decode}] predictions: {result['predictions']}")
        print(f"[{decode}] wrote {written['prediction']}")
        print(f"[{decode}] wrote {written['embedding']}  (shape {tuple(result['hidden'].shape)})")

        mismatches = _check_pinned(result)
        if mismatches:
            any_mismatch = True
            print(f"[{decode}] MISMATCH vs pinned baseline (mirai-migration-plan.md §1):",
                  file=sys.stderr)
            for m in mismatches:
                print(m, file=sys.stderr)
        else:
            print(f"[{decode}] OK: rounded predictions bit-equal to pinned baseline")

    return 1 if any_mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
