"""Phase 5 verification: image_encoder.onnx + risk_model.onnx + calibrator.json compose
end-to-end to reproduce the Phase 0 demo predictions.

Run inside the mirai-export env (torch>=2.1, onnx>=1.15, onnxruntime>=1.17). The export
scripts and scripts/run_onnx_pipeline.py must have been run at least once (not for the
tests themselves -- the tests call into run_onnx_pipeline.run_pipeline directly -- but
for the .onnx files to exist on disk).

Contract source: mirai-migration-plan.md §6; docs/architecture.md §1.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort
import pytest

from .conftest import (
    ATOL_ORT,
    FIXTURES_DIR,
    IMAGE_ENCODER_ONNX,
    RISK_MODEL_ONNX,
    RTOL,
)

# Pull run_pipeline + its constants from the Phase 5 script (scripts/ is not a package).
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from calibrator_from_json import load_calibrator  # noqa: E402
from run_onnx_pipeline import (  # noqa: E402
    CALIBRATOR_JSON,
    MODEL_VERSION,
    PINNED_PREDICTIONS,
    run_pipeline,
)

# Plan §6 tolerances (calibrator amplifies via base_slope ~5x).
ATOL_CALIBRATED = 1e-4
RTOL_CALIBRATED = 1e-3


if not IMAGE_ENCODER_ONNX.exists() or not RISK_MODEL_ONNX.exists():
    pytest.skip(
        f"{IMAGE_ENCODER_ONNX} or {RISK_MODEL_ONNX} not found; run Phase 2/3 export scripts first.",
        allow_module_level=True,
    )
if not CALIBRATOR_JSON.exists():
    pytest.skip(
        f"{CALIBRATOR_JSON} not found; run `python scripts/export_calibrator.py` first.",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def encoder_session() -> ort.InferenceSession:
    return ort.InferenceSession(
        str(IMAGE_ENCODER_ONNX), providers=["CPUExecutionProvider"]
    )


@pytest.fixture(scope="module")
def risk_session() -> ort.InferenceSession:
    return ort.InferenceSession(
        str(RISK_MODEL_ONNX), providers=["CPUExecutionProvider"]
    )


@pytest.fixture(scope="module")
def calibrator():
    return load_calibrator(CALIBRATOR_JSON)


@pytest.fixture(scope="module")
def pipeline_results(encoder_session, risk_session, calibrator):
    """Run the composed pipeline once per decode path and cache for all parity tests."""
    return {
        decode: run_pipeline(decode, encoder_session, risk_session, calibrator)
        for decode in ("pydicom", "dcmtk")
    }


def _suffix(decode: str) -> str:
    return "_dcmtk" if decode == "dcmtk" else ""


# ---------------------------------------------------------------------------
# Numerical parity -- composed ORT output vs Phase 0 fixtures.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("decode", ["pydicom", "dcmtk"])
def test_logit_parity(pipeline_results, decode: str) -> None:
    got = pipeline_results[decode]["logit"]
    gold = np.load(FIXTURES_DIR / f"raw_logit{_suffix(decode)}.npy")
    assert got.shape == (1, 5) and got.dtype == np.float32, (got.shape, got.dtype)
    max_abs = float(np.abs(got - gold).max())
    np.testing.assert_allclose(
        got, gold, atol=ATOL_ORT, rtol=RTOL,
        err_msg=(
            f"[{decode}] end-to-end logit diverges from raw_logit{_suffix(decode)}.npy "
            f"(max abs diff {max_abs:.3e}, tolerance {ATOL_ORT:.0e})"
        ),
    )


@pytest.mark.parametrize("decode", ["pydicom", "dcmtk"])
def test_embedding_parity(pipeline_results, decode: str) -> None:
    """XAI embedding == xai_hidden{,_dcmtk}.npy (post-ReLU, per Phase 1 decision).

    Tolerance pre-bumped from the plan's 1e-5 to ATOL_ORT=2e-5 because the image encoder's
    ORT drift (~1e-5) propagates through the risk model. Empirical max abs diff is recorded
    in PHASE_5_REPORT.md; if it exceeds 2e-5 on either decode path, stop and flag rather
    than loosening silently (CLAUDE.md "do not silently loosen" rule).
    """
    got = pipeline_results[decode]["hidden"]
    gold = np.load(FIXTURES_DIR / f"xai_hidden{_suffix(decode)}.npy")
    assert got.shape == (1, 612) and got.dtype == np.float32, (got.shape, got.dtype)
    max_abs = float(np.abs(got - gold).max())
    np.testing.assert_allclose(
        got, gold, atol=ATOL_ORT, rtol=RTOL,
        err_msg=(
            f"[{decode}] end-to-end hidden_pre_hazard diverges from xai_hidden{_suffix(decode)}.npy "
            f"(max abs diff {max_abs:.3e}, tolerance {ATOL_ORT:.0e})"
        ),
    )


@pytest.mark.parametrize("decode", ["pydicom", "dcmtk"])
def test_calibrated_parity(pipeline_results, decode: str) -> None:
    """Calibrated fp64 probabilities within plan §6 bound (atol=1e-4, rtol=1e-3)."""
    got = pipeline_results[decode]["calibrated"]
    gold = np.load(FIXTURES_DIR / f"calibrated{_suffix(decode)}.npy")
    assert got.shape == (5,) and got.dtype == np.float64, (got.shape, got.dtype)
    max_abs = float(np.abs(got - gold).max())
    np.testing.assert_allclose(
        got, gold, atol=ATOL_CALIBRATED, rtol=RTOL_CALIBRATED,
        err_msg=(
            f"[{decode}] calibrated probabilities diverge from calibrated{_suffix(decode)}.npy "
            f"(max abs diff {max_abs:.3e}, atol={ATOL_CALIBRATED:.0e}, rtol={RTOL_CALIBRATED:.0e})"
        ),
    )


# ---------------------------------------------------------------------------
# User-visible gate: 4-dp rounded predictions must match pinned baseline exactly.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("decode", ["pydicom", "dcmtk"])
def test_rounded_predictions_bit_equal_pinned(pipeline_results, decode: str) -> None:
    """round(calibrated, 4) per year must be bit-equal to mirai-migration-plan.md §1."""
    got = pipeline_results[decode]["predictions"]
    expected = PINNED_PREDICTIONS[decode]
    # Iterate in the same canonical order predictions.json uses.
    for year in ("Year 1", "Year 2", "Year 3", "Year 4", "Year 5"):
        assert got[year] == expected[year], (
            f"[{decode}] {year}: got {got[year]}, pinned {expected[year]} "
            f"(4-dp rounding drifted across ORT boundary)"
        )


@pytest.mark.parametrize("decode", ["pydicom", "dcmtk"])
def test_predictions_json_matches_phase0(pipeline_results, decode: str) -> None:
    """Our predictions dict + modelVersion bit-match Phase 0's predictions{,_dcmtk}.json."""
    phase0 = json.loads((FIXTURES_DIR / f"predictions{_suffix(decode)}.json").read_text())
    got_predictions = pipeline_results[decode]["predictions"]
    assert got_predictions == phase0["predictions"], (got_predictions, phase0["predictions"])
    assert MODEL_VERSION == phase0.get("modelVersion"), (MODEL_VERSION, phase0.get("modelVersion"))
