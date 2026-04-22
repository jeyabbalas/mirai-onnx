"""Phase 0 baseline: re-run the upstream Mirai pipeline and assert every
captured tensor matches the fixtures committed under tests/reference/fixtures/.

Each pipeline run takes ~25s on Rosetta-translated CPU, so we cache via
session-scoped fixtures and parametrize tests over both DICOM-decoding paths.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest

from tests.reference import conftest as cfg
from tests.reference.capture_reference import (
    run_pipeline_with_hooks,
    set_determinism,
)

FIXTURES = cfg.FIXTURES_DIR

# (view_int, view_str, side_int, side_str) for the four images.
VIEW_CODES = [
    (0, "CC", 1, "L"),
    (0, "CC", 0, "R"),
    (1, "MLO", 1, "L"),
    (1, "MLO", 0, "R"),
]


def _suffix(run: str) -> str:
    return "_dcmtk" if run == "dcmtk" else ""


# ---------------------------------------------------------------------------
# Session-scoped fixtures (the expensive ones)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _determinism():
    set_determinism()


@pytest.fixture(scope="session")
def captured_pydicom():
    return run_pipeline_with_hooks(use_dcmtk=False)


@pytest.fixture(scope="session")
def captured_dcmtk():
    return run_pipeline_with_hooks(use_dcmtk=True)


@pytest.fixture(scope="session")
def manifest():
    path = FIXTURES / "MANIFEST.json"
    assert path.exists(), f"Run capture_reference.py first; {path} missing."
    return json.loads(path.read_text())


# Per-test fixture that dispatches to the right run.
@pytest.fixture(params=["pydicom", "dcmtk"])
def run_pair(request, captured_pydicom, captured_dcmtk):
    return request.param, {"pydicom": captured_pydicom, "dcmtk": captured_dcmtk}[
        request.param
    ]


# ---------------------------------------------------------------------------
# Smoke / housekeeping
# ---------------------------------------------------------------------------
def test_no_nans(run_pair):
    _, captured = run_pair
    for name in (
        "image_encoder_out",
        "pool_hidden",
        "image_hidden_in_pool",
        "risk_factor_vector",
        "xai_hidden",
        "raw_logit",
        "raw_sigmoid",
        "calibrated",
    ):
        arr = np.asarray(captured[name])
        assert np.isfinite(arr).all(), f"{name} has NaN/Inf"


def test_batch_order_matches_manifest(run_pair, manifest):
    run, captured = run_pair
    expected = [(s["view"], s["side"]) for s in manifest["runs"][run]["batch_order"]]
    assert captured["_batch_slot_to_view_side"] == expected


def test_rf_dim_unchanged(run_pair, manifest):
    run, captured = run_pair
    assert captured["rf_dim"] == manifest["runs"][run]["shapes"]["rf_dim"]


def test_predictions_match_upstream_pydicom(captured_pydicom):
    # Pin the upstream demo numbers explicitly so a regression in the model load
    # path or the pipeline shows up here, not just as fixture-drift.
    expected = {"Year 1": 0.0314, "Year 2": 0.0505, "Year 3": 0.0711, "Year 4": 0.0935, "Year 5": 0.1052}
    assert captured_pydicom["predictions"] == expected


def test_predictions_match_upstream_dcmtk(captured_dcmtk):
    expected = {"Year 1": 0.0298, "Year 2": 0.0483, "Year 3": 0.0684, "Year 4": 0.09, "Year 5": 0.1016}
    assert captured_dcmtk["predictions"] == expected


# ---------------------------------------------------------------------------
# Per-image fixtures
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("view,view_str,side,side_str", VIEW_CODES)
def test_dicom_raw_uint16(run_pair, view, view_str, side, side_str):
    run, captured = run_pair
    suffix = _suffix(run)
    fixture = np.load(FIXTURES / f"dicom_raw_uint16{suffix}" / f"{view_str}_{side_str}.npy")
    actual = captured["dicom_raw_uint16"][(view, side)]
    assert actual.dtype == np.uint16
    assert actual.shape == fixture.shape
    # uint16 is bit-exact: same windowing + same dicom = identical pixels.
    np.testing.assert_array_equal(actual, fixture)


@pytest.mark.parametrize("view,view_str,side,side_str", VIEW_CODES)
def test_preproc_tensor(run_pair, view, view_str, side, side_str):
    run, captured = run_pair
    suffix = _suffix(run)
    fixture = np.load(FIXTURES / f"preproc_tensor{suffix}" / f"{view_str}_{side_str}.npy")
    actual = captured["preproc_tensor"][(view, side)]
    assert actual.dtype == np.float32
    assert actual.shape == fixture.shape
    np.testing.assert_allclose(actual, fixture, atol=cfg.ATOL_FP32, rtol=cfg.RTOL_FP32)


# ---------------------------------------------------------------------------
# Single-tensor fixtures
# ---------------------------------------------------------------------------
SINGLE_TENSORS = [
    ("image_encoder_out", np.float32, cfg.ATOL_FP32),
    ("image_hidden_in_pool", np.float32, cfg.ATOL_FP32),
    ("pool_hidden", np.float32, cfg.ATOL_FP32),
    ("risk_factor_vector", np.float32, cfg.ATOL_FP32),
    ("xai_hidden", np.float32, cfg.ATOL_FP32),
    ("raw_logit", np.float32, cfg.ATOL_FP32),
    ("raw_sigmoid", np.float32, cfg.ATOL_FP32),
    ("calibrated", np.float64, cfg.ATOL_FP64),
]


@pytest.mark.parametrize("name,dtype,atol", SINGLE_TENSORS)
def test_single_tensor(run_pair, name, dtype, atol):
    run, captured = run_pair
    suffix = _suffix(run)
    fixture = np.load(FIXTURES / f"{name}{suffix}.npy")
    actual = np.asarray(captured[name])
    assert actual.dtype == dtype, f"{name}: dtype {actual.dtype} != {dtype}"
    assert actual.shape == fixture.shape
    np.testing.assert_allclose(actual, fixture, atol=atol, rtol=0)


# ---------------------------------------------------------------------------
# Per-key predicted risk factors
# ---------------------------------------------------------------------------
def test_pred_risk_factors_per_key(run_pair):
    run, captured = run_pair
    suffix = _suffix(run)
    pred_dir = FIXTURES / f"pred_risk_factors_per_key{suffix}"
    keys = captured["risk_factor_keys"]
    for key in keys:
        fixture = np.load(pred_dir / f"{key}.npy")
        actual = captured["pred_risk_factors_per_key"][key]
        np.testing.assert_allclose(
            actual, fixture, atol=cfg.ATOL_FP32, rtol=0,
            err_msg=f"per-key probs differ for {key}",
        )


# ---------------------------------------------------------------------------
# JSON fixtures
# ---------------------------------------------------------------------------
def test_predictions_json_round_trip(run_pair):
    run, captured = run_pair
    suffix = _suffix(run)
    fixture = json.loads((FIXTURES / f"predictions{suffix}.json").read_text())
    actual = json.loads(json.dumps({"predictions": captured["predictions"], "modelVersion": captured["modelVersion"]}))
    assert actual == fixture


def test_batch_order_json(run_pair):
    run, captured = run_pair
    suffix = _suffix(run)
    fixture = json.loads((FIXTURES / f"batch_order{suffix}.json").read_text())
    expected = [(int(s["view"]), int(s["side"])) for s in fixture]
    assert captured["_batch_slot_to_view_side"] == expected


# ---------------------------------------------------------------------------
# Cross-checks (catch capture bugs even if fixtures match)
# ---------------------------------------------------------------------------
def test_xai_tail_equals_relu_risk_factor_vector(run_pair):
    _, captured = run_pair
    rf_dim = captured["rf_dim"]
    np.testing.assert_allclose(
        captured["xai_hidden"][:, -rf_dim:],
        np.maximum(captured["risk_factor_vector"], 0.0),
        atol=cfg.ATOL_FP32, rtol=0,
    )


def test_xai_equals_relu_pool_hidden(run_pair):
    _, captured = run_pair
    np.testing.assert_allclose(
        captured["xai_hidden"],
        np.maximum(captured["pool_hidden"], 0.0),
        atol=cfg.ATOL_FP32, rtol=0,
    )


def test_calibrated_round_to_predictions(run_pair):
    _, captured = run_pair
    for i, p in enumerate(captured["calibrated"]):
        assert round(float(p), 4) == captured["predictions"][f"Year {i + 1}"]


def test_per_key_concat_equals_risk_factor_vector(run_pair):
    _, captured = run_pair
    keys = captured["risk_factor_keys"]
    cat = np.concatenate(
        [captured["pred_risk_factors_per_key"][k] for k in keys], axis=1
    )
    np.testing.assert_allclose(
        cat, captured["risk_factor_vector"], atol=cfg.ATOL_FP32, rtol=0,
    )


# ---------------------------------------------------------------------------
# Manifest integrity (catches stale fixtures)
# ---------------------------------------------------------------------------
def test_manifest_file_hashes(manifest):
    for run_name, data in manifest["runs"].items():
        for relpath, info in data["fixture_files"].items():
            path = FIXTURES / relpath
            assert path.exists(), f"{relpath} missing"
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == info["sha256"], f"{relpath}: hash drift"


def test_manifest_snapshot_hashes(manifest):
    import os
    for name, info in manifest["snapshots"].items():
        if name == "remote_snapshot_uri":
            continue
        path = pathlib.Path(os.path.expanduser(info["path"]))
        assert path.exists(), f"{name} snapshot missing on disk"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == info["sha256"], f"{name}: snapshot hash drift"
