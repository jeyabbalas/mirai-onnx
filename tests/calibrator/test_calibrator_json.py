"""Phase 4 verification: calibrator.json + calibrator_from_json helper reproduce
the pickled MiraiCalibrator exactly on both random inputs and Phase 0 fixtures.

The export script `scripts/export_calibrator.py` must have been run first.

Runs cleanly under mirai-py38 or mirai-export; the per-year random-parity tests
additionally require the source pickle at ~/.mirai/snapshots/calibrators/... to
be present (they skip cleanly if not).
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import pickle
import sys

import numpy as np
import pytest

from .conftest import (
    ATOL_PARITY,
    CALIBRATOR_JSON,
    EXPECTED_N_YEARS,
    EXPECTED_PREDICTIONS_DCMTK,
    EXPECTED_PREDICTIONS_PYDICOM,
    EXPECTED_SHA256,
    FIXTURES_DIR,
    MIRAI_CONFIG,
    MIRAI_ROOT,
    PROJECT_ROOT,
    RTOL,
    SCHEMA_VERSION,
    SCRIPTS_DIR,
)

# Make scripts/calibrator_from_json.py importable without installing it.
sys.path.insert(0, str(SCRIPTS_DIR))
from calibrator_from_json import (  # noqa: E402
    apply_calibrator_json,
    load_calibrator,
    predict_proba_year,
)


# ---------------------------------------------------------------------------
# Module-level skip: require the JSON file to exist.
# ---------------------------------------------------------------------------
if not CALIBRATOR_JSON.exists():
    pytest.skip(
        f"{CALIBRATOR_JSON} not found; run `python scripts/export_calibrator.py` first.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Module-scoped fixtures.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def calibrator_json() -> dict:
    """Raw parsed JSON payload."""
    with CALIBRATOR_JSON.open("r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def calibrator_loaded() -> dict:
    """Parsed via the helper (validates schema + returns `{year: params}` dict)."""
    return load_calibrator(CALIBRATOR_JSON)


def _pickle_path() -> pathlib.Path:
    """Resolve the pickle path from mirai_trained.json."""
    cfg = json.loads(MIRAI_CONFIG.read_text())
    return pathlib.Path(cfg["calibrator_path"]).expanduser().resolve()


def _load_reference_pickle():
    """Load the source pickle; skip the caller if it's missing."""
    pkl_path = _pickle_path()
    if not pkl_path.exists():
        pytest.skip(f"reference pickle not found at {pkl_path}")
    sys.path.insert(0, str(MIRAI_ROOT))
    from onconet.models.calibrator import MiraiCalibrator  # noqa: F401
    with pkl_path.open("rb") as f:
        return pickle.load(f), pkl_path


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Structural tests.
# ---------------------------------------------------------------------------
def test_schema_version(calibrator_json: dict) -> None:
    assert calibrator_json.get("schema_version") == SCHEMA_VERSION


def test_source_sha_matches_pickle(calibrator_json: dict) -> None:
    """source_pickle_sha256 in the JSON matches the JSON's claimed value, and
    if the pickle is present, matches the pickle too."""
    sha_in_json = calibrator_json.get("source_pickle_sha256")
    assert sha_in_json == EXPECTED_SHA256, sha_in_json

    pkl_path = _pickle_path()
    if pkl_path.exists():
        assert _sha256(pkl_path) == EXPECTED_SHA256, (
            f"source pickle SHA drifted: {_sha256(pkl_path)}"
        )


def test_years_structure(calibrator_json: dict) -> None:
    years = calibrator_json["years"]
    assert isinstance(years, list)
    assert len(years) == EXPECTED_N_YEARS

    required = {"index", "base_slope", "base_offset", "calibrator_slope", "calibrator_offset"}
    indices_seen = set()
    for entry in years:
        assert isinstance(entry, dict)
        assert required.issubset(entry.keys()), entry.keys()
        idx = entry["index"]
        assert isinstance(idx, int)
        assert 0 <= idx < EXPECTED_N_YEARS
        assert idx not in indices_seen, f"duplicate index {idx}"
        indices_seen.add(idx)
        for k in ("base_slope", "base_offset", "calibrator_slope", "calibrator_offset"):
            v = entry[k]
            assert isinstance(v, (int, float)), f"{k}: {type(v).__name__}"
            assert math.isfinite(v), f"{k}: {v!r}"
    assert indices_seen == set(range(EXPECTED_N_YEARS))


def test_years_sorted_by_index(calibrator_json: dict) -> None:
    """JSON writes the list in index order, so TS consumers can iterate by position."""
    indices = [e["index"] for e in calibrator_json["years"]]
    assert indices == list(range(EXPECTED_N_YEARS))


def test_load_calibrator_returns_dict(calibrator_loaded: dict) -> None:
    assert set(calibrator_loaded.keys()) == set(range(EXPECTED_N_YEARS))
    for params in calibrator_loaded.values():
        assert set(params.keys()) == {
            "base_slope", "base_offset", "calibrator_slope", "calibrator_offset"
        }


# ---------------------------------------------------------------------------
# Per-year random-input parity (vs source pickle).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("year", list(range(EXPECTED_N_YEARS)))
def test_per_year_random_parity(calibrator_loaded: dict, year: int) -> None:
    """100 RNG-seeded probabilities: JSON path == pickled predict_proba to atol=1e-9."""
    ref, _ = _load_reference_pickle()
    rng = np.random.default_rng(seed=2026_04_23 + year)
    # Probabilities in (1e-6, 1-1e-6); extreme values can saturate the exp.
    probs = rng.uniform(1e-6, 1.0 - 1e-6, size=100).reshape(-1, 1)

    expected = ref[year].predict_proba(probs, expand=False)
    actual = predict_proba_year(calibrator_loaded[year], probs, expand=False)

    max_abs = float(np.abs(actual - expected).max())
    np.testing.assert_allclose(
        actual, expected, atol=ATOL_PARITY, rtol=RTOL,
        err_msg=f"year {year}: max abs diff {max_abs:.3e} (tol {ATOL_PARITY:.0e})",
    )


@pytest.mark.parametrize("year", list(range(EXPECTED_N_YEARS)))
def test_per_year_random_parity_expand(calibrator_loaded: dict, year: int) -> None:
    """Same as above but with expand=True — proves the (2, n) shape matches."""
    ref, _ = _load_reference_pickle()
    rng = np.random.default_rng(seed=0xC417 + year)
    probs = rng.uniform(1e-6, 1.0 - 1e-6, size=50).reshape(-1, 1)

    expected = ref[year].predict_proba(probs, expand=True)
    actual = predict_proba_year(calibrator_loaded[year], probs, expand=True)

    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    np.testing.assert_allclose(actual, expected, atol=ATOL_PARITY, rtol=RTOL)


# ---------------------------------------------------------------------------
# Fixture parity (no pickle needed).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("decode", ["pydicom", "dcmtk"])
def test_fixture_parity(calibrator_loaded: dict, decode: str) -> None:
    """apply_calibrator_json on raw_sigmoid{,_dcmtk}.npy == calibrated{,_dcmtk}.npy."""
    suffix = "_dcmtk" if decode == "dcmtk" else ""
    raw_sigmoid = np.load(FIXTURES_DIR / f"raw_sigmoid{suffix}.npy")
    gold = np.load(FIXTURES_DIR / f"calibrated{suffix}.npy")

    assert raw_sigmoid.shape == (1, 5), raw_sigmoid.shape
    assert gold.shape == (5,), gold.shape
    assert gold.dtype == np.float64, gold.dtype

    out = apply_calibrator_json(calibrator_loaded, raw_sigmoid)
    assert out.shape == (5,), out.shape
    assert out.dtype == np.float64, out.dtype

    max_abs = float(np.abs(out - gold).max())
    np.testing.assert_allclose(
        out, gold, atol=ATOL_PARITY, rtol=RTOL,
        err_msg=(
            f"[{decode}] calibrator-from-json output diverges from calibrated{suffix}.npy "
            f"(max abs diff {max_abs:.3e}, tolerance {ATOL_PARITY:.0e})"
        ),
    )


@pytest.mark.parametrize(
    "decode,expected",
    [
        ("pydicom", EXPECTED_PREDICTIONS_PYDICOM),
        ("dcmtk", EXPECTED_PREDICTIONS_DCMTK),
    ],
)
def test_predictions_4dp_parity(calibrator_loaded: dict, decode: str, expected) -> None:
    """Post-4dp-round, matches the pinned demo predictions from plan §1 bit-exact.

    Also cross-checks against the Phase 0 predictions JSON, which is what the
    upstream `mirai-predict` CLI emits.
    """
    suffix = "_dcmtk" if decode == "dcmtk" else ""
    raw_sigmoid = np.load(FIXTURES_DIR / f"raw_sigmoid{suffix}.npy")

    out = apply_calibrator_json(calibrator_loaded, raw_sigmoid)
    rounded = np.round(out, 4).tolist()

    assert rounded == expected, (
        f"[{decode}] rounded predictions {rounded} != pinned {expected}"
    )

    # Also match Phase 0's captured predictions JSON exactly. Fixture shape is
    # {"predictions": {"Year 1": ..., "Year 2": ..., ...}, "modelVersion": "..."}.
    phase0 = json.loads((FIXTURES_DIR / f"predictions{suffix}.json").read_text())
    phase0_values = [
        round(float(phase0["predictions"][f"Year {i + 1}"]), 4)
        for i in range(EXPECTED_N_YEARS)
    ]
    assert rounded == phase0_values, (rounded, phase0_values)


# ---------------------------------------------------------------------------
# Behavioral: expand=True shape semantics.
# ---------------------------------------------------------------------------
def test_expand_true_returns_complementary_rows(calibrator_loaded: dict) -> None:
    """expand=True returns [1 - p, p] with matching shapes (matches calibrator.py:35-36)."""
    params = calibrator_loaded[0]
    X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=np.float64)

    expanded = predict_proba_year(params, X, expand=True)
    pos_only = predict_proba_year(params, X, expand=False)

    assert expanded.shape == (2,) + pos_only.shape, (expanded.shape, pos_only.shape)
    np.testing.assert_array_equal(expanded[1], pos_only)
    np.testing.assert_allclose(expanded[0], 1.0 - pos_only, atol=0.0, rtol=0.0)
