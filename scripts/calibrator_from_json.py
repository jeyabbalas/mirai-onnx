"""Phase 4: Apply the exported Mirai calibrator from its portable JSON format.

Line-for-line port of `MiraiCalibrator.predict_proba`
(external/Mirai/onconet/models/calibrator.py:32-38) plus a helper that
reproduces `_apply_calibrator` from tests/reference/capture_reference.py:139-148.

Phase 5's end-to-end ONNX pipeline calls `apply_calibrator_json` to calibrate the
sigmoid output of `risk_model.onnx` without touching the original pickle.

Pure numpy + json; no torch, no onconet dependency.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Dict, Mapping, Union

import numpy as np

SCHEMA_VERSION = 1
EXPECTED_N_YEARS = 5
YearParams = Dict[str, float]
CalibratorDict = Dict[int, YearParams]

PathLike = Union[str, pathlib.Path]


def load_calibrator(json_path: PathLike) -> CalibratorDict:
    """Load a calibrator from its JSON representation.

    Returns a dict mapping year_index (0..4) to a dict of the four Platt-scaling
    scalars (`base_slope`, `base_offset`, `calibrator_slope`, `calibrator_offset`).
    All values are Python floats (fp64).
    """
    path = pathlib.Path(json_path)
    with path.open("r") as f:
        payload = json.load(f)

    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported schema_version {version!r} (expected {SCHEMA_VERSION})"
        )
    years = payload.get("years")
    if not isinstance(years, list):
        raise ValueError(f"{path}: missing or non-list 'years'")
    if len(years) != EXPECTED_N_YEARS:
        raise ValueError(f"{path}: expected {EXPECTED_N_YEARS} year entries, got {len(years)}")

    required_keys = {"index", "base_slope", "base_offset", "calibrator_slope", "calibrator_offset"}
    out: CalibratorDict = {}
    for entry in years:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: year entry is not an object: {entry!r}")
        missing = required_keys - entry.keys()
        if missing:
            raise ValueError(f"{path}: year entry missing keys {sorted(missing)}: {entry!r}")
        idx = entry["index"]
        if not isinstance(idx, int) or idx < 0 or idx >= EXPECTED_N_YEARS:
            raise ValueError(f"{path}: invalid year index {idx!r}")
        if idx in out:
            raise ValueError(f"{path}: duplicate year index {idx}")
        params = {k: float(entry[k]) for k in ("base_slope", "base_offset",
                                                "calibrator_slope", "calibrator_offset")}
        for k, v in params.items():
            if not math.isfinite(v):
                raise ValueError(f"{path}: year {idx} {k} is non-finite: {v!r}")
        out[idx] = params

    if set(out.keys()) != set(range(EXPECTED_N_YEARS)):
        raise ValueError(f"{path}: year indices {sorted(out.keys())} != {list(range(EXPECTED_N_YEARS))}")
    return out


def predict_proba_year(
    params: Mapping[str, float],
    X: np.ndarray,
    expand: bool = True,
) -> np.ndarray:
    """Apply a single year's calibrator to probabilities in `X`.

    Mirrors `MiraiCalibrator.predict_proba` exactly
    (external/Mirai/onconet/models/calibrator.py:32-38):

        _y = base_slope * X + base_offset
        _y = calibrator_slope * _y + calibrator_offset
        pos_prob = 1 / (1 + exp(_y))         # note: exp(+_y), not exp(-_y)
        expand=True  -> np.array([1-pos_prob, pos_prob])
        expand=False -> pos_prob

    Arithmetic is done in fp64 regardless of `X.dtype`, matching the upstream
    Python scalars (fp64) × potentially-fp32 inputs promotion semantics.
    """
    X_f64 = np.asarray(X, dtype=np.float64)
    _y = params["base_slope"] * X_f64 + params["base_offset"]
    _y = params["calibrator_slope"] * _y + params["calibrator_offset"]
    pos_prob = 1.0 / (1.0 + np.exp(_y))
    if expand:
        return np.array([1.0 - pos_prob, pos_prob])
    return pos_prob


def apply_calibrator_json(calibrator: CalibratorDict, raw_sigmoid: np.ndarray) -> np.ndarray:
    """Apply the full per-year calibrator to a `(1, n_years)` sigmoid output.

    Mirrors `_apply_calibrator` (tests/reference/capture_reference.py:139-148).
    Returns a `(n_years,)` fp64 array.
    """
    raw_sigmoid = np.asarray(raw_sigmoid)
    if raw_sigmoid.ndim != 2 or raw_sigmoid.shape[0] != 1:
        raise ValueError(
            f"expected raw_sigmoid shape (1, n_years), got {raw_sigmoid.shape}"
        )
    n_years = raw_sigmoid.shape[1]
    if set(calibrator.keys()) != set(range(n_years)):
        raise ValueError(
            f"calibrator keys {sorted(calibrator.keys())} do not cover 0..{n_years - 1}"
        )

    out = np.zeros(n_years, dtype=np.float64)
    for i in range(n_years):
        x_i = raw_sigmoid[0, i].reshape(-1, 1)  # match capture shape exactly
        out[i] = predict_proba_year(calibrator[i], x_i, expand=True).flatten()[1]
    return out
