"""Phase 4: Export the Mirai calibrator to a portable JSON format.

Produces models/calibrator.json with 5 year entries (Platt-scaling parameters) so
TypeScript and other non-Python runtimes can apply the per-year calibration
without unpickling a `MiraiCalibrator`.

Source: external/Mirai/onconet/models/calibrator.py:4-38 (pure numpy; no torch).
Contract source: mirai-migration-plan.md §5.

Output schema:

    {
      "schema_version": 1,
      "source_pickle_sha256": "<64-hex>",
      "years": [
        {"index": 0, "base_slope": ..., "base_offset": ...,
         "calibrator_slope": ..., "calibrator_offset": ...},
        ...  // 5 entries total, indices 0..4
      ]
    }

JSON key naming follows the plan's shortened convention (`base_slope` / `base_offset`)
even though the Python attributes are `base_estimator_slope` / `base_estimator_offset`.
The TypeScript port in Phase 8 will hard-code these shortened keys.

Usage:
    conda activate mirai-py38
    python scripts/export_calibrator.py
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import pickle
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MIRAI_ROOT = PROJECT_ROOT / "external" / "Mirai"
MIRAI_CONFIG = MIRAI_ROOT / "onconet" / "configs" / "mirai_trained.json"
OUT_DIR = PROJECT_ROOT / "models"
OUT_PATH = OUT_DIR / "calibrator.json"

# Pinned in tests/reference/fixtures/MANIFEST.json -> snapshots.calibrator_path.
EXPECTED_SHA256 = "822092d81272c97883d54a4bde0bc1cdcebadb861dea8467cb93201fedb73efa"
EXPECTED_N_YEARS = 5
SCHEMA_VERSION = 1


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _coerce_scalar(x, name: str) -> float:
    """Accept numpy 0-d array, single-element 1-D array, numpy scalar, or Python
    float; return a Python float.

    sklearn's CalibratedClassifierCV populates these from coef_[0] / intercept_[0] /
    a_ / b_. With a scalar base estimator fit on a single feature, `coef_[0]` is
    1-D of shape (1,) rather than a 0-d scalar; flatten and take the single
    element. The calibrator math (`slope * X`) broadcasts identically either way,
    so storing the value as a float preserves semantics exactly.
    """
    arr = np.asarray(x).reshape(-1)
    if arr.size != 1:
        raise RuntimeError(f"{name}: expected 1-element scalar, got shape {np.asarray(x).shape}")
    value = float(arr[0])
    if not math.isfinite(value):
        raise RuntimeError(f"{name}: non-finite value {value!r}")
    return value


def _resolve_calibrator_path() -> pathlib.Path:
    cfg = json.loads(MIRAI_CONFIG.read_text())
    raw = cfg.get("calibrator_path")
    if not raw:
        raise RuntimeError(f"calibrator_path missing from {MIRAI_CONFIG}")
    return pathlib.Path(raw).expanduser().resolve()


def main() -> int:
    # Allow pickle.load to reconstruct MiraiCalibrator objects.
    # Pure-numpy module; no torch is imported transitively.
    sys.path.insert(0, str(MIRAI_ROOT))
    from onconet.models.calibrator import MiraiCalibrator  # noqa: F401

    pkl_path = _resolve_calibrator_path()
    print(f"[export] calibrator pickle: {pkl_path}")
    if not pkl_path.exists():
        raise RuntimeError(f"calibrator pickle not found at {pkl_path}")

    actual_sha = _sha256(pkl_path)
    print(f"[export] sha256: {actual_sha}")
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError(
            f"calibrator pickle SHA mismatch:\n  expected {EXPECTED_SHA256}\n  actual   {actual_sha}"
        )

    with pkl_path.open("rb") as f:
        calibrator = pickle.load(f)

    if not isinstance(calibrator, dict):
        raise RuntimeError(f"expected dict of calibrators, got {type(calibrator).__name__}")
    expected_keys = set(range(EXPECTED_N_YEARS))
    if set(calibrator.keys()) != expected_keys:
        raise RuntimeError(
            f"expected year indices {sorted(expected_keys)}, got {sorted(calibrator.keys())}"
        )

    years = []
    for i in range(EXPECTED_N_YEARS):
        c = calibrator[i]
        if type(c).__name__ != "MiraiCalibrator":
            raise RuntimeError(
                f"year {i}: expected MiraiCalibrator, got {type(c).__name__}"
            )
        entry = {
            "index": i,
            "base_slope": _coerce_scalar(c.base_estimator_slope, f"year {i} base_estimator_slope"),
            "base_offset": _coerce_scalar(c.base_estimator_offset, f"year {i} base_estimator_offset"),
            "calibrator_slope": _coerce_scalar(c.calibrator_slope, f"year {i} calibrator_slope"),
            "calibrator_offset": _coerce_scalar(c.calibrator_offset, f"year {i} calibrator_offset"),
        }
        print(
            f"[export] year {i}: base_slope={entry['base_slope']:.6g} "
            f"base_offset={entry['base_offset']:.6g} "
            f"calibrator_slope={entry['calibrator_slope']:.6g} "
            f"calibrator_offset={entry['calibrator_offset']:.6g}"
        )
        years.append(entry)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_pickle_sha256": EXPECTED_SHA256,
        "years": years,
    }
    with OUT_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    out_sha = _sha256(OUT_PATH)
    print(f"[export] wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes, sha256 {out_sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
