# Phase 4 Report — Calibrator Portable Export

**Date:** 2026-04-23
**Branch:** main
**Status:** Complete. `models/calibrator.json` exported and validated; 20/20 calibrator tests green; Phase 0 baseline (56/56), Phase 2 image-encoder tests (8/8), and Phase 3 risk-model tests (13/13) still green.

## Deliverables

| File | Purpose |
|---|---|
| `models/calibrator.json` | Portable Platt-scaling parameters for 5 years (1162 bytes). |
| `scripts/export_calibrator.py` | One-shot export. Loads the pickled `MiraiCalibrator` dict, verifies the source SHA-256, coerces each per-year scalar, writes JSON. |
| `scripts/calibrator_from_json.py` | Pure-numpy helper: `load_calibrator`, `predict_proba_year`, `apply_calibrator_json`. No torch, no onconet. Used by Phase 5's end-to-end ONNX pipeline. |
| `tests/calibrator/conftest.py` | Paths + tolerance constants (`ATOL_PARITY = 1e-9`, `RTOL = 0.0`) + pinned demo prediction vectors. |
| `tests/calibrator/test_calibrator_json.py` | 20 parametrized pytest cases — structural, per-year random-input parity, Phase 0 fixture parity, 4-dp rounded parity, and `expand=True` shape semantics. |

No edits under `external/Mirai/` or `tests/reference/fixtures/`.

## JSON schema

```json
{
  "schema_version": 1,
  "source_pickle_sha256": "822092d81272c97883d54a4bde0bc1cdcebadb861dea8467cb93201fedb73efa",
  "years": [
    {"index": 0, "base_slope": 4.60744..., "base_offset": -6.6888..., "calibrator_slope": -1.12909..., "calibrator_offset": -0.610537...},
    {"index": 1, ...},
    {"index": 2, ...},
    {"index": 3, ...},
    {"index": 4, "base_slope": 3.40573..., "base_offset": -4.63191..., "calibrator_slope": -1.14641..., "calibrator_offset": -0.533472...}
  ]
}
```

Key-name mapping vs the source class (`external/Mirai/onconet/models/calibrator.py:4-38`):

| JSON key | `MiraiCalibrator` attribute |
|---|---|
| `base_slope` | `base_estimator_slope` |
| `base_offset` | `base_estimator_offset` |
| `calibrator_slope` | `calibrator_slope` |
| `calibrator_offset` | `calibrator_offset` |

The shortened JSON keys follow `mirai-migration-plan.md` §5 and will be hard-coded by the TypeScript port in Phase 8.

## Formula

Line-for-line port of `MiraiCalibrator.predict_proba` (calibrator.py:32-38):

```
_y = base_slope * X + base_offset
_y = calibrator_slope * _y + calibrator_offset
pos_prob = 1 / (1 + exp(_y))            # note: exp(+_y), not exp(-_y)
expand=True  → np.array([1 - pos_prob, pos_prob])
expand=False → pos_prob
```

The **exp(+_y)** sign is a deliberate quirk of how this particular calibrator was fit — do not "correct" it to `exp(-_y)` in any downstream port.

## Parity measurements

All measurements at `atol=1e-9`, `rtol=0`, run on the capture host.

| Check | Max abs diff |
|---|---|
| `apply_calibrator_json(raw_sigmoid.npy)` vs `calibrated.npy` (pydicom) | **0.0** (bit-exact) |
| `apply_calibrator_json(raw_sigmoid_dcmtk.npy)` vs `calibrated_dcmtk.npy` | **0.0** (bit-exact) |
| Per-year random-input parity vs pickled `predict_proba` (5 × 100 samples) | **0.0** (bit-exact, worst of all 500 samples) |

Both pydicom and dcmtk 4-dp-rounded outputs match the pinned demo predictions in `mirai-migration-plan.md` §1:

| Year | pydicom | dcmtk |
|---|---|---|
| 1 | 0.0314 | 0.0298 |
| 2 | 0.0505 | 0.0483 |
| 3 | 0.0711 | 0.0684 |
| 4 | 0.0935 | 0.09 |
| 5 | 0.1052 | 0.1016 |

The bit-exact result is expected: arithmetic is `float64 × float64` + `np.exp`, the operation order is preserved from the capture path (`tests/reference/capture_reference.py:139-148`), and numpy broadcasts `slope (1,)` against `X (n,1)` identically to `float × X (n,1)`. Storing the slope as a Python float instead of a shape-(1,) numpy array is a lossless coercion.

## Scalar coercion note

`MiraiCalibrator.base_estimator_slope` arrives from `sklearn.linear_model._base.LinearClassifierMixin.coef_[0]`, which is shape `(1,)` for a single-feature base estimator — not a 0-d scalar. `export_calibrator.py::_coerce_scalar` flattens and extracts the single element via `float(arr[0])`. The other three attributes (`base_estimator_offset` from `intercept_[0]`, `calibrator_slope` from `a_`, `calibrator_offset` from `b_`) are already 0-d or Python floats. The coercion is documented in the script; all five years coerce cleanly.

## Tests

`tests/calibrator/test_calibrator_json.py` — 20 cases:

Structural (5):
- `test_schema_version` — `schema_version == 1`.
- `test_source_sha_matches_pickle` — JSON's claimed SHA matches the pinned value, and (if the pickle is present) matches the on-disk pickle.
- `test_years_structure` — exactly 5 entries, all four scalars per entry, indices 0..4, no NaN/Inf.
- `test_years_sorted_by_index` — list is emitted in ascending index order (so TS can index by position).
- `test_load_calibrator_returns_dict` — helper returns the expected shape.

Per-year random-input parity, parametrized over year 0..4 (10):
- `test_per_year_random_parity[0..4]` — 100 rng-seeded probabilities in `(1e-6, 1 - 1e-6)`, JSON helper vs pickled `predict_proba(expand=False)` within `atol=1e-9` (observed: **0.0** bit-exact).
- `test_per_year_random_parity_expand[0..4]` — same with `expand=True`, confirms `(2, n)` shape matches.

Fixture parity (2):
- `test_fixture_parity[pydicom]` / `test_fixture_parity[dcmtk]` — `apply_calibrator_json` on `raw_sigmoid{,_dcmtk}.npy` vs `calibrated{,_dcmtk}.npy` within `atol=1e-9`.

4-dp rounded parity (2):
- `test_predictions_4dp_parity[pydicom-expected0]` / `test_predictions_4dp_parity[dcmtk-expected1]` — after 4-decimal rounding, matches both the plan's pinned vector **and** Phase 0's `predictions{,_dcmtk}.json` (nested-dict format `{"predictions": {"Year 1": ..., ...}}`).

Expand semantics (1):
- `test_expand_true_returns_complementary_rows` — `expand=True` returns `[1-p, p]` with row-0 bit-exactly equal to `1 - row-1`.

Run summary:

```
$ /opt/homebrew/Caskroom/miniforge/base/envs/mirai-py38/bin/python -m pytest tests/calibrator/ -v
============================== 20 passed in 0.36s ==============================

$ /opt/homebrew/Caskroom/miniforge/base/envs/mirai-export/bin/python -m pytest tests/calibrator/ -v
============================== 20 passed in 1.72s ==============================
```

Phase 0 still green:

```
$ /opt/homebrew/Caskroom/miniforge/base/envs/mirai-py38/bin/python -m pytest tests/reference/test_baseline.py
======================= 56 passed, 3 warnings in 50.65s ========================
```

## Environment

Primary: `mirai-py38` (same env that captured Phase 0 fixtures). Reasons:
- The source pickle contains `MiraiCalibrator` instances; `pickle.load` needs that class importable. The class lives at `external/Mirai/onconet/models/calibrator.py` and has **no torch import** — only `numpy`.
- Running parity tests in the capture env closes the loop: if this env produced `calibrated{,_dcmtk}.npy`, reproducing them from JSON guarantees the JSON is faithful under identical numerics.

Cross-env verified: the same 20 tests also pass under `mirai-export` (used for Phase 2/3 ONNX export). The helper is environment-agnostic because its math is pure numpy scalars in fp64.

## Pinned facts (verified at runtime)

- Source pickle SHA-256 `822092d81272c97883d54a4bde0bc1cdcebadb861dea8467cb93201fedb73efa` (865 bytes) — matches `tests/reference/fixtures/MANIFEST.json → snapshots.calibrator_path`.
- Pickle is a `dict[int, MiraiCalibrator]` with keys exactly `{0, 1, 2, 3, 4}`.
- Each year's four scalars are finite; `base_slope` and `base_offset` arrive shape-(1,) (from sklearn `coef_[0]` / `intercept_[0]`); `calibrator_slope` and `calibrator_offset` arrive as 0-d numpy scalars.
- Output JSON is 1162 bytes, SHA-256 `952111bd9807478e20f88734bfe6895c99696173f130c56a24cc212de30beb22` on this run (not pinned — regenerates deterministically from the script + pickle).

## Out-of-scope notes for downstream phases

- **Phase 5** (`scripts/run_onnx_pipeline.py`) should `sys.path.insert(0, "scripts")` and call `from calibrator_from_json import apply_calibrator_json`. The helper returns `(5,)` fp64 — exactly what Phase 5's `onnx_prediction{,_dcmtk}.json` writer needs.
- **Phase 8** (browser TS) will hard-code the five JSON keys. The 15-line TS function is listed verbatim in `mirai-migration-plan.md` §9 task 1; the only subtlety a port can get wrong is the **sign of the exponent** (`exp(+y)`, not `exp(-y)`).
- **The `from_sk_calibrator` helper** at `calibrator.py:41-53` is for posterity only (building a `MiraiCalibrator` from a sklearn object). Phase 4 doesn't use it and Phase 8 will not need it either.
