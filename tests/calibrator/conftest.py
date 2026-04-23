"""Path constants and tolerances for Phase 4 calibrator-JSON tests.

Intentionally separate from tests/reference/conftest.py and tests/onnx/conftest.py
because Phase 4's math is pure numpy + fp64 scalars — no torch, no ORT. The
parity target is essentially bit-exact (atol=1e-9 on fp64) since we're replaying
the same four multiply-add-exp operations in the same order.

Runs cleanly under either `mirai-py38` or `mirai-export`; `mirai-py38` is
preferred because the parity tests cross-check against the pickled
MiraiCalibrator, which was the fixture-capture environment.
"""

from __future__ import annotations

import pathlib

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
MIRAI_ROOT = PROJECT_ROOT / "external" / "Mirai"
MIRAI_CONFIG = MIRAI_ROOT / "onconet" / "configs" / "mirai_trained.json"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "reference" / "fixtures"
MODELS_DIR = PROJECT_ROOT / "models"
CALIBRATOR_JSON = MODELS_DIR / "calibrator.json"

# Pinned in tests/reference/fixtures/MANIFEST.json -> snapshots.calibrator_path.
EXPECTED_SHA256 = "822092d81272c97883d54a4bde0bc1cdcebadb861dea8467cb93201fedb73efa"
EXPECTED_N_YEARS = 5
SCHEMA_VERSION = 1

# Phase 4 arithmetic is fp64 in + fp64 scalars + np.exp; same ordering as the
# Phase 0 _apply_calibrator code path. Effectively bit-exact.
ATOL_PARITY = 1e-9
RTOL = 0.0

# Pinned demo predictions (mirai-migration-plan.md §1). These are what
# `mirai-predict` emits on the four demo DICOMs after 4-decimal rounding.
EXPECTED_PREDICTIONS_PYDICOM = [0.0314, 0.0505, 0.0711, 0.0935, 0.1052]
EXPECTED_PREDICTIONS_DCMTK = [0.0298, 0.0483, 0.0684, 0.09, 0.1016]
