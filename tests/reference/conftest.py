"""Path constants and pytest fixtures shared by every test under tests/reference/."""

from __future__ import annotations

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
MIRAI_ROOT = PROJECT_ROOT / "external" / "Mirai"
MIRAI_CONFIG = MIRAI_ROOT / "onconet" / "configs" / "mirai_trained.json"
DEMO_DATA_DIR = PROJECT_ROOT / "mirai_demo_data"
DEMO_DICOMS = [
    DEMO_DATA_DIR / "ccl1.dcm",
    DEMO_DATA_DIR / "ccr1.dcm",
    DEMO_DATA_DIR / "mlol2.dcm",
    DEMO_DATA_DIR / "mlor2.dcm",
]

FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"
PREVIEW_DIR = FIXTURES_DIR / "preview"

# Tolerances. Tight; we run on the same machine/wheel that captured the fixtures.
ATOL_FP32 = 1e-6
RTOL_FP32 = 0.0
ATOL_FP64 = 1e-9
