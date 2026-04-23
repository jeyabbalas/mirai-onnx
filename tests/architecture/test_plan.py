"""Lint-style tests for docs/architecture.md (Phase 1 deliverable).

Stdlib-only. No torch, no numpy, no Mirai imports. This module asserts that the
architecture document contains the committed strings and structure that Phase 2
and Phase 3 will grep for when producing the ONNX exports. If these tests fail
the document has drifted out of sync with the planned export surface.
"""

from __future__ import annotations

import pathlib
import re

import pytest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DOC_PATH = PROJECT_ROOT / "docs" / "architecture.md"


REQUIRED_H2_SECTIONS = [
    "## 1. Scope and Audience",
    "## 2. Source Authority and Provenance",
    "## 3. End-to-End Shape Table",
    "## 4. nn.Module Dependency Graph",
    "## 5. Control-Flow Branches and Resolution at Export",
    "## 6. Eval-Mode Situation",
    "## 7. Per-Key FC Double-Call",
    "## 8. XAI-Hidden Choice (Pre-relu vs Post-relu)",
    "## 9. Export Surface Specification",
    "## 10. Slot-Ordering Convention",
    "## 11. Refactor Targets for Phase 2/3",
    "## 12. Export-Time Settings",
    "## 13. Snapshot Loading and .eval() Sequencing",
    "## 14. Notes and Pitfalls",
    "## 15. How to Validate This Plan",
]

RISK_MODEL_INPUT_NAMES = [
    "img_feats",
    "view_seq",
    "side_seq",
    "time_seq",
    "rf_vector",
    "rf_known_mask",
]

RISK_MODEL_OUTPUT_NAMES = ["logit", "hidden_pre_hazard"]

RISK_MODEL_SHAPE_STRINGS = [
    "(B, 4, 512)",
    "(B, 4)",
    "(B, 100)",
    "(B, 5)",
    "(B, 612)",
]

PHASE0_FIXTURE_STEMS = [
    "image_encoder_out",
    "risk_factor_vector",
    "pool_hidden",
    "xai_hidden",
    "raw_logit",
    "batch_order",
]

FORBIDDEN_RF_DIM_PHRASES = [
    "rf_dim = 34",
    "rf_dim=34",
    "rf_dim is 34",
]

FORBIDDEN_SLOT_ORDER_LITERAL = "R-CC, R-MLO, L-CC, L-MLO"


@pytest.fixture(scope="module")
def doc_text() -> str:
    assert DOC_PATH.exists(), f"architecture doc is missing: {DOC_PATH}"
    return DOC_PATH.read_text(encoding="utf-8")


def _section_body(doc: str, heading: str) -> str:
    """Return text from `heading` up to the next H2 heading."""
    start = doc.index(heading)
    after = doc[start + len(heading):]
    match = re.search(r"\n## ", after)
    return after[: match.start()] if match else after


# T1
def test_doc_exists() -> None:
    assert DOC_PATH.exists(), f"missing {DOC_PATH}"


# T2 — one test id per required heading
@pytest.mark.parametrize("heading", REQUIRED_H2_SECTIONS)
def test_required_h2_heading_present(doc_text: str, heading: str) -> None:
    pattern = re.compile(rf"^{re.escape(heading)}$", re.MULTILINE)
    assert pattern.search(doc_text), (
        f"H2 heading missing or not on its own line: {heading!r}"
    )


# T3
def test_image_encoder_io(doc_text: str) -> None:
    body = _section_body(doc_text, "## 9. Export Surface Specification")
    # the image-encoder subsection is fenced by "### 9.1" ... "### 9.2"
    sub_start = body.index("### 9.1")
    sub_end = body.index("### 9.2")
    section = body[sub_start:sub_end]
    for needle in ["input", "output", "(N, 3, 2048, 1664)", "(N, 512)", '"N"']:
        assert needle in section, f"§9.1 missing literal {needle!r}"
    assert re.search(r"dynamic", section, re.IGNORECASE), (
        "§9.1 must mention dynamic axes"
    )


# T4
@pytest.mark.parametrize("name", RISK_MODEL_INPUT_NAMES)
def test_risk_model_input_name(doc_text: str, name: str) -> None:
    assert name in doc_text, f"risk_model input name {name!r} missing from doc"


# T5
@pytest.mark.parametrize("name", RISK_MODEL_OUTPUT_NAMES)
def test_risk_model_output_name(doc_text: str, name: str) -> None:
    assert name in doc_text, f"risk_model output name {name!r} missing from doc"


# T6
@pytest.mark.parametrize("shape", RISK_MODEL_SHAPE_STRINGS)
def test_risk_model_shape(doc_text: str, shape: str) -> None:
    assert shape in doc_text, f"risk_model shape {shape!r} missing from doc"


# T7
def test_risk_model_dtypes(doc_text: str) -> None:
    body = _section_body(doc_text, "## 9. Export Surface Specification")
    for dtype in ["fp32", "int64"]:
        assert dtype in body, f"§9 missing dtype {dtype!r}"


# T8
def test_opset_committed(doc_text: str) -> None:
    assert re.search(r"opset_version\s*[=:]\s*17\b", doc_text) or (
        "opset_version 17" in doc_text
    ), "doc must commit to opset_version=17"


# T9
def test_xai_choice_unambiguous(doc_text: str) -> None:
    assert "XAI-hidden = post-relu" in doc_text, (
        "doc must commit: XAI-hidden = post-relu"
    )
    assert "XAI-hidden = pre-relu" not in doc_text, (
        "doc must not leave pre-relu as the committed XAI-hidden choice"
    )


# T10
def test_rf_dim_dynamic(doc_text: str) -> None:
    assert "length_risk_factor_vector" in doc_text, (
        "doc must cite length_risk_factor_vector as the runtime source of rf_dim"
    )
    for forbidden in FORBIDDEN_RF_DIM_PHRASES:
        assert forbidden.lower() not in doc_text.lower(), (
            f"doc contains forbidden rf_dim literal {forbidden!r}"
        )
    assert "rf_dim" in doc_text, "doc must mention rf_dim"
    assert "100" in doc_text, "doc must mention 100 (the rf_dim value)"


# T11
def test_batch_order_cited(doc_text: str) -> None:
    assert "batch_order.json" in doc_text, (
        "doc must cite batch_order.json as the slot-order source of truth"
    )
    assert FORBIDDEN_SLOT_ORDER_LITERAL not in doc_text, (
        f"doc must not hardcode slot order {FORBIDDEN_SLOT_ORDER_LITERAL!r}"
    )


# T12
def test_eval_mode_documented(doc_text: str) -> None:
    body = _section_body(doc_text, "## 6. Eval-Mode Situation")
    for needle in [".eval()", "upstream", "eval"]:
        assert needle in body, f"§6 missing literal {needle!r}"


# T13
def test_per_key_fc_double_call_documented(doc_text: str) -> None:
    body = _section_body(doc_text, "## 7. Per-Key FC Double-Call")
    assert "_fc" in body, "§7 must reference the {key}_fc modules"
    assert "get_pred_rf_loss" in body, (
        "§7 must cite get_pred_rf_loss as the second-call site"
    )
    assert re.search(r"exactly once", body, re.IGNORECASE), (
        "§7 must state the 'exactly once' export rule"
    )


# T14
@pytest.mark.parametrize("stem", PHASE0_FIXTURE_STEMS)
def test_phase0_fixture_referenced(doc_text: str, stem: str) -> None:
    assert stem in doc_text, (
        f"doc must reference the Phase 0 fixture stem {stem!r} as a validation target"
    )


# T15
def test_refactor_targets_have_file_line_refs(doc_text: str) -> None:
    body = _section_body(doc_text, "## 11. Refactor Targets for Phase 2/3")
    matches = re.findall(r"external/Mirai/onconet/[\w/.]+\.py:\d+", body)
    assert len(matches) >= 3, (
        f"§11 must contain at least 3 file:line refs under external/Mirai/onconet/, "
        f"found {len(matches)}"
    )
