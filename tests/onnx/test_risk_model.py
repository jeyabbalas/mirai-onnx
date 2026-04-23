"""Phase 3 verification: risk_model.onnx is well-formed and numerically faithful.

Run inside the mirai-export env (torch>=2.1, onnx>=1.15, onnxruntime>=1.17). The
export script scripts/export_risk_model.py must have been run first.
"""

from __future__ import annotations

import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from .conftest import (
    ATOL_ORT,
    FIXTURES_DIR,
    RISK_MODEL_ONNX,
    RTOL,
)


# ---------------------------------------------------------------------------
# Module-level skip: require the ONNX file to exist.
# ---------------------------------------------------------------------------
if not RISK_MODEL_ONNX.exists():
    pytest.skip(
        f"{RISK_MODEL_ONNX} not found; run `python scripts/export_risk_model.py` first.",
        allow_module_level=True,
    )


# Risk model is much smaller than the image encoder (no Conv stack). Cap at 50 MB.
MAX_FILE_SIZE_MB = 50

EXPECTED_INPUT_NAMES = [
    "img_feats", "view_seq", "side_seq", "time_seq", "rf_vector", "rf_known_mask",
]
EXPECTED_OUTPUT_NAMES = ["logit", "hidden_pre_hazard"]
EXPECTED_INPUT_SHAPES = {
    "img_feats": ["B", 4, 512],
    "view_seq": ["B", 4],
    "side_seq": ["B", 4],
    "time_seq": ["B", 4],
    "rf_vector": ["B", 100],
    "rf_known_mask": ["B", 100],
}
EXPECTED_OUTPUT_SHAPES = {
    "logit": ["B", 5],
    "hidden_pre_hazard": ["B", 612],
}
ONNX_FLOAT = 1
ONNX_INT64 = 7


@pytest.fixture(scope="module")
def proto() -> onnx.ModelProto:
    return onnx.load(str(RISK_MODEL_ONNX))


@pytest.fixture(scope="module")
def session() -> ort.InferenceSession:
    return ort.InferenceSession(
        str(RISK_MODEL_ONNX), providers=["CPUExecutionProvider"]
    )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _slot_seqs(suffix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = json.loads((FIXTURES_DIR / f"batch_order{suffix}.json").read_text())
    views = np.array([[int(e["view"]) for e in order]], dtype=np.int64)
    sides = np.array([[int(e["side"]) for e in order]], dtype=np.int64)
    times = np.zeros((1, len(order)), dtype=np.int64)
    return views, sides, times


def _load_inputs(decode: str) -> dict:
    """Build the demo input dict (rf_vector / rf_known_mask = zeros invokes
    the model-predicted-RF path, matching the Phase 0 demo)."""
    suffix = "_dcmtk" if decode == "dcmtk" else ""
    img_feats = np.load(FIXTURES_DIR / f"image_encoder_out{suffix}.npy").astype(np.float32, copy=False)
    assert img_feats.shape == (1, 4, 512), img_feats.shape
    view_seq, side_seq, time_seq = _slot_seqs(suffix)
    return {
        "img_feats": img_feats,
        "view_seq": view_seq,
        "side_seq": side_seq,
        "time_seq": time_seq,
        "rf_vector": np.zeros((1, 100), dtype=np.float32),
        "rf_known_mask": np.zeros((1, 100), dtype=np.float32),
    }


def _load_gold(decode: str) -> tuple[np.ndarray, np.ndarray]:
    suffix = "_dcmtk" if decode == "dcmtk" else ""
    return (
        np.load(FIXTURES_DIR / f"raw_logit{suffix}.npy"),
        np.load(FIXTURES_DIR / f"xai_hidden{suffix}.npy"),
    )


def _value_info_shape(value_info) -> list:
    return [
        d.dim_param if d.HasField("dim_param") else d.dim_value
        for d in value_info.type.tensor_type.shape.dim
    ]


# ---------------------------------------------------------------------------
# Structural tests.
# ---------------------------------------------------------------------------
def test_checker_passes(proto: onnx.ModelProto) -> None:
    onnx.checker.check_model(proto)


def test_file_size_under_50mb() -> None:
    size_mb = RISK_MODEL_ONNX.stat().st_size / 1024 / 1024
    assert size_mb < MAX_FILE_SIZE_MB, f"ONNX file is {size_mb:.1f} MB (>= {MAX_FILE_SIZE_MB} MB cap)"


def test_no_dynamic_branches(proto: onnx.ModelProto) -> None:
    """No If/Loop/Scan means no Python branch (e.g. RiskFactorPool's `self.training`
    branch, or AllImageTransformer's mask_input bernoulli) leaked into the trace."""
    bad = [n.op_type for n in proto.graph.node if n.op_type in {"If", "Loop", "Scan"}]
    assert not bad, f"Dynamic control-flow nodes found in graph: {bad}"


def test_opset_17(proto: onnx.ModelProto) -> None:
    default_opset = [op.version for op in proto.opset_import if op.domain in ("", "ai.onnx")]
    assert 17 in default_opset, f"expected opset 17, got {default_opset}"


def test_input_output_spec(proto: onnx.ModelProto) -> None:
    # Run shape inference so the `logit` output's second dim resolves from the
    # post-Add symbolic name to the literal 5 (the tracer leaves it symbolic
    # because Cumulative_Probability_Layer.forward derives `T` via .size()).
    inferred = onnx.shape_inference.infer_shapes(proto)

    input_names = [v.name for v in inferred.graph.input]
    output_names = [v.name for v in inferred.graph.output]
    assert input_names == EXPECTED_INPUT_NAMES, input_names
    assert output_names == EXPECTED_OUTPUT_NAMES, output_names

    by_input = {v.name: v for v in inferred.graph.input}
    for name, expected_shape in EXPECTED_INPUT_SHAPES.items():
        v = by_input[name]
        assert _value_info_shape(v) == expected_shape, (name, _value_info_shape(v))
        expected_dtype = ONNX_INT64 if name in {"view_seq", "side_seq", "time_seq"} else ONNX_FLOAT
        assert v.type.tensor_type.elem_type == expected_dtype, (name, v.type.tensor_type.elem_type)

    by_output = {v.name: v for v in inferred.graph.output}
    for name, expected_shape in EXPECTED_OUTPUT_SHAPES.items():
        v = by_output[name]
        assert _value_info_shape(v) == expected_shape, (name, _value_info_shape(v))
        assert v.type.tensor_type.elem_type == ONNX_FLOAT, (name, v.type.tensor_type.elem_type)


# ---------------------------------------------------------------------------
# Numerical parity tests (pydicom + dcmtk decode paths).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("decode", ["pydicom", "dcmtk"])
def test_parity_logit(session: ort.InferenceSession, decode: str) -> None:
    inputs = _load_inputs(decode)
    gold_logit, _ = _load_gold(decode)
    ort_logit, _ = session.run(None, inputs)
    assert ort_logit.shape == (1, 5), ort_logit.shape
    assert ort_logit.dtype == np.float32, ort_logit.dtype
    max_abs = float(np.abs(ort_logit - gold_logit).max())
    np.testing.assert_allclose(
        ort_logit, gold_logit, atol=ATOL_ORT, rtol=RTOL,
        err_msg=(
            f"[{decode}] onnxruntime logit diverges from raw_logit"
            f"{'_dcmtk' if decode=='dcmtk' else ''}.npy "
            f"(max abs diff {max_abs:.3e}, tolerance {ATOL_ORT:.0e})"
        ),
    )


@pytest.mark.parametrize("decode", ["pydicom", "dcmtk"])
def test_parity_hidden(session: ort.InferenceSession, decode: str) -> None:
    inputs = _load_inputs(decode)
    _, gold_hidden = _load_gold(decode)
    _, ort_hidden = session.run(None, inputs)
    assert ort_hidden.shape == (1, 612), ort_hidden.shape
    assert ort_hidden.dtype == np.float32, ort_hidden.dtype
    max_abs = float(np.abs(ort_hidden - gold_hidden).max())
    np.testing.assert_allclose(
        ort_hidden, gold_hidden, atol=ATOL_ORT, rtol=RTOL,
        err_msg=(
            f"[{decode}] onnxruntime hidden_pre_hazard diverges from xai_hidden"
            f"{'_dcmtk' if decode=='dcmtk' else ''}.npy "
            f"(max abs diff {max_abs:.3e}, tolerance {ATOL_ORT:.0e})"
        ),
    )


def test_dynamic_batch_axis(session: ort.InferenceSession) -> None:
    """Run B=1 and B=2 (duplicated input); B=2 outputs must be bit-exact slice copies."""
    inputs = _load_inputs("pydicom")
    logit_one, hidden_one = session.run(None, inputs)
    assert logit_one.shape == (1, 5)
    assert hidden_one.shape == (1, 612)

    inputs_b2 = {k: np.concatenate([v, v], axis=0) for k, v in inputs.items()}
    logit_two, hidden_two = session.run(None, inputs_b2)
    assert logit_two.shape == (2, 5)
    assert hidden_two.shape == (2, 612)

    np.testing.assert_array_equal(logit_two[0:1], logit_one)
    np.testing.assert_array_equal(logit_two[1:2], logit_one)
    np.testing.assert_array_equal(hidden_two[0:1], hidden_one)
    np.testing.assert_array_equal(hidden_two[1:2], hidden_one)


# ---------------------------------------------------------------------------
# Behavioral tests for the rf_known_mask blend (single decode path; pydicom).
# ---------------------------------------------------------------------------
def test_zero_mask_invokes_predicted_rfs(session: ort.InferenceSession) -> None:
    """rf_vector=0, rf_known_mask=0 -> last 100 dims of hidden = relu(model-predicted RFs).

    Phase 0 already proved (test_baseline cross-check #4) that the per-key probs
    concatenated in `risk_factor_keys` order equal `risk_factor_vector.npy`. So
    matching `relu(risk_factor_vector.npy)` here transitively proves per-key correctness.
    """
    inputs = _load_inputs("pydicom")
    _, hidden = session.run(None, inputs)

    rf_vec_pred = np.load(FIXTURES_DIR / "risk_factor_vector.npy")
    expected_rf_block = np.maximum(0.0, rf_vec_pred)
    actual_rf_block = hidden[:, -100:]
    max_abs = float(np.abs(actual_rf_block - expected_rf_block).max())
    np.testing.assert_allclose(
        actual_rf_block, expected_rf_block, atol=ATOL_ORT, rtol=RTOL,
        err_msg=(
            f"hidden_pre_hazard[:, -100:] does not match relu(risk_factor_vector.npy) "
            f"(max abs diff {max_abs:.3e})"
        ),
    )


def test_image_hidden_matches_phase0(session: ort.InferenceSession) -> None:
    """First 512 dims of hidden_pre_hazard equal relu(image_hidden_in_pool.npy)."""
    inputs = _load_inputs("pydicom")
    _, hidden = session.run(None, inputs)

    image_hidden = np.load(FIXTURES_DIR / "image_hidden_in_pool.npy")
    assert image_hidden.shape == (1, 512), image_hidden.shape
    expected_image_block = np.maximum(0.0, image_hidden)
    actual_image_block = hidden[:, :512]
    max_abs = float(np.abs(actual_image_block - expected_image_block).max())
    np.testing.assert_allclose(
        actual_image_block, expected_image_block, atol=ATOL_ORT, rtol=RTOL,
        err_msg=(
            f"hidden_pre_hazard[:, :512] does not match relu(image_hidden_in_pool.npy) "
            f"(max abs diff {max_abs:.3e})"
        ),
    )


def test_user_supplied_rfs_pass_through(session: ort.InferenceSession) -> None:
    """rf_known_mask=ones -> last 100 dims of hidden = relu(rf_vector) bit-exact.

    Proves the mask blend isn't fused/folded away. With rf_known_mask=ones, the
    predicted-RF branch contributes 0 (1 - 1 = 0 exactly in fp32), so the
    user-supplied vector flows through unchanged. We use a vector with both negative
    and positive values so ReLU is also exercised, not just identity.
    """
    inputs = _load_inputs("pydicom")
    user_rfs = np.linspace(-0.5, 1.0, 100, dtype=np.float32).reshape(1, 100)
    inputs["rf_vector"] = user_rfs
    inputs["rf_known_mask"] = np.ones((1, 100), dtype=np.float32)

    _, hidden = session.run(None, inputs)
    expected_rf_block = np.maximum(0.0, user_rfs)
    actual_rf_block = hidden[:, -100:]
    np.testing.assert_array_equal(
        actual_rf_block, expected_rf_block,
        err_msg="user-supplied rf_vector did not flow through the mask blend bit-exactly",
    )
