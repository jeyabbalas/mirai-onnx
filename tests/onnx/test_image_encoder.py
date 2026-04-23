"""Phase 2 verification: image_encoder.onnx is well-formed and numerically faithful.

Run inside the mirai-export env (torch>=2.1, onnx>=1.15, onnxruntime>=1.17). The
export script scripts/export_image_encoder.py must have been run first.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from .conftest import (
    ATOL_ORT,
    FIXTURES_DIR,
    IMAGE_ENCODER_ONNX,
    MAX_FILE_SIZE_MB,
    RTOL,
)


# ---------------------------------------------------------------------------
# Module-level skip: require the ONNX file to exist.
# ---------------------------------------------------------------------------
if not IMAGE_ENCODER_ONNX.exists():
    pytest.skip(
        f"{IMAGE_ENCODER_ONNX} not found; run `python scripts/export_image_encoder.py` first.",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def proto() -> onnx.ModelProto:
    return onnx.load(str(IMAGE_ENCODER_ONNX))


@pytest.fixture(scope="module")
def session() -> ort.InferenceSession:
    return ort.InferenceSession(
        str(IMAGE_ENCODER_ONNX), providers=["CPUExecutionProvider"]
    )


# ---------------------------------------------------------------------------
# Helper: load a 4-view stack for a given decode path (pydicom or dcmtk).
# ---------------------------------------------------------------------------
def _load_stack(decode: str):
    suffix = "_dcmtk" if decode == "dcmtk" else ""
    order = json.loads((FIXTURES_DIR / f"batch_order{suffix}.json").read_text())
    stack = np.stack(
        [
            np.load(FIXTURES_DIR / f"preproc_tensor{suffix}" / f"{e['view_str']}_{e['side_str']}.npy")
            for e in order
        ]
    ).astype(np.float32, copy=False)
    gold = np.load(FIXTURES_DIR / f"image_encoder_out{suffix}.npy").reshape(4, -1)
    return stack, gold


# ---------------------------------------------------------------------------
# Structural tests.
# ---------------------------------------------------------------------------
def test_checker_passes(proto: onnx.ModelProto) -> None:
    onnx.checker.check_model(proto)


def test_file_size_under_200mb() -> None:
    size_mb = IMAGE_ENCODER_ONNX.stat().st_size / 1024 / 1024
    assert size_mb < MAX_FILE_SIZE_MB, f"ONNX file is {size_mb:.1f} MB (>= {MAX_FILE_SIZE_MB} MB cap)"


def test_no_dynamic_branches(proto: onnx.ModelProto) -> None:
    """No If/Loop means no Python branch leaked into the traced graph."""
    bad = [n.op_type for n in proto.graph.node if n.op_type in {"If", "Loop", "Scan"}]
    assert not bad, f"Dynamic control-flow nodes found in graph: {bad}"


def test_opset_17(proto: onnx.ModelProto) -> None:
    default_opset = [op.version for op in proto.opset_import if op.domain in ("", "ai.onnx")]
    assert 17 in default_opset, f"expected opset 17, got {default_opset}"


def test_input_output_spec(proto: onnx.ModelProto) -> None:
    assert len(proto.graph.input) == 1
    assert len(proto.graph.output) == 1

    inp = proto.graph.input[0]
    out = proto.graph.output[0]
    assert inp.name == "input", f"input name = {inp.name!r}"
    assert out.name == "output", f"output name = {out.name!r}"

    def shape(value_info):
        return [
            d.dim_param if d.HasField("dim_param") else d.dim_value
            for d in value_info.type.tensor_type.shape.dim
        ]

    assert shape(inp) == ["N", 3, 2048, 1664], shape(inp)
    assert shape(out) == ["N", 512], shape(out)

    # fp32 dtype (ONNX enum 1).
    assert inp.type.tensor_type.elem_type == 1
    assert out.type.tensor_type.elem_type == 1


# ---------------------------------------------------------------------------
# Numerical parity tests (pydicom + dcmtk).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("decode", ["pydicom", "dcmtk"])
def test_parity_against_fixture(session: ort.InferenceSession, decode: str) -> None:
    stack, gold = _load_stack(decode)
    out = session.run(None, {"input": stack})[0]
    assert out.shape == (4, 512), out.shape
    assert out.dtype == np.float32, out.dtype
    max_abs = float(np.abs(out - gold).max())
    np.testing.assert_allclose(
        out,
        gold,
        atol=ATOL_ORT,
        rtol=RTOL,
        err_msg=(
            f"[{decode}] onnxruntime encoder output diverges from image_encoder_out{'_dcmtk' if decode=='dcmtk' else ''}.npy "
            f"(max abs diff {max_abs:.3e}, tolerance {ATOL_ORT:.0e})"
        ),
    )


def test_dynamic_batch_axis(session: ort.InferenceSession) -> None:
    """Run N=1 and N=2, confirm outputs are bit-exact slices of the N=4 output."""
    stack, _ = _load_stack("pydicom")
    full = session.run(None, {"input": stack})[0]

    one = session.run(None, {"input": stack[:1]})[0]
    assert one.shape == (1, 512)
    np.testing.assert_array_equal(one, full[:1])

    two = session.run(None, {"input": stack[:2]})[0]
    assert two.shape == (2, 512)
    np.testing.assert_array_equal(two, full[:2])
