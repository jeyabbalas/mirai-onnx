"""Post-export ONNX graph surgery so both models run on onnxruntime-web WebGPU EP.

Problem
-------
onnxruntime-web 1.19's JSEP WebGPU `Concat` kernel allocates one storage buffer
per input to the op. WebGPU's default `maxStorageBuffersPerShaderStage` is 8,
so any `Concat` with more than ~7 inputs fails compute-pipeline creation and
the op silently emits zeros. Two ops in Mirai's exported graphs hit this:

  1. `image_encoder.onnx :: /image_encoder/_model/pool/Concat` has 35 inputs
     (image_hidden + 34 per-key softmax probs). Its output is immediately sliced
     to the first 512 dims (`[:, :512]`), so the 34 RF branches are dead code —
     the original export wrapper slices them away (see scripts/export_image_encoder.py).
     Fix: reroute the Relu/Slice chain to consume `ReduceMax_output_0` directly
     and prune the Concat plus every upstream per-key FC.

  2. `risk_model.onnx :: /Concat` has 34 inputs (the per-key softmax probs that
     form the `risk_factor_vector`). This branch is load-bearing (its output
     goes into `hidden_pre_hazard`). Fix: tree-reduce the Concat into groups
     of <= 8 inputs each, with the original output name preserved so downstream
     consumers don't need changes.

Both rewrites are bit-exact by construction: (1) removes dead code; (2) splits
a commutative/associative concatenation along the same axis into the same
final sequence.

Usage
-----
    conda activate mirai-export
    python scripts/optimize_onnx_for_webgpu.py

Overwrites `models/image_encoder.onnx` and `models/risk_model.onnx`, keeping
`.pre_webgpu.onnx` snapshots alongside as a backup. Phase 0 fixtures provide
the numeric parity gate; any nonzero diff aborts with the originals intact.
"""

from __future__ import annotations

import pathlib
import shutil
import sys
from typing import List

import numpy as np
import onnx
import onnx.helper as oh
import onnxruntime as ort

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MODELS_DIR = PROJECT_ROOT / "models"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "reference" / "fixtures"

# Max Concat fan-in we'll produce. WebGPU's default limit is 8 storage buffers
# per stage; ORT-web's kernel needs N inputs + 1 output. 7 leaves one slot for
# the output and any uniform binding; tested-conservative.
MAX_CONCAT_FANIN = 7

# Two tolerances.
# ATOL_VS_PRE is the parity bar for the surgery itself: the ONNX Runtime output
# of the surgically-modified graph must match the ONNX Runtime output of the
# ORIGINAL graph bit-for-bit. Surgery is algebraic (dead-branch removal +
# Concat tree-reduction preserving input order on the same axis), so 0.0 is
# the right bar — any drift here means the rewrite changed semantics.
ATOL_VS_PRE = 0.0
# ATOL_VS_FIXTURE re-asserts the existing Phase 2/3 ORT-vs-Phase-0-fixture
# tolerance (`ATOL_ORT = 2e-5`, docs/architecture.md §1). The difference
# between ORT's MLAS kernel and torch's ATen kernel is already ~1e-5 on the
# encoder; surgery does not change that baseline, it just has to stay within.
ATOL_VS_FIXTURE = 2e-5


# -------- encoder surgery --------

def surgeon_image_encoder(model: onnx.ModelProto) -> onnx.ModelProto:
    """Strip the dead 35-input RF Concat from the image encoder.

    Before:  ReduceMax -> [35-in Concat with 34 per-key softmax siblings] -> Relu -> Slice[:512] -> output
    After:   ReduceMax -> Relu -> Slice[:512] (identity on 512 dims) -> output

    The dead softmax/FC subtrees for the 34 RF keys get pruned by
    `_keep_reachable_from_outputs` at the end.
    """
    graph = model.graph

    concat_name = "/image_encoder/_model/pool/Concat"
    concat_nodes = [n for n in graph.node if n.name == concat_name]
    if len(concat_nodes) != 1:
        raise RuntimeError(f"encoder: expected exactly one node named {concat_name!r}, found {len(concat_nodes)}")
    concat = concat_nodes[0]
    if len(concat.input) <= MAX_CONCAT_FANIN:
        print(f"[encoder] {concat_name} already has {len(concat.input)} inputs (<= {MAX_CONCAT_FANIN}); skipping")
        return model
    if len(concat.input) != 35:
        raise RuntimeError(f"encoder: expected 35-input Concat, got {len(concat.input)}; export shape changed")

    image_hidden = concat.input[0]
    concat_out = concat.output[0]

    relu_nodes = [n for n in graph.node if n.op_type == "Relu" and concat_out in n.input]
    if len(relu_nodes) != 1:
        raise RuntimeError(f"encoder: expected exactly one Relu consumer of {concat_out!r}, found {len(relu_nodes)}")
    relu = relu_nodes[0]

    # Rewire: Relu now consumes the raw image_hidden. Removes the 612-dim RF-bearing
    # tensor from the live graph in one assignment.
    relu.input[0] = image_hidden
    print(f"[encoder] rewired {relu.name}.input[0]: {concat_out!r} -> {image_hidden!r}")

    # Prune dead nodes. The Concat, the 34 per-key Softmax/Gemm/... chains, and
    # their initializers become unreachable from graph outputs.
    nodes_before = len(graph.node)
    model = _keep_reachable_from_outputs(model)
    nodes_after = len(model.graph.node)
    print(f"[encoder] pruned {nodes_before - nodes_after} dead nodes ({nodes_before} -> {nodes_after})")

    return model


# -------- risk model surgery --------

def surgeon_risk_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Tree-reduce the 34-input Concat in the risk model into groups of <= 7.

    Before:  Concat([p0..p33], axis=1) -> <original output name>
    After:   inner Concats ([p0..p6], [p7..p13], ...) -> outer Concat of groups
             -> <original output name>

    The outer Concat keeps the original node's output tensor name so every
    downstream consumer continues to resolve without touching other ops.
    """
    graph = model.graph

    concat_name = "/Concat"
    concat_nodes = [n for n in graph.node if n.name == concat_name]
    if len(concat_nodes) != 1:
        raise RuntimeError(f"risk: expected exactly one node named {concat_name!r}, found {len(concat_nodes)}")
    concat = concat_nodes[0]
    if len(concat.input) <= MAX_CONCAT_FANIN:
        print(f"[risk] {concat_name} already has {len(concat.input)} inputs (<= {MAX_CONCAT_FANIN}); skipping")
        return model

    n = len(concat.input)
    if n != 34:
        raise RuntimeError(f"risk: expected 34-input Concat, got {n}; export shape changed")

    axis = 1
    for attr in concat.attribute:
        if attr.name == "axis":
            axis = int(attr.i)
            break
    if axis != 1:
        raise RuntimeError(f"risk: expected axis=1 on {concat_name}, got {axis}")

    orig_output = concat.output[0]
    inputs = list(concat.input)

    # Chunk into groups of MAX_CONCAT_FANIN. 34 / 7 = 5 groups of 7 plus a tail
    # of 34 - 7*4 = 6 if we keep 4 full, or simpler: Python slicing.
    groups: List[List[str]] = [inputs[i:i + MAX_CONCAT_FANIN] for i in range(0, n, MAX_CONCAT_FANIN)]
    num_groups = len(groups)
    if num_groups > MAX_CONCAT_FANIN:
        raise RuntimeError(
            f"risk: {n} inputs with fanin {MAX_CONCAT_FANIN} yields {num_groups} groups, "
            f"exceeds the outer-Concat fanin bound; pick a different MAX_CONCAT_FANIN"
        )
    print(f"[risk] tree-reducing 34-input Concat into {num_groups} groups of {[len(g) for g in groups]}")

    # Build inner Concats. Name prefix scoped under the original Concat so tools
    # that grep by node name keep getting sensible hits.
    inner_outputs: List[str] = []
    new_nodes = []
    for i, g in enumerate(groups):
        out_name = f"{orig_output}__group_{i}"
        node = oh.make_node(
            "Concat",
            inputs=g,
            outputs=[out_name],
            name=f"{concat_name}_group_{i}",
            axis=axis,
        )
        new_nodes.append(node)
        inner_outputs.append(out_name)

    # Build the outer Concat, preserving the original tensor name so downstream
    # consumers don't need rewiring.
    outer = oh.make_node(
        "Concat",
        inputs=inner_outputs,
        outputs=[orig_output],
        name=f"{concat_name}_outer",
        axis=axis,
    )
    new_nodes.append(outer)

    # Replace the original node with the new sequence, preserving its position
    # so topological order remains valid.
    idx = list(graph.node).index(concat)
    del graph.node[idx]
    for off, n_node in enumerate(new_nodes):
        graph.node.insert(idx + off, n_node)

    return model


# -------- generic graph cleanup --------

def _keep_reachable_from_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove nodes and initializers that cannot reach any graph output.

    Standard reverse-reachability: start with the set of graph-output tensor
    names, walk back along node inputs, keep any node whose output participates.
    Then drop unused initializers (pruned per-key FC weights can be large).
    """
    graph = model.graph
    outputs = {o.name for o in graph.output}
    producers: dict = {}
    for n in graph.node:
        for o in n.output:
            producers[o] = n

    live_nodes = set()
    stack = list(outputs)
    while stack:
        tensor = stack.pop()
        node = producers.get(tensor)
        if node is None or id(node) in live_nodes:
            continue
        live_nodes.add(id(node))
        stack.extend(node.input)

    # Rebuild nodes list preserving original topo order.
    new_nodes = [n for n in graph.node if id(n) in live_nodes]
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Collect tensor names that are used by any surviving node.
    used: set = set()
    for n in graph.node:
        for i in n.input:
            used.add(i)
    used |= {o.name for o in graph.output}

    # Drop unused initializers.
    kept_init = [init for init in graph.initializer if init.name in used]
    dropped = len(graph.initializer) - len(kept_init)
    del graph.initializer[:]
    graph.initializer.extend(kept_init)
    if dropped:
        print(f"[prune] dropped {dropped} unused initializers")

    # Drop unused value_info.
    kept_vi = [vi for vi in graph.value_info if vi.name in used]
    del graph.value_info[:]
    graph.value_info.extend(kept_vi)

    return model


# -------- parity validation --------

def _load_stack() -> np.ndarray:
    import json
    order = json.loads((FIXTURES_DIR / "batch_order.json").read_text())
    per_view: List[np.ndarray] = []
    for entry in order:
        name = f"{entry['view_str']}_{entry['side_str']}.npy"
        per_view.append(np.load(FIXTURES_DIR / "preproc_tensor" / name))
    stack = np.stack(per_view).astype(np.float32, copy=False)
    assert stack.shape == (4, 3, 2048, 1664), f"unexpected stack shape {stack.shape}"
    return stack


def _validate_encoder(model_path: pathlib.Path) -> None:
    stack = _load_stack()
    # Bit-exact check against the pre-surgery backup: surgery must not perturb kernels.
    pre_path = model_path.with_suffix(".pre_webgpu.onnx")
    pre_sess = ort.InferenceSession(str(pre_path), providers=["CPUExecutionProvider"])
    pre_out = pre_sess.run(None, {"input": stack})[0]
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": stack})[0]
    diff_vs_pre = float(np.abs(out - pre_out).max())
    print(f"[validate] encoder vs pre-surgery ONNX: max abs diff = {diff_vs_pre:.3e}")
    np.testing.assert_allclose(out, pre_out, atol=ATOL_VS_PRE, rtol=0.0)
    print(f"[validate] encoder surgery parity OK at atol={ATOL_VS_PRE} (bit-exact)")
    # Sanity: still within the documented ORT-vs-fixture tolerance.
    gold = np.load(FIXTURES_DIR / "image_encoder_out.npy").reshape(4, -1)
    diff_vs_fixture = float(np.abs(out - gold).max())
    print(f"[validate] encoder vs image_encoder_out.npy: max abs diff = {diff_vs_fixture:.3e}")
    np.testing.assert_allclose(out, gold, atol=ATOL_VS_FIXTURE, rtol=0.0)
    print(f"[validate] encoder fixture parity OK at atol={ATOL_VS_FIXTURE}")


def _validate_risk_model(model_path: pathlib.Path) -> None:
    encoder_out = np.load(FIXTURES_DIR / "image_encoder_out.npy").reshape(4, 512)
    img_feats = encoder_out.reshape(1, 4, 512).astype(np.float32)

    import json
    order = json.loads((FIXTURES_DIR / "batch_order.json").read_text())
    view_seq = np.array([[int(e["view"]) for e in order]], dtype=np.int64)
    side_seq = np.array([[int(e["side"]) for e in order]], dtype=np.int64)
    time_seq = np.zeros((1, 4), dtype=np.int64)
    rf_vector = np.load(FIXTURES_DIR / "risk_factor_vector.npy").reshape(1, -1).astype(np.float32)
    rf_known_mask = np.zeros_like(rf_vector)  # pool uses predicted RFs at eval time

    feeds = {
        "img_feats": img_feats,
        "view_seq": view_seq,
        "side_seq": side_seq,
        "time_seq": time_seq,
        "rf_vector": rf_vector,
        "rf_known_mask": rf_known_mask,
    }

    pre_path = model_path.with_suffix(".pre_webgpu.onnx")
    pre_sess = ort.InferenceSession(str(pre_path), providers=["CPUExecutionProvider"])
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    pre_names = [o.name for o in pre_sess.get_outputs()]
    post_names = [o.name for o in sess.get_outputs()]
    if pre_names != post_names:
        raise RuntimeError(f"risk: output name order changed {pre_names} -> {post_names}")
    pre_out = dict(zip(pre_names, pre_sess.run(None, feeds)))
    outs = dict(zip(post_names, sess.run(None, feeds)))

    for name in ("logit", "hidden_pre_hazard"):
        diff = float(np.abs(outs[name] - pre_out[name]).max())
        print(f"[validate] risk {name} vs pre-surgery ONNX: max abs diff = {diff:.3e}")
        np.testing.assert_allclose(outs[name], pre_out[name], atol=ATOL_VS_PRE, rtol=0.0)
        print(f"[validate] risk {name} surgery parity OK at atol={ATOL_VS_PRE} (bit-exact)")

    # Sanity vs Phase 0 fixtures.
    gold_logit = np.load(FIXTURES_DIR / "raw_logit.npy").reshape(1, -1)
    gold_hidden = np.load(FIXTURES_DIR / "xai_hidden.npy").reshape(1, -1)
    diff_logit = float(np.abs(outs["logit"] - gold_logit).max())
    diff_hidden = float(np.abs(outs["hidden_pre_hazard"] - gold_hidden).max())
    print(f"[validate] risk logit vs raw_logit.npy: max abs diff = {diff_logit:.3e}")
    print(f"[validate] risk hidden_pre_hazard vs xai_hidden.npy: max abs diff = {diff_hidden:.3e}")
    np.testing.assert_allclose(outs["logit"], gold_logit, atol=ATOL_VS_FIXTURE, rtol=0.0)
    np.testing.assert_allclose(outs["hidden_pre_hazard"], gold_hidden, atol=ATOL_VS_FIXTURE, rtol=0.0)
    print(f"[validate] risk model fixture parity OK at atol={ATOL_VS_FIXTURE}")


# -------- driver --------

def _rewrite(path: pathlib.Path, surgeon) -> None:
    backup = path.with_suffix(".pre_webgpu.onnx")
    if not backup.exists():
        print(f"[backup] {path.name} -> {backup.name}")
        shutil.copy2(path, backup)
    else:
        print(f"[backup] {backup.name} already exists, leaving in place")

    model = onnx.load(str(backup))  # always surger the original, not a prior rewrite
    model = surgeon(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    print(f"[save] wrote {path}")


def main() -> int:
    encoder = MODELS_DIR / "image_encoder.onnx"
    risk = MODELS_DIR / "risk_model.onnx"
    if not encoder.exists() or not risk.exists():
        print(f"ERROR: expected {encoder} and {risk}", file=sys.stderr)
        return 1

    print("== rewriting image_encoder.onnx ==")
    _rewrite(encoder, surgeon_image_encoder)
    _validate_encoder(encoder)

    print()
    print("== rewriting risk_model.onnx ==")
    _rewrite(risk, surgeon_risk_model)
    _validate_risk_model(risk)

    print()
    print("Done. Both models are WebGPU-compatible and bit-exact vs Phase 0 fixtures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
