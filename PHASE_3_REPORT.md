# Phase 3 Report — Risk Model ONNX Export

**Date:** 2026-04-23
**Branch:** main
**Status:** Complete. `models/risk_model.onnx` exported and validated; 13/13 risk-model tests green; Phase 0 baseline (56/56) and Phase 2 image-encoder tests (8/8) still green.

## Deliverables

| File | Purpose |
|---|---|
| `models/risk_model.onnx` | The risk-model ONNX graph (7.65 MB). |
| `scripts/export_risk_model.py` | Export script (loads MiraiFull, holds `RiskModelExport(nn.Module)` inline, runs `torch.onnx.export`, validates against Phase 0 fixtures). |
| `tests/onnx/test_risk_model.py` | 13 pytest cases: structural, parity (parametrized over pydicom + dcmtk), and behavioral (rf_known_mask blend semantics). |
| `tests/onnx/conftest.py` | One added line: `RISK_MODEL_ONNX = MODELS_DIR / "risk_model.onnx"`. |

No edits under `external/Mirai/` or `tests/reference/fixtures/`. The original migration plan §4 named `onconet/models/mirai_full_export.py` as the wrapper location; that path lives under the read-only upstream tree, so the wrapper class lives inline in `scripts/export_risk_model.py` (mirroring `scripts/export_image_encoder.py`'s `ImageEncoderExport` pattern).

## ONNX graph spec

Inputs:

| Name | Shape | Dtype |
|---|---|---|
| `img_feats` | `(B, 4, 512)` | fp32 |
| `view_seq` | `(B, 4)` | int64 |
| `side_seq` | `(B, 4)` | int64 |
| `time_seq` | `(B, 4)` | int64 |
| `rf_vector` | `(B, 100)` | fp32 |
| `rf_known_mask` | `(B, 100)` | fp32 |

Outputs:

| Name | Shape | Dtype |
|---|---|---|
| `logit` | `(B, 5)` | fp32 |
| `hidden_pre_hazard` | `(B, 612)` | fp32 — post-ReLU XAI embedding |

Op-count snapshot (209 nodes total, opset `ai.onnx 17`):

```
Gemm                     36   # 34 per-key *_fc + hazard_fc + base_hazard_fc
MatMul                   12   # transformer attention (3 q/k/v + 1 aggregate per layer × 2 layers + scores) + projection_layer
Constant                 36
Sigmoid                  20   # 20 binary RF keys
Softmax                  16   # 14 multi-class RF keys + 2 attention softmaxes (one per transformer layer)
Add                      15
Unsqueeze                15
Gather                   10   # view/side/time embedding lookups + transformer-layer module access
Concat                    9
Shape                     8
Transpose                 7
Mul                       6
Reshape                   6
Relu                      3
LayerNormalization        2
ReduceSum                 2
Div, Sub, ConstantOfShape, Equal, Where, Expand   each 1
```

100 initializers, including `prob_of_failure_layer.upper_triagular_mask` (dims `[5, 5]`, fp32). No `If`, `Loop`, or `Scan` nodes — the tensor-level `rf_known_mask` blend keeps every runtime mode (all-user / partial / all-predicted) inside a single static graph.

Dynamic axis: the batch dim `B` is dynamic on every input and output. All other dims are static.

## Parity measurements

Two-track scheme (CLAUDE.md "Tolerances"). All measurements at `RTOL=0`. Phase 0 fixtures captured under `mirai-py38` (torch 1.9.0). Export and ORT runs under `mirai-export` (torch 2.2.2, onnx 1.17, onnxruntime 1.19.2, CPUExecutionProvider).

| Decode path | Output | PyTorch wrapper max abs diff | ONNX Runtime max abs diff |
|---|---|---|---|
| pydicom | `logit` | 2.384e-07 | 8.941e-07 |
| pydicom | `hidden_pre_hazard` | 2.384e-07 | 5.960e-07 |
| dcmtk | `logit` | (test passes; ORT bound below covers it) | within 2e-5 |
| dcmtk | `hidden_pre_hazard` | (test passes; ORT bound below covers it) | within 2e-5 |

Pinned tolerances:

- `ATOL_TORCH = 1e-6` (deviation from Phase 2's `0.0` — see "Deviation" below).
- `ATOL_ORT = 2e-5` (matches Phase 2's empirical bound; risk model's actual ORT diffs are sub-microvolt, well inside the bound).

## Deviation: PyTorch wrapper tolerance is 1e-6, not 0.0

Phase 2's `ATOL_TORCH = 0.0` did not survive into Phase 3. Observed max abs diff between the PyTorch wrapper and Phase 0 fixtures is **2.384e-07** for both `logit` and `hidden_pre_hazard` — exactly one fp32 ULP at the value magnitudes in play (hidden activations near 0.7-1.0; ULP ≈ value × 2⁻²² ≈ 1.7e-07). The wrapper code is correct; the divergence is intrinsic torch-1.9 (fixture capture, `mirai-py38`) vs torch-2.2 (exporter, `mirai-export`) ATen kernel-reduction-order rounding.

Why Phase 2 escaped this: the image-encoder graph is Conv + BN + ReLU + GlobalMaxPool, whose CPU kernels happen to be byte-deterministic across the two torch versions on this machine. The risk-model graph is attention (`MatMul` chains, `Softmax`), `LayerNormalization`, `Linear` — these have minor cross-version differences in fp32 accumulation order.

This is not loosenable below 1 ULP without re-capturing Phase 0 under torch 2.2 (which would invalidate the cross-version reproducibility property the fixtures aim for). Pinning `ATOL_TORCH = 1e-6` matches `tests/reference/conftest.py:ATOL_FP32 = 1e-6` (the same fp32 floor Phase 0 itself uses on its internal cross-checks), with ~4× headroom over the observed maximum. `docs/architecture.md` §1's tolerance table should be updated to reflect this in a follow-up.

## Wrapper design — what was reimplemented and why

The export wrapper (`RiskModelExport` in `scripts/export_risk_model.py:74`) reimplements just the parts of the upstream forward path that need to change for export. All weights stay shared with the loaded `MiraiFull` instance — no re-instantiation, no weight surgery.

Three upstream branches needed replacement:

1. **`AllImageTransformer.mask_input`** (`hiddens_transfomer.py:55-66`). Skipped entirely. In eval mode `mask_prob=0` ⇒ `is_mask=0` ⇒ `is_kept=1` ⇒ `mask_embedding(ones)` returns the row at `padding_idx=1` (zero) ⇒ `masked_x = x*1 + 0 = x`. The bernoulli node would otherwise leak into the trace.

2. **The try/except at `hiddens_transfomer.py:97-106`** invokes `pool.get_pred_rf_loss(hidden, risk_factors=None)`, which silently no-ops via the bare `except` — but along the way calls every per-key `*_fc` a **second** time with the post-ReLU hidden. Phase 0 defends against this in `tests/reference/_hooks.py:make_hook` with a `if k not in captured` first-call guard. The wrapper just doesn't make the second call.

3. **`RiskFactorPool.forward`** (`risk_factor_pool.py:54-59`) uses `if not self.training and use_pred_risk_factors_if_unk` and `np.random.random()` to choose between user-supplied vs model-predicted RFs. The wrapper replaces this with a tensor-level blend:

   ```python
   rf_used = rf_known_mask * rf_vector + (1.0 - rf_known_mask) * rf_predicted
   ```

   so the same ONNX graph serves all three runtime modes (all-user, partial, all-predicted) without an `If` node. Always running all 34 per-key FCs is cheap (~34 small Linears) and avoids graph-level branching.

Two minor cleanups:

- **Out-of-place `F.relu`** instead of upstream's `nn.ReLU(inplace=True)` (`hiddens_transfomer.py:121`) — cleaner trace; no storage-aliasing concerns.
- **Skip the dropouts** (`AllImageTransformer.dropout`, `RiskFactorPool.dropout`, `MultiHead_Attention.dropout`). They are identity in eval mode anyway; bypassing makes the graph smaller. The trained snapshot's effective dropout is empirically zero (Phase 0's cross-check `xai_hidden == relu(pool_hidden)` would fail otherwise — see `tests/reference/test_baseline.py::test_xai_equals_relu_pool_hidden`).

## Tests

`tests/onnx/test_risk_model.py` — 13 cases:

Structural (5):
- `test_checker_passes` — `onnx.checker.check_model` succeeds.
- `test_file_size_under_50mb` — actual file size 7.65 MB.
- `test_no_dynamic_branches` — no `If`/`Loop`/`Scan` in `proto.graph.node`.
- `test_opset_17` — opset 17 in `proto.opset_import`.
- `test_input_output_spec` — exact match on input/output names, shapes (after `onnx.shape_inference.infer_shapes` to resolve the symbolic post-Add dim on `logit`), and dtypes.

Parity, parametrized over `["pydicom", "dcmtk"]` (4 cases):
- `test_parity_logit[pydicom|dcmtk]` — ORT `logit` vs `raw_logit{,_dcmtk}.npy` at `atol=ATOL_ORT=2e-5`.
- `test_parity_hidden[pydicom|dcmtk]` — ORT `hidden_pre_hazard` vs `xai_hidden{,_dcmtk}.npy` at `atol=ATOL_ORT=2e-5`.

Behavioral (4):
- `test_dynamic_batch_axis` — at `B=2` (duplicated input), outputs are bit-exact slice copies of `B=1`.
- `test_zero_mask_invokes_predicted_rfs` — `rf_vector=0, rf_known_mask=0` ⇒ `hidden_pre_hazard[:, -100:] == relu(risk_factor_vector.npy)` at `atol=ATOL_ORT`. Transitively validates the per-key FCs because Phase 0 already proved the per-key probs concatenate to `risk_factor_vector` (`test_baseline.py::test_per_key_concat_equals_risk_factor_vector`).
- `test_image_hidden_matches_phase0` — same input ⇒ `hidden_pre_hazard[:, :512] == relu(image_hidden_in_pool.npy)` at `atol=ATOL_ORT`.
- `test_user_supplied_rfs_pass_through` — `rf_known_mask=ones`, `rf_vector=linspace(-0.5, 1.0, 100)` ⇒ `hidden_pre_hazard[:, -100:] == relu(rf_vector)` **bit-exact**. Proves the blend isn't fused/folded away (the 1×rf + 0×rf_predicted path is exact in fp32 since 0×anything=0 and 1×x=x).

All 13 pass. Run summary:

```
$ /opt/homebrew/Caskroom/miniforge/base/envs/mirai-export/bin/python -m pytest tests/onnx/ -v
============================== 21 passed in 8.68s ==============================
```

(13 risk-model + 8 image-encoder.)

Phase 0 still green:

```
$ /opt/homebrew/Caskroom/miniforge/base/envs/mirai-py38/bin/python -m pytest tests/reference/test_baseline.py -v
======================= 56 passed, 3 warnings in 50.81s ========================
```

## Pinned facts (newly verified at runtime under torch 2.2)

- `model.transformer.pool.length_risk_factor_vector == 100` (asserted in the export script and the wrapper's `__init__`).
- `model.image_repr_dim == 512`.
- The 34 per-key FCs are present in the ONNX graph as 34 of the 36 `Gemm` nodes (the other two are `prob_of_failure_layer.hazard_fc` 612→5 and `prob_of_failure_layer.base_hazard_fc` 612→1).
- `prob_of_failure_layer.upper_triagular_mask` is serialized as a `(5, 5)` fp32 initializer.
- `mask_input` is identity in eval mode (proven by inspection; verified empirically by the bit-exact behavior of `test_user_supplied_rfs_pass_through`).

## Out-of-scope notes for downstream phases

- **`padding_idx=-1` on time/view/side embeddings** (`hiddens_transfomer.py:145-147`). At single-exam inference the ONNX `Gather` is fed only `{0, 1}`, so PyTorch's `padding_idx` zeroing semantics are never exercised — graph behaves identically. Future callers feeding `view_seq=2` (PAD) will get whatever `view_embed.weight[2]` holds, not a zero — unlike PyTorch's `padding_idx` behavior (which only zeroes the row at construction time anyway, so this is effectively the same). Not a problem for Phase 4-9.
- The cross-version torch ULP drift documented above is an honest fp32 floor. Phase 4 (calibrator) operates on outputs in fp64 and so won't compound this. Phase 5 (Python-side end-to-end ONNX pipeline) should expect total drift on the order of `1e-5` from the chained `image_encoder.onnx + risk_model.onnx + calibrator` graph, well within the plan's `1e-4` end-to-end bound.
