# Mirai Architecture for ONNX Export

**Phase 1 deliverable.** Status: complete 2026-04-22.

This document is the single source of truth for Phase 2/3 (ONNX export of the image encoder and risk model) and Phase 4 (calibrator extraction). Every shape, every control-flow branch, every export decision — already decided — is recorded here. Downstream phases should read this document end-to-end before touching any export code. The committed strings in `## 9` and `## 12` are asserted verbatim by `tests/architecture/test_plan.py`.

---

## 1. Scope and Audience

This document covers the export of three artefacts:

- `image_encoder.onnx` — per-view image feature extraction (Phase 2).
- `risk_model.onnx` — four-view transformer + pool + hazard head, with the XAI embedding as a named output (Phase 3).
- `calibrator.json` — the 5-year Platt-scaling parameters, language-neutral (Phase 4).

Out of scope: TypeScript preprocessing (Phase 6), TypeScript risk-factor vectorizer (Phase 7), browser wiring (Phase 8), Docker cross-machine reference (deferred).

Audience: the executor agent (or engineer) implementing Phase 2/3/4, plus the reviewer.

**Two-track tolerances.** Phase 2 (2026-04-23) established empirically that torch → ONNX → ORT cannot reproduce the Phase 0 fixtures at `ATOL_FP32 = 1e-6`. The PyTorch wrapper forward (same torch graph that captured Phase 0) remains bit-exact, but ORT's MLAS kernels round differently from PyTorch's ATen kernels by up to ~1 ULP on fp32 features. This is independent of `do_constant_folding` and every `ort.GraphOptimizationLevel` setting. Tolerances are therefore split:

| Validation stage | Tolerance | Rationale |
|---|---|---|
| Phase 0 pytorch-internal tensors (pool_hidden, raw_logit, etc.) | `ATOL_FP32 = 1e-6`, `ATOL_FP64 = 1e-9`, `RTOL = 0` | Same torch version, same machine; bit-exact reachable. |
| Phase 2/3 **PyTorch export-wrapper** forward vs Phase 0 fixtures | `ATOL_TORCH = 0.0` (exact) | Wrapper is a thin view/slice over the same torch graph. Any deviation is a wrapper bug. |
| Phase 2/3 **ONNX Runtime (CPU)** session vs Phase 0 fixtures | `ATOL_ORT = 2e-5`, `RTOL = 0` | ≈2× worst observed (9.5e-6 pydicom, 1.05e-5 dcmtk single-outlier on the image encoder). ≤4e-6 relative on features ≤5 magnitude — orders of magnitude below the 4-decimal rounding floor of final predictions. |

Do not loosen these tolerances further without a new empirical measurement and a note in the relevant phase report. See `PHASE_2_REPORT.md` § "Parity — caveat worth flagging" for the measurements that produced the ORT bound.

---

## 2. Source Authority and Provenance

All exports target the exact upstream artefacts captured in `tests/reference/fixtures/MANIFEST.json`. Any divergence is a regression.

| Key | Value | Source |
|---|---|---|
| Mirai git SHA | `4af944449863966a5a9c66b44e56e3c141223897` | `MANIFEST.env.git.mirai_sha` |
| onconet version | `0.14.1` | `MANIFEST.runs.*.modelVersion` |
| torch version (capture) | `1.9.0` | `MANIFEST.env.torch_version` |
| Config file | `external/Mirai/onconet/configs/mirai_trained.json` (sha256 `32fdedfa...`) | `MANIFEST.config.config_sha256` |
| Image encoder snapshot | sha256 `e66261f434474686462e5c7035de37452f47e4e7cc8831086be19397383e8af3`, 48.3 MB | `MANIFEST.snapshots.img_encoder_snapshot` |
| Transformer snapshot | sha256 `cb4ef50459d747d38edfb846b3cd29869ccad9b7e291eb5f2fe66559d914e610`, 116.0 MB | `MANIFEST.snapshots.transformer_snapshot` |
| Calibrator | sha256 `822092d81272c97883d54a4bde0bc1cdcebadb861dea8467cb93201fedb73efa`, 865 bytes | `MANIFEST.snapshots.calibrator_path` |

Pinned demo predictions (pydicom decode path) — Phase 2/3/4 must reproduce these to 4 decimal places:

| Year | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Prediction | 0.0314 | 0.0505 | 0.0711 | 0.0935 | 0.1052 |

dcmtk decode path (for differential debugging only; the browser target is pydicom):

| Year | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Prediction | 0.0298 | 0.0483 | 0.0684 | 0.09 | 0.1016 |

---

## 3. End-to-End Shape Table

Every tensor from DICOM uint16 through calibrated probabilities. Shapes are exact; dtypes are exact. `file:line` refers to the upstream Mirai code at the pinned SHA. `Fixture` refers to files under `tests/reference/fixtures/` (pydicom path; dcmtk path is suffix `_dcmtk`).

| # | Stage | Tensor | Shape | Dtype | Source (file:line) | Phase 0 fixture |
|---|---|---|---|---|---|---|
| 1 | DICOM decode (per view) | raw uint16 | `(3062, 2394)` | uint16 | `onconet/utils/dicom.py:dicom_to_arr` | `dicom_raw_uint16/{CC,MLO}_{L,R}.npy` |
| 2 | Scale + align-to-left + normalize (per view) | preprocessed tensor | `(3, 2048, 1664)` | float32 | `onconet/transformers/image.py:Scale_2d` + `AlignToLeft` + `Normalize_2d` | `preproc_tensor/{CC,MLO}_{L,R}.npy` |
| 3 | Batch collate | `batch['x']` | `(1, 3, 4, 2048, 1664)` | float32 | `external/Mirai/onconet/models/mirai_full.py:194-196` | `MANIFEST.runs.pydicom.shapes.preproc_batch_x` |
| 4 | Batch collate | `batch['view_seq']`, `batch['side_seq']`, `batch['time_seq']` | `(1, 4)` | int64 | `external/Mirai/onconet/models/mirai_full.py:190-192` | `batch_order.json` |
| 5 | MiraiFull.forward entry | `x` after `.transpose(1,2).view` | `(4, 3, 2048, 1664)` | float32 | `external/Mirai/onconet/models/mirai_full.py:55` | — |
| 6 | Image encoder raw hidden | `hidden` from `ResNet.forward` | `(4, 612)` | float32 | `external/Mirai/onconet/models/resnet_base.py:244` | — (pre-slice, not captured) |
| 7 | Image encoder sliced to `img_repr_dim` | `img_x` after slice | `(1, 4, 512)` | float32 | `external/Mirai/onconet/models/mirai_full.py:58-59` | `image_encoder_out.npy` |
| 8 | Transformer input | `img_x` | `(1, 4, 512)` | float32 | `external/Mirai/onconet/models/hiddens_transfomer.py:78` | `image_encoder_out.npy` |
| 9 | After `projection_layer` | `masked_x` | `(1, 4, 512)` | float32 | `external/Mirai/onconet/models/hiddens_transfomer.py:90` | — |
| 10 | After self-attention stack | `transformer_hidden` | `(1, 4, 512)` | float32 | `external/Mirai/onconet/models/hiddens_transfomer.py:91` | — |
| 11 | Reshaped for pool | `img_like_hidden` | `(1, 512, 4, 1)` | float32 | `external/Mirai/onconet/models/hiddens_transfomer.py:93` | — |
| 12 | `RiskFactorPool.internal_pool` output | `image_hidden` | `(1, 512)` | float32 | `external/Mirai/onconet/models/pools/risk_factor_pool.py:40` | `image_hidden_in_pool.npy` |
| 13 | Per-key `*_fc` output (×34) | `key_logit` | `(1, num_key_features)` where `num_key_features ∈ {1,3,4,5,6,7,13}` | float32 | `external/Mirai/onconet/models/pools/risk_factor_pool.py:47` | `pred_risk_factors_per_key/<key>.npy` (post-sigmoid/softmax) |
| 14 | Concatenated predicted RF block | `risk_factors_hidden` | `(1, 100)` | float32 | `external/Mirai/onconet/models/pools/risk_factor_pool.py:67` | `risk_factor_vector.npy` |
| 15 | Pool hidden (pre-relu) | `hidden` returned from pool | `(1, 612)` | float32 | `external/Mirai/onconet/models/pools/risk_factor_pool.py:70-72` | `pool_hidden.npy` |
| 16 | XAI hidden (post-relu) | `hidden` after in-place `self.relu` | `(1, 612)` | float32 | `external/Mirai/onconet/models/hiddens_transfomer.py:121` | `xai_hidden.npy` |
| 17 | Raw cumulative logit (5-year) | `logit` from `Cumulative_Probability_Layer.forward` | `(1, 5)` | float32 | `external/Mirai/onconet/models/cumulative_probability_layer.py:32` | `raw_logit.npy` |
| 18 | Sigmoid of raw logit | `probs` | `(1, 5)` | float32 | `external/Mirai/onconet/models/mirai_full.py:155` | `raw_sigmoid.npy` |
| 19 | Calibrated per-year probabilities | `pred_y` | `(5,)` | **float64** | `external/Mirai/onconet/models/mirai_full.py:156-162` | `calibrated.npy` |

`rf_dim=100` is read at runtime from `model.transformer.pool.length_risk_factor_vector` (`tests/reference/_hooks.py:138`). Do not hardcode `rf_dim` in export wrappers — assert it equals 100 against `MANIFEST.runs.pydicom.shapes.rf_dim` and fail loudly on mismatch.

---

## 4. nn.Module Dependency Graph

Only inference-time modules are listed. Module paths are the Python-attribute chain from the loaded `MiraiFull` instance.

```
MiraiFull  (external/Mirai/onconet/models/mirai_full.py:30)
├── image_encoder : CustomResnet (a wrapper around ResNet)
│   └── _model : ResNet  (external/Mirai/onconet/models/resnet_base.py:13)
│       ├── downsampler : Downsampler  (resnet_base.py:275)
│       │   ├── conv1 : Conv2d(3 → inplanes, 7×7, stride=2)
│       │   ├── bn1   : BatchNorm2d
│       │   ├── relu  : ReLU(inplace=True)
│       │   └── maxpool : MaxPool2d(3×3, stride=2)
│       ├── layer1_{0..} : BasicBlocks (onconet.models.blocks.*)
│       ├── layer2_{0..} : BasicBlocks
│       ├── layer3_{0..} : BasicBlocks
│       ├── layer4_{0..} : BasicBlocks
│       ├── pool : RiskFactorPool  (onconet/models/pools/risk_factor_pool.py:14)
│       │   ├── internal_pool : GlobalMaxPool (onconet.models.pools.max_pool)
│       │   ├── dropout : Dropout(p=args.dropout)  — eval no-op
│       │   └── {key}_fc : Linear(hidden_dim → num_key_features) × 34 keys
│       ├── relu : ReLU(inplace=True)             — only when not pool.replaces_fc()
│       ├── dropout : Dropout                      — eval no-op
│       ├── fc : Linear(hidden_dim → num_classes)  — logit is discarded by MiraiFull branch
│       └── prob_of_failure_layer : Cumulative_Probability_Layer  — only if survival_analysis_setup; unused by MiraiFull branch
└── transformer : AllImageTransformer  (external/Mirai/onconet/models/hiddens_transfomer.py:17)
    ├── projection_layer : Linear(precomputed_hidden_dim → hidden_dim)
    ├── mask_embedding : Embedding(2, precomputed_hidden_dim, padding_idx=1)  — used only by mask_input
    ├── kept_images_vec : Parameter(shape (1, 4, 1), requires_grad=False)     — used only by mask_input
    ├── transformer : Transformer  (hiddens_transfomer.py:137)
    │   ├── time_embed : Embedding(MAX_TIME+1, 32, padding_idx=-1)
    │   ├── view_embed : Embedding(MAX_VIEWS+1, 32, padding_idx=-1)
    │   ├── side_embed : Embedding(MAX_SIDES+1, 32, padding_idx=-1)
    │   ├── embed_add_fc   : Linear(96 → hidden_dim)
    │   ├── embed_scale_fc : Linear(96 → hidden_dim)
    │   └── transformer_layer_{0..L-1} : TransformerLayer
    │       ├── multihead_attention : MultiHead_Attention
    │       │   ├── query : Linear(hidden_dim → hidden_dim)
    │       │   ├── key   : Linear(hidden_dim → hidden_dim)
    │       │   ├── value : Linear(hidden_dim → hidden_dim)
    │       │   ├── aggregate_fc : Linear(hidden_dim → hidden_dim)
    │       │   └── dropout : Dropout  — eval no-op
    │       ├── layernorm_attn : LayerNorm
    │       ├── fc1 : Linear(hidden_dim → hidden_dim)
    │       ├── relu : ReLU(inplace=False)  — TransformerLayer uses default ReLU (not inplace)
    │       ├── fc2 : Linear(hidden_dim → hidden_dim)
    │       └── layernorm_fc : LayerNorm
    ├── pred_masked_img_fc : Linear                 — unused at inference
    ├── pool : RiskFactorPool  (same class as encoder's pool, separate instance)
    │   ├── internal_pool : GlobalMaxPool
    │   ├── dropout : Dropout                        — eval no-op
    │   └── {key}_fc : Linear × 34                   — weights disjoint from encoder's pool
    ├── relu : ReLU(inplace=True)                   — only when not pool.replaces_fc()
    ├── dropout : Dropout                            — eval no-op
    ├── fc : Linear                                  — logit overridden by prob_of_failure_layer
    └── prob_of_failure_layer : Cumulative_Probability_Layer  (cumulative_probability_layer.py:7)
        ├── hazard_fc : Linear(hidden_dim → max_followup=5)
        ├── base_hazard_fc : Linear(hidden_dim → 1)
        ├── relu : ReLU(inplace=True)
        └── upper_triagular_mask : Parameter(shape (5,5), requires_grad=False)
```

Two important observations about the graph:

- **The image encoder's pool and the transformer's pool are BOTH `RiskFactorPool`** and each has its own 34 `{key}_fc` Linear layers. They are distinct submodules with disjoint weights. The encoder's pool output is `(B*N, 612)` → `MiraiFull.forward` slices to `(B*N, 512)` by dropping the rf block.
- **`prob_of_failure_layer` is attached under the transformer**, not at the top level. The path is `model.transformer.prob_of_failure_layer`.

---

## 5. Control-Flow Branches and Resolution at Export

Every inference-path branch that depends on `self.training`, `self.args.*`, or `risk_factors is not None`. "Live" = the side that executes with the pinned snapshot + `mirai_trained.json` config + Phase 0-verified module training modes (`MiraiFull` in train mode, snapshot-loaded children in eval mode via `torch.load` — see `## 6`).

| # | File:line | Condition | Live side | Why | Export strategy |
|---|---|---|---|---|---|
| B1 | `external/Mirai/onconet/models/mirai_full.py:36-39` | `if args.img_encoder_snapshot is not None` | True | `mirai_trained.json` pins the snapshot | Construct `MiraiFull(args)` normally so `load_model` runs |
| B2 | `external/Mirai/onconet/models/mirai_full.py:41-43` | `if hasattr(args, "freeze_image_encoder") and args.freeze_image_encoder` | False | Attribute unset in `mirai_trained.json` | Ignore; `requires_grad` does not affect forward |
| B3 | `external/Mirai/onconet/models/mirai_full.py:46-50` | `if args.transformer_snapshot is not None` | True | `mirai_trained.json` pins the snapshot | As B1 |
| B4 | `external/Mirai/onconet/models/mirai_full.py:116-121` | try/except setting `model._model.args.use_precomputed_hiddens` / `cuda` | succeeds silently | `model._model` exists; attributes settable | Before construction, explicitly set `args.use_precomputed_hiddens = False` and `args.cuda = False` so the try succeeds deterministically |
| B5 | `external/Mirai/onconet/models/resnet_base.py:41-42` | `if hasattr(args, 'use_spatial_transformer') and args.use_spatial_transformer` | False | Unset in `mirai_trained.json` | No STN in encoder; ignore |
| B6 | `external/Mirai/onconet/models/resnet_base.py:50` | `if not self.args.use_precomputed_hiddens` (builds `Downsampler`) | True | `use_precomputed_hiddens=False` | `Downsampler` present; export wrapper owns the full encoder |
| B7 | `external/Mirai/onconet/models/resnet_base.py:84` | `if not self.pool.replaces_fc()` (attaches `relu`, `dropout`, `fc`) | True | `RiskFactorPool.replaces_fc() == False` (`risk_factor_pool.py:33-34`) | These modules exist on the encoder but their outputs are discarded by `MiraiFull.forward:59` slice |
| B8 | `external/Mirai/onconet/models/resnet_base.py:96-97` | `if args.survival_analysis_setup` (encoder `prob_of_failure_layer`) | depends on saved encoder args; not used on encoder branch | The encoder's own `prob_of_failure_layer` output is discarded by `MiraiFull.forward` | Ignore; encoder export consumes `hidden`, not `logit` |
| B9 | `external/Mirai/onconet/models/resnet_base.py:197-198` | `if self.args.use_precomputed_hiddens: x = x.transpose(2,1)` | False | Set `False` at B4 | No transpose |
| B10 | `external/Mirai/onconet/models/resnet_base.py:199-200` | `if args.use_spatial_transformer: x = self.stn(x)` | False | As B5 | No STN |
| B11 | `external/Mirai/onconet/models/resnet_base.py:201-203` | `if self.args.cuda and self.args.model_parallel` | False | `cuda=False` at B4; `model_parallel` unset | No per-GPU scatter |
| B12 | `external/Mirai/onconet/models/resnet_base.py:209-210` | `if self.args.use_region_annotation` | False | Unset | No `region_logit` |
| B13 | `external/Mirai/onconet/models/resnet_base.py:211-212` | `if self.args.predict_birads` | False | Unset | No `birads_logit` |
| B14 | `external/Mirai/onconet/models/resnet_base.py:214-218` | `if self.args.pred_risk_factors: try: self.pool.get_pred_rf_loss(...) except: pass` | `True` enters; inner call throws silently because `risk_factors=None` | `pred_risk_factors=True` in config | Drop this try block in export wrapper; it causes the per-key FC second call (see `## 7`) |
| B15 | `external/Mirai/onconet/models/resnet_base.py:219-222` | `if self.args.use_precomputed_hiddens: return 4-tuple else: return 3-tuple` | False → 3-tuple `(logit, hidden, activ_dict)` | As B4 | Encoder export reads `hidden` (second element) |
| B16 | `external/Mirai/onconet/models/resnet_base.py:227-230` | `if self.args.use_risk_factors` (encoder's `aggregate_and_classify`) | True | `use_risk_factors=True` | Encoder's pool receives `risk_factors=None`; pool uses predicted RFs (see B20) |
| B17 | `external/Mirai/onconet/models/resnet_base.py:232-240` | `if not self.pool.replaces_fc()` (applies relu/dropout/fc) | True | As B7 | Encoder `hidden` is post-relu; wrapper must replicate |
| B18 | `external/Mirai/onconet/models/resnet_base.py:242-243` | `if self.args.survival_analysis_setup: logit = self.prob_of_failure_layer(hidden)` | depends; encoder's logit is discarded anyway | — | Ignore |
| B19 | `external/Mirai/onconet/models/pools/risk_factor_pool.py:38-39` | `if self.args.replace_snapshot_pool: x = x.data` | False | Unset | No `.data` unwrap |
| B20 | `external/Mirai/onconet/models/pools/risk_factor_pool.py:43` | `if self.args.pred_risk_factors` | True | `pred_risk_factors=True` | Execute per-key `_fc` loop once per key (see B23–B26) |
| B21 | `external/Mirai/onconet/models/pools/risk_factor_pool.py:49-52` | `if num_class == 1: sigmoid else: softmax` | both sides live per-key | Per-key class counts from `risk_factor_key_to_num_class` | Reproduce exactly in the wrapper's pool forward |
| B22 | `external/Mirai/onconet/models/pools/risk_factor_pool.py:54-56` | `if not self.training and self.args.use_pred_risk_factors_if_unk` | False | `use_pred_risk_factors_if_unk=False` in config (MANIFEST.config) | Drop |
| B23 | `external/Mirai/onconet/models/pools/risk_factor_pool.py:57-59` | `elif self.training and self.args.mask_prob > 0 and gold_rf is not None` | False | Pool is in eval mode at inference; `mask_prob` unset (0) | Drop; also removes the `np.random.random()` call (non-differentiable) |
| B24 | `external/Mirai/onconet/models/pools/risk_factor_pool.py:66` | `if (not self.training and use_pred_risk_factors_at_test) or (self.training and mask_prob > 0)` | True (first clause: pool in eval, `use_pred_risk_factors_at_test=True`) | Pool's `self.training == False` via `torch.load`; `use_pred_risk_factors_at_test=True` | Wrapper's pool always uses the tensor blend: `rf_used = rf_known_mask * rf_vector + (1 - rf_known_mask) * rf_predicted` |
| B25 | `external/Mirai/onconet/models/pools/risk_factor_pool.py:69` | `risk_factors_hidden = torch.cat(risk_factors, dim=1) if risk_factors_hidden is None else risk_factors_hidden` | no-op (`risk_factors_hidden` is set from B24) | — | Dead in wrapper |
| B26 | `external/Mirai/onconet/models/pools/risk_factor_pool.py:71` | `hidden = self.dropout(hidden)` | eval = identity | — | Drop or keep; no effect |
| B27 | `external/Mirai/onconet/models/hiddens_transfomer.py:42-46` | `if not self.pool.replaces_fc(): self.relu / self.dropout / self.fc` | True (constructor) | As B7 | These modules exist; wrapper reuses `self.relu` (but out-of-place) |
| B28 | `external/Mirai/onconet/models/hiddens_transfomer.py:48-53` | `if args.survival_analysis_setup: self.prob_of_failure_layer = ...` | True | `survival_analysis_setup=True` (implicit via `max_followup=5` and weights-present check) | Present; wrapper uses it |
| B29 | `external/Mirai/onconet/models/hiddens_transfomer.py:57` | `mask_prob = self.args.mask_prob if self.training and self.args.pred_missing_mammos else 0` | 0 | `pred_missing_mammos` unset | Skip `mask_input` entirely in wrapper (see `## 11` R1) |
| B30 | `external/Mirai/onconet/models/hiddens_transfomer.py:63-65` | `if self.args.also_pred_given_mammos` | False | Unset | Drop |
| B31 | `external/Mirai/onconet/models/hiddens_transfomer.py:97-106` | try/except with `if self.args.predict_birads` / `if self.args.pred_risk_factors` / `if self.args.pred_missing_mammos` | only `pred_risk_factors=True` enters; inner `get_pred_rf_loss` triggers the per-key FC second call (see `## 7`) | — | Drop the entire try/except in the wrapper; return `(logit, hidden_post_relu)` directly |
| B32 | `external/Mirai/onconet/models/hiddens_transfomer.py:112-115` | `if self.args.use_risk_factors: self.pool(x, risk_factors) else: self.pool(x)` | True → first branch | `use_risk_factors=True` | Wrapper's transformer branch passes the blended rf to the pool |
| B33 | `external/Mirai/onconet/models/hiddens_transfomer.py:117-125` | `if not self.pool.replaces_fc(): try: hidden = self.relu(hidden) except: pass; hidden = self.dropout(hidden); logit = self.fc(hidden)` | True; relu fires | `replaces_fc() == False` | Wrapper replaces with out-of-place `F.relu`; drops dropout (eval no-op); drops `self.fc` (logit overridden by `prob_of_failure_layer`) |
| B34 | `external/Mirai/onconet/models/hiddens_transfomer.py:127-131` | `if self.args.survival_analysis_setup: if pred_both_sides: ... else: logit = self.prob_of_failure_layer(hidden)` | True, `pred_both_sides=False` → `self.prob_of_failure_layer(hidden)` | Config values | Wrapper uses `self.prob_of_failure_layer` directly |
| B35 | `external/Mirai/onconet/models/cumulative_probability_layer.py:25-26` | `if self.args.make_probs_indep: return self.hazards(x)` | False | Unset | Skip; always run the cumulative path |

The remaining branches inside `TransformerLayer` and `MultiHead_Attention` are unconditional compute (no `self.training`/args branches); they export as-is.

---

## 6. Eval-Mode Situation

Upstream Mirai never calls `.eval()` on the model in `MiraiModel.run_model` (`external/Mirai/onconet/models/mirai_full.py:200-271`). The function only moves the model to CPU (`mirai_full.py:149`). Inference correctness depends entirely on what training mode was saved in each snapshot.

Phase 0 recorded the actual training flags in `tests/reference/_hooks.py:126-135`. The observed table on the demo snapshots:

| Module | `self.training` at inference |
|---|---|
| `MiraiFull` (outer) | `True` (default `nn.Module` state; freshly constructed) |
| `MiraiFull.image_encoder` (`CustomResnet` + `ResNet`) | `False` (restored from `torch.load`) |
| `MiraiFull.transformer` (`AllImageTransformer`) | `False` (restored from `torch.load`) |
| `MiraiFull.transformer.pool` (`RiskFactorPool`) | `False` (restored from `torch.load`) |
| `MiraiFull.transformer.prob_of_failure_layer` (`Cumulative_Probability_Layer`) | `False` (restored from `torch.load`) |

This arrangement happens to be correct: BatchNorm uses running statistics, Dropout is a no-op, and `RiskFactorPool.forward`'s `self.training`-guarded branches (`risk_factor_pool.py:54-66`) resolve to the predicted-RF path that the demo relies on. But it is fragile. **Phase 2 and Phase 3 export wrappers MUST call `.eval()` on the entire model tree before `torch.onnx.export`**, even though upstream code does not. This both normalizes state (so the wrapper's behavior does not depend on which pickled mode happened to be saved) and ensures `torch.onnx.export` traces the correct eval-mode paths for BatchNorm and Dropout.

Concretely:

```python
model = MiraiFull(args)         # constructs; load_model runs its arg mutations
model = model.cpu()
model.eval()                    # recursive; puts every submodule in eval
wrapper = ImageEncoderExport(model)  # or RiskModelExport(model)
wrapper.eval()                  # belt-and-suspenders; new wrappers inherit train by default
```

If the wrapper constructs intermediate modules (e.g. a clean `RiskFactorPool` replacement), those new modules also start in train mode and must be explicitly `.eval()`-ed.

---

## 7. Per-Key FC Double-Call

Each of the 34 `{key}_fc` Linear layers in `RiskFactorPool` is invoked more than once per inference, and a naive hook or a naive export wrapper will double-count.

Call sites:

1. **First call** — `external/Mirai/onconet/models/pools/risk_factor_pool.py:47` inside `RiskFactorPool.forward`. Input: `hidden`, the raw `internal_pool` output `(B, 512)`. The output feeds the concatenated `risk_factor_vector` that goes into `hidden_pre_hazard`.
2. **In-place ReLU** — `external/Mirai/onconet/models/hiddens_transfomer.py:121` runs `hidden = self.relu(hidden)` with `nn.ReLU(inplace=True)` on the pool output. This mutates the storage.
3. **Second call (on both encoder and transformer paths)** — `external/Mirai/onconet/models/pools/risk_factor_pool.py:81` inside `get_pred_rf_loss`. Input: `hidden[:, :-length_risk_factor_vector]`, i.e. the image-only slice of the post-relu hidden. Reached from:
   - `external/Mirai/onconet/models/hiddens_transfomer.py:101` (inside the try/except in `AllImageTransformer.forward`).
   - `external/Mirai/onconet/models/resnet_base.py:216` (inside the try/except in `ResNet.forward`).
   In both sites the second call raises at `risk_factor_pool.py:82` (`risk_factors[i]` with `risk_factors=None`) and is swallowed by the try/except. So the second call executes for at most one key (the first) before the loop bails.

**Phase 0's `_hooks.py:171-180` guards against the double-count** with:

```python
def make_hook(k):
    def hook(module, inputs, output):
        if k not in captured["pred_risk_factor_logits_per_key"]:
            captured["pred_risk_factor_logits_per_key"][k] = _to_numpy(output)
    return hook
```

This keeps only the first call per key.

**Export rule.** Phase 2 and Phase 3 wrappers must invoke each `{key}_fc` module **exactly once**, on the raw `internal_pool` hidden (the first-call input). Do not mirror the upstream double-call. Concretely: the wrapper's pool `forward` calls `{key}_fc(image_hidden)` once per key and concatenates; the wrapper's transformer `forward` does NOT call `get_pred_rf_loss`.

---

## 8. XAI-Hidden Choice (Pre-relu vs Post-relu)

Two candidate tensors are available:

| Variant | Shape | Fixture | Source |
|---|---|---|---|
| Pre-relu | `(1, 612)` fp32 | `pool_hidden.npy` / `pool_hidden_dcmtk.npy` | Raw `RiskFactorPool.forward` output at `risk_factor_pool.py:70-72` |
| Post-relu | `(1, 612)` fp32 | `xai_hidden.npy` / `xai_hidden_dcmtk.npy` | Direct input to `prob_of_failure_layer` — output of `hiddens_transfomer.py:121` in-place `self.relu(hidden)` |

**Commitment: `XAI-hidden = post-relu`.** The `risk_model.onnx` output named `hidden_pre_hazard` is the post-relu tensor, validated by Phase 3 against `xai_hidden{,_dcmtk}.npy`.

Rationale. The post-relu hidden is the tensor the hazard head actually consumes, so it is the most faithful representation for downstream interpretability work (attribution, embedding-space comparisons) — if a consumer wants to explain what drove the risk score, they need the input the scoring function saw. The pre-relu tensor retains sign information that ReLU discards, so it remains useful for consumers who care about that signal; it is still available as `pool_hidden{,_dcmtk}.npy` fixtures and as an optional Phase 3+ deliverable, but it is not the default `hidden_pre_hazard` output.

---

## 9. Export Surface Specification

The strings in this section are the contract. Phase 2/3 exports must match them verbatim. The lint test `tests/architecture/test_plan.py` enforces the structural invariants.

### 9.1 `image_encoder.onnx`

| Role | Name | Shape | Dtype | Dynamic axes |
|---|---|---|---|---|
| input | `input` | `(N, 3, 2048, 1664)` | fp32 | `{0: "N"}` |
| output | `output` | `(N, 512)` | fp32 | `{0: "N"}` |

Semantics. `input` is the **post-preprocessing** per-view tensor produced by the pipeline described in `## 3` rows 1–2 (DICOM → windowing → `Scale_2d` to `(H=2048, W=1664)` → align-to-left → normalize with `mean=7047.99`, `std=12005.5` → 3-channel expand). Normalization is **not** part of the ONNX graph; it lives in the preprocessor (TS in Phase 6, Python in Phase 5).

The output is the **post-slice, post-relu** image feature — matching `image_encoder_out.npy` (`(1, 4, 512)` when `N=4`, reshaped to `(1, 4, 512)` by the caller).

Implementation note. The underlying `ResNet.forward` returns `(logit, hidden_612, activ_dict)`. The export wrapper takes `hidden_612`, applies the `[:, :512]` slice from `external/Mirai/onconet/models/mirai_full.py:59`, and returns it. The encoder's own `prob_of_failure_layer` (if any) and `self.fc` are bypassed (their outputs would be discarded anyway; excluding them keeps the graph smaller).

### 9.2 `risk_model.onnx`

Inputs:

| Name | Shape | Dtype | Semantics |
|---|---|---|---|
| `img_feats` | `(B, 4, 512)` | fp32 | Per-view image features stacked for the four views; equals `image_encoder.onnx` outputs reshaped |
| `view_seq` | `(B, 4)` | int64 | View code per slot: `CC=0`, `MLO=1`. Valid range `[0, MAX_VIEWS=2]` (PAD=2 unused for single-exam inference); see `external/Mirai/onconet/models/hiddens_transfomer.py:14` |
| `side_seq` | `(B, 4)` | int64 | Side code per slot: `R=0`, `L=1`. Valid range `[0, MAX_SIDES=2]` (PAD=2 unused); see `hiddens_transfomer.py:15` |
| `time_seq` | `(B, 4)` | int64 | Time-offset code per slot. Valid range `[0, MAX_TIME=10]`; use `zeros((B, 4))` for single-exam inference; see `hiddens_transfomer.py:13` |
| `rf_vector` | `(B, 100)` | fp32 | User-supplied risk-factor vector (per-position). Fill with zeros to invoke model-predicted fill-in |
| `rf_known_mask` | `(B, 100)` | fp32 | Per-position mask: `1.0` where `rf_vector` is valid, `0.0` where to use model prediction. Fill with zeros for all-model-predicted behaviour |

Outputs:

| Name | Shape | Dtype | Semantics |
|---|---|---|---|
| `logit` | `(B, 5)` | fp32 | Pre-sigmoid 5-year cumulative probabilities from `Cumulative_Probability_Layer`; matches `raw_logit{,_dcmtk}.npy` |
| `hidden_pre_hazard` | `(B, 612)` | fp32 | **Post-relu** XAI embedding (see `## 8`); matches `xai_hidden{,_dcmtk}.npy` |

Dynamic axes: `{0: "B"}` on every input and output.

Internal blend. Within the ONNX graph, the pool computes:

```
rf_predicted = concat(per_key_probs_in_keys_order, dim=1)     # (B, 100)
rf_used      = rf_known_mask * rf_vector + (1 - rf_known_mask) * rf_predicted
hidden_pre   = concat(image_hidden, rf_used, dim=1)           # (B, 612)
hidden_post  = F.relu(hidden_pre)                             # out-of-place
logit        = prob_of_failure_layer(hidden_post)
```

---

## 10. Slot-Ordering Convention

The graph is **order-agnostic**: given consistent `view_seq` and `side_seq`, the caller may place the four views in any slot of the batch. The ONNX model does not know or assume an order.

The caller reads the slot-to-(view, side) mapping from `tests/reference/fixtures/batch_order.json` (and `batch_order_dcmtk.json` for the dcmtk path). For the demo inputs (`ccl1.dcm ccr1.dcm mlol2.dcm mlor2.dcm`), that order is `[(CC,L), (CC,R), (MLO,L), (MLO,R)]`, driven by `dict` iteration in `MiraiModel.run_model` (`external/Mirai/onconet/models/mirai_full.py:216-229`) over the CLI input order.

Do **not** hardcode any slot ordering into export wrappers. The earlier CSV-training-era order (right-then-left, CC-then-MLO) mentioned in `onconet.datasets.csv_mammo_cancer` is not what the demo inference produces. Phase 5's Python harness and Phase 8's TS pipeline must read `batch_order.json` at call time.

---

## 11. Refactor Targets for Phase 2/3

All refactors live in **new export-wrapper modules** (e.g. `onconet/models/mirai_full_export.py`, or a repo-root `scripts/` module). Nothing under `external/Mirai/` is edited.

| # | Target (file:line in upstream Mirai) | Issue | Wrapper-side resolution |
|---|---|---|---|
| R1 | `external/Mirai/onconet/models/hiddens_transfomer.py:55-66` (`AllImageTransformer.mask_input`) | At inference, `mask_prob` resolves to 0 (B29), but `torch.bernoulli` / `torch.cat` / `Embedding` lookups still appear in a traced graph. | In the transformer export wrapper, skip `mask_input` entirely; feed `projection_layer` directly. |
| R2 | `external/Mirai/onconet/models/hiddens_transfomer.py:97-106` (try/except in `AllImageTransformer.forward`) | Triggers `get_pred_rf_loss` which fires the per-key FC second call (B31, `## 7`). | Drop the try/except. Return `(logit, hidden_post_relu)` directly. |
| R3 | `external/Mirai/onconet/models/hiddens_transfomer.py:110-133` (`aggregate_and_classify`) | In-place `self.relu(hidden)` mutates storage that the pool forward hook aliased; dropout is eval no-op; `self.fc(hidden)` is a discarded logit. | Wrapper replaces with: `_, hidden_pre = self.pool(img_like_hidden, rf_inputs)`; `hidden_post = F.relu(hidden_pre)` (out-of-place); `logit = self.prob_of_failure_layer(hidden_post)`. |
| R4 | `external/Mirai/onconet/models/pools/risk_factor_pool.py:36-72` (`RiskFactorPool.forward`) | Branches on `self.training`, `use_pred_risk_factors_*`, `np.random.random()`. | Clean wrapper `forward`: (a) `_, image_hidden = self.internal_pool(x)`; (b) per-key `key_logit = self._modules[f"{key}_fc"](image_hidden)` called **exactly once** per key; sigmoid (binary) or softmax (multi-class) per `risk_factor_key_to_num_class`; (c) `rf_predicted = torch.cat(per_key_probs, dim=1)`; (d) `rf_used = rf_known_mask * rf_vector + (1 - rf_known_mask) * rf_predicted`; (e) `hidden = torch.cat([image_hidden, rf_used], dim=1)`. Skip dropout. |
| R5 | `external/Mirai/onconet/models/cumulative_probability_layer.py:7-17` (`upper_triagular_mask` as non-trainable `Parameter`) | Serializes as an ONNX initializer; shape is `(max_followup, max_followup)`. | No code refactor. Wrapper pins `max_followup=5`; never re-instantiate `Cumulative_Probability_Layer` after snapshot load. |
| R6 | `external/Mirai/onconet/models/mirai_full.py:53-61` (`MiraiFull.forward`, the `[:, :, :img_repr_dim]` slice at line 59) | Image encoder's raw hidden is `(B*N, 612)`; slice drops the rf block to get `(B*N, 512)`. | Image-encoder wrapper **must** apply the slice. Risk-model wrapper **must not** apply the slice (its `img_feats` input is already `(B, 4, 512)`). |
| R7 | `external/Mirai/onconet/models/mirai_full.py:116-121` (`MiraiModel.load_model` try/except) | Silently swallows failures setting `use_precomputed_hiddens` and `cuda`. | Set `args.use_precomputed_hiddens = False` and `args.cuda = False` **before** constructing `MiraiFull` so the try block succeeds deterministically. |
| R8 | `external/Mirai/onconet/models/factory.py:84-119` (`load_model`) | Mutates snapshot's `args` to match run-time `use_pred_risk_factors_*`, `pred_risk_factors`, etc. | Construct `MiraiFull(args)` via its standard constructor (which calls `load_model` internally). Do **not** bypass by calling `torch.load` on the snapshot path directly — the mutations would be missed. |

---

## 12. Export-Time Settings

- **Exporter.** Classic `torch.onnx.export` (tracer-based). `dynamo_export` is not used — our graph has no Python dynamism once the `## 5` branches are resolved, and the classic tracer is stable across torch 1.x / 2.x.
- **Opset.** `opset_version=17`. This is well-supported in `onnxruntime-web ≥1.17` and covers every op Mirai needs (Conv, BatchNorm, Gemm, MatMul, LayerNorm, Gather, Softmax, Sigmoid, Relu, Concat, Slice, Reshape, Transpose, Unsqueeze, Div, Sub, Mul, Add, ReduceMax).
- **`do_constant_folding=True`.** Folds initializers, shrinks graph size.
- **Tracer inputs.** Build from Phase 0 fixtures:
  - Image encoder: `torch.from_numpy(np.stack([preproc_tensor[v] for v in batch_order]))` → `(4, 3, 2048, 1664)` fp32.
  - Risk model: `torch.from_numpy(np.load("image_encoder_out.npy"))` for `img_feats`; `view_seq`/`side_seq` as `torch.tensor(..., dtype=torch.int64)` reshaped `(1, 4)` from `batch_order.json`; `time_seq = torch.zeros((1, 4), dtype=torch.int64)`; `rf_vector = torch.zeros((1, 100), dtype=torch.float32)`; `rf_known_mask = torch.zeros_like(rf_vector)`.
- **Dynamic-axes recipes.**
  ```python
  # image_encoder
  dynamic_axes = {"input": {0: "N"}, "output": {0: "N"}}
  # risk_model
  dynamic_axes = {
      "img_feats":     {0: "B"},
      "view_seq":      {0: "B"},
      "side_seq":      {0: "B"},
      "time_seq":      {0: "B"},
      "rf_vector":     {0: "B"},
      "rf_known_mask": {0: "B"},
      "logit":            {0: "B"},
      "hidden_pre_hazard":{0: "B"},
  }
  ```
- **Validation after export.** Run `onnx.checker.check_model(model_proto)`; then validate in two stages with the two-track tolerances from §1:
  1. **PyTorch wrapper forward** vs the Phase 0 fixture: `atol=0.0, rtol=0` (bit-exact). This is the correctness gate for the wrapper code itself — wrapper bugs (wrong slice, missed `.eval()`, mutated storage) show up here.
  2. **ONNX Runtime CPU session** vs the Phase 0 fixture: `atol=2e-5, rtol=0`. This absorbs intrinsic torch-ATen vs ORT-MLAS kernel-rounding differences. Phase 2 established the bound empirically (9.5e-6 pydicom, 1.05e-5 dcmtk on the image encoder); Phase 3 should run the same two-stage check, and if ORT diff exceeds 2e-5 on the risk model, record the new measurement in `PHASE_3_REPORT.md` rather than silently bumping the tolerance.
- **Torch version for the export environment.** The capture environment `mirai-py38` (torch 1.9.0) is the reference runner, not the exporter. torch 1.9.0's ONNX exporter is old and lacks many quality-of-life features (e.g. proper `LayerNorm` fusion). Phase 2 should set up a separate venv (e.g. `mirai-export`) with `torch>=2.1` for running `torch.onnx.export`, while continuing to import the same `onconet` package. Snapshot files load fine in either torch version.

---

## 13. Snapshot Loading and .eval() Sequencing

Phase 2 and Phase 3 export scripts follow this sequence, in order:

1. **Parse the config.** Load `external/Mirai/onconet/configs/mirai_trained.json` via `onconet.predict._load_config` (same entry point as Phase 0). This produces an `args` namespace.
2. **Pin runtime flags.** `args.cuda = False`; `args.use_precomputed_hiddens = False`. These prevent B4 and B9 surprises and avoid GPU-only paths.
3. **Construct `MiraiFull`.** `model = MiraiFull(args)`. Internally this calls `load_model` on each snapshot (`external/Mirai/onconet/models/factory.py:84-119`), which mutates the snapshot `args` for `use_pred_risk_factors_at_test`, `pred_risk_factors`, etc. Do **not** bypass this step with a manual `torch.load`.
4. **Unwrap `DataParallel`.** If `isinstance(model, nn.DataParallel): model = model.module`. (The demo snapshots are not wrapped, but defensive.)
5. **Move to CPU and call `.eval()` on the entire tree.** `model = model.cpu(); model.eval()`. This is the explicit step upstream skips (see `## 6`).
6. **Construct the export wrapper.** `ImageEncoderExport(model)` or `RiskModelExport(model)`. The wrapper references `model.image_encoder` or `model.transformer` — it does not deep-copy. Call `.eval()` on the wrapper too (belt-and-suspenders).
7. **Verify `rf_dim`.** `assert model.transformer.pool.length_risk_factor_vector == 100` against `MANIFEST.runs.pydicom.shapes.rf_dim`. Fail loudly on mismatch.
8. **Build dummy inputs** from Phase 0 fixtures (see `## 12`).
9. **Export.** `torch.onnx.export(wrapper, dummy_inputs, path, input_names=[...], output_names=[...], dynamic_axes=..., opset_version=17, do_constant_folding=True)`.
10. **Validate.** `onnx.checker.check_model(path)`; open an `ort.InferenceSession`; compare outputs to Phase 0 fixtures at `atol=1e-6, rtol=0`.

---

## 14. Notes and Pitfalls

- **`RiskFactorPool` constructor mutates `args`.** `external/Mirai/onconet/models/pools/risk_factor_pool.py:29-31` sets `args.img_only_dim = args.hidden_dim`, `args.rf_dim = length_risk_factor_vector`, and `args.hidden_dim = rf_dim + img_only_dim` (so after construction, `args.hidden_dim` goes from 512 to 612). Do not re-instantiate `MiraiFull` after weights load; you will desync dimensions.
- **`upper_triagular_mask` is a non-trainable `Parameter`, not a buffer.** `external/Mirai/onconet/models/cumulative_probability_layer.py:14-17`. It serializes as an ONNX initializer just fine — but if you change `max_followup`, the initializer shape changes silently and Phase 3's validation will mismatch. Pin `max_followup=5`.
- **Hidden-transformer import cycle.** `external/Mirai/onconet/models/hiddens_transfomer.py` imports `from onconet.models.pools.factory import get_pool` and `from onconet.models.cumulative_probability_layer import Cumulative_Probability_Layer`. If a future refactor hoists these modules out of `external/Mirai/`, respect the import order.
- **`MiraiFull.forward` slicing asymmetry.** `external/Mirai/onconet/models/mirai_full.py:59` slices `[:, :, :img_repr_dim]` on the **image-encoder branch only**. The risk-model branch consumes `img_x` after this slice. Any export wrapper that touches both branches must apply the slice on exactly the encoder side.
- **In-place ReLU storage aliasing.** `external/Mirai/onconet/models/hiddens_transfomer.py:121` runs `hidden = self.relu(hidden)` with `nn.ReLU(inplace=True)` on the pool output. The same storage is captured by the pool's forward hook. Any code that needs both pre- and post-relu views must `.copy()` the numpy array after `.numpy()` — Phase 0's `_hooks.py:23-26` does this explicitly.
- **Positional embeddings use `padding_idx=-1`** (`external/Mirai/onconet/models/hiddens_transfomer.py:145-147`). In PyTorch this means "the last valid index, sign-corrected" — here `MAX_TIME=10`, `MAX_VIEWS=2`, `MAX_SIDES=2` — so padding_idx resolves to 10, 2, 2 respectively. For single-exam inference we never pass PAD values (view/side/time all strictly below their MAX), so the `padding_idx` effect is unused. ONNX's `Gather` does not encode padding_idx semantics; Phase 3's validation tests must not exercise PAD values.
- **The `.transpose(1,2).unsqueeze(-1)` reshape** at `external/Mirai/onconet/models/hiddens_transfomer.py:93` converts `(B, N, hidden_dim)` into `(B, hidden_dim, N, 1)`, an image-like layout consumed by `GlobalMaxPool`. This reshape must be preserved verbatim in the risk-model wrapper — easy to lose when refactoring.

---

## 15. How to Validate This Plan

1. **Lint test.** From the repo root with the `mirai-py38` conda env active:
   ```bash
   pytest tests/architecture/test_plan.py -v
   ```
   All tests pass.
2. **Read the doc top-to-bottom.** Confirm every committed decision reads correctly. In particular: `## 8` commits to post-relu unambiguously; `## 9.1` and `## 9.2` list the ONNX I/O names and shapes; `## 12` commits `opset_version=17` and classic `torch.onnx.export`.
3. **Spot-check refactor targets.** Open any three rows from `## 11` by their file:line references and verify the cited code is at that line in the pinned Mirai SHA `4af944449863966a5a9c66b44e56e3c141223897`.
4. **Spot-check shape rows.** Verify any three rows from `## 3` against `tests/reference/fixtures/MANIFEST.json`.
5. **Surface audit.** Confirm no file under `tests/reference/fixtures/` or `external/Mirai/` was modified — `git status` should list only the Phase 1 additions (`docs/architecture.md`, `tests/architecture/__init__.py`, `tests/architecture/test_plan.py`, `CLAUDE.md`, `PHASE_1_REPORT.md`).

Sign-off: steps 1–5 all green.
