# Mirai → ONNX → Web (WebGPU) Migration Plan

This document is a phased, verifiable plan for compiling the Mirai mammography-based breast cancer risk model into ONNX and porting its surrounding preprocessing/postprocessing pipeline into TypeScript, so that the full pipeline can run client-side in a browser via `onnxruntime-web` + WebGPU.

The plan is structured so that each phase can be assigned to an agent (e.g. Claude Code) as a self-contained work unit with an explicit *automated verification condition* (a script the agent must make pass) and a *manual verification condition* (what you check by hand before moving on).

---

## 0. Context and Constraints

### 0.1 What Mirai is, structurally
`MiraiFull` (`onconet/models/mirai_full.py`) is a two-stage model:

1. **Image encoder** — a `custom_resnet` (ResNet-18-ish; `BasicBlock,2 × 4` in `configs/mirai_base.json`) applied *independently* to each of four "For Presentation" views (L CC, L MLO, R CC, R MLO). Input per image is `(3, 2048, 1664)` (CHW; height before width) with mean=`7047.99`, std=`12005.5`. Output per image is a 512-d vector (`img_only_dim`). Note: `Scale_2d` reads `width, height = args.img_size = [1664, 2048]` then calls `Resize((height, width))`, hence the H,W order.

2. **Risk transformer** — an `AllImageTransformer` that (a) adds learned view/side/time positional embeddings to the four 512-d image vectors, (b) runs a multi-head self-attention transformer over them, (c) pools through a `RiskFactorPool` that *concatenates* the pooled image hidden with a `rf_dim`-element risk-factor vector (`rf_dim=100` for the trained config; **34 is the count of `risk_factor_keys`, not the vector dimensionality** — each key produces between 1 and ~10 binary/one-hot dims). The authoritative source for `rf_dim` is `model.transformer.pool.length_risk_factor_vector`. The concatenation is then fed into a `Cumulative_Probability_Layer` to produce 5-year cumulative hazards. A per-year Platt-scaling `MiraiCalibrator` is applied outside the torch graph.

### 0.2 The "XAI embedding" you want to expose
The tensor you asked about — the representation that directly feeds the hazard head — is the variable named `hidden` returned by `AllImageTransformer.aggregate_and_classify`. It is exactly `[image_feature (img_only_dim=512) ; risk_factor_vector (rf_dim=100)]` concatenated along dim=1, giving shape `(B, 612)`. It is the **sole** input to `prob_of_failure_layer`. Exposing this tensor is a first-class deliverable of this plan, not a bolt-on.

**Important caveat about pre-relu vs post-relu.** The `hidden` returned by `aggregate_and_classify` and consumed by `prob_of_failure_layer` is **post-relu**, because `RiskFactorPool.replaces_fc()` returns False — which triggers `hidden = self.relu(hidden)` (in place, on the pool output) inside `aggregate_and_classify`. Phase 0 captured both:

- `pool_hidden{,_dcmtk}.npy` — `(1, 612)` fp32, **pre-relu** (raw `RiskFactorPool` output).
- `xai_hidden{,_dcmtk}.npy` — `(1, 612)` fp32, **post-relu** (the actual input to `prob_of_failure_layer`).

Phase 1 must explicitly decide which one is the public XAI embedding. Default = post-relu (matches what feeds the hazard head); pre-relu is also available in the fixtures if downstream consumers prefer it.

### 0.3 Runtime risk-factor modes to preserve
The Python model has three effective modes, controlled by `use_pred_risk_factors_at_test` and `use_pred_risk_factors_if_unk` in `RiskFactorPool.forward`. The web build must preserve all three, ideally via a single ONNX graph with a `rf_known_mask` input:

| Mode | Gold RF vector | Known mask | Behavior |
|------|----------------|------------|----------|
| All user-supplied | real values | all ones | use user RFs verbatim |
| Partial user-supplied | real values for known, zeros for unknown | ones for known, zeros for unknown | use user RFs where known, model-predicted where unknown |
| All model-predicted | zeros | all zeros | use model-predicted RFs everywhere |

### 0.4 Guiding principles for the whole migration

- **Numerical parity is king.** At every boundary we compare TS/ONNX outputs against the Python reference on the **same inputs**, with a clear tolerance (`atol=1e-5` for fp32 tensors, stricter for deterministic ops).
- **Freeze a golden reference early.** Phase 0 captures fixed inputs and fixed outputs for every intermediate tensor we care about; every later phase validates against that frozen reference, not against a freshly-run model.
- **Export in two ONNX files, not one.** An `image_encoder.onnx` (per-view) and a `risk_model.onnx` (four-view + RFs). This mirrors the natural seam in the PyTorch code, keeps per-image tensors small enough for WebGPU, and makes the calibrator + embedding live in a single, cheap graph.
- **Keep non-differentiable logic out of ONNX.** DICOM decoding, windowing, align-to-left, and calibration are *not* in the ONNX graph. They are in deterministic TS/Python code that we test against each other directly.
- **One agent per phase, but one reference harness shared across all phases.** See §11.

---

## 1. Phase 0 — Baseline and Golden Reference Capture

**Status: COMPLETE (2026-04-22).** Authoritative records:
- `PHASE_0_REPORT.md` (repository root) — every file created, every command run, every finding.
- `tests/reference/README.md` — fixture inventory and reproduction commands.
- `tests/reference/ENV.md` — env-creation commands and the pylibjpeg workaround.
- `tests/reference/fixtures/MANIFEST.json` — versions, snapshot SHA-256s, per-fixture SHA-256/dtype/shape, asserts that passed at capture time. **Treat MANIFEST as source of truth for paths and dimensionalities.**

### What exists on disk now (Phase 1+ should import these, not regenerate them)

Code under `tests/reference/`:
- `conftest.py` — path constants and tolerances (`ATOL_FP32 = 1e-6`, `ATOL_FP64 = 1e-9`, `RTOL = 0`).
- `_hooks.py` — `install_dicom_capture`, `install_collate_capture`, `install_post_load_hooks`, `derive_pred_risk_factors_per_key`. Reusable forward-hook + monkey-patch helpers.
- `capture_reference.py` — one-shot capture script (`--regenerate`, `--only pydicom|dcmtk|both`).
- `test_baseline.py` — 56 parametrized pytest assertions; passes idempotently in ~50 s.

109 fixture files under `tests/reference/fixtures/`, captured for **both** pydicom and dcmtk DICOM-decode paths (suffix `_dcmtk` on the dcmtk variants):

| Fixture | Shape / dtype | Notes |
|---|---|---|
| `predictions{,_dcmtk}.json` | 5 floats | Bit-equal to `mirai-predict` on demo |
| `batch_order{,_dcmtk}.json` | 4 (view,side) tuples | **Authoritative slot ordering** |
| `dicom_raw_uint16{,_dcmtk}/{CC,MLO}_{L,R}.npy` | uint16, per-DICOM | Bit-exact across re-runs |
| `preproc_tensor{,_dcmtk}/{CC,MLO}_{L,R}.npy` | `(3, 2048, 1664)` fp32 | Per-image encoder input |
| `image_encoder_out{,_dcmtk}.npy` | `(1, 4, 512)` fp32 | Stacked image features |
| `image_hidden_in_pool{,_dcmtk}.npy` | `(1, 512)` fp32 | Pool image-side input slice |
| `pool_hidden{,_dcmtk}.npy` | `(1, 612)` fp32 | **Pre-relu** pool output |
| `risk_factor_vector{,_dcmtk}.npy` | `(1, 100)` fp32 | **Model-predicted, NOT zero** |
| `pred_risk_factors_per_key{,_dcmtk}/<key>.npy` | varies | 34 files per path |
| `xai_hidden{,_dcmtk}.npy` | `(1, 612)` fp32 | **Post-relu**, input to `prob_of_failure_layer` |
| `raw_logit{,_dcmtk}.npy` | `(1, 5)` fp32 | `Cumulative_Probability_Layer` output |
| `raw_sigmoid{,_dcmtk}.npy` | `(1, 5)` fp32 | `1/(1+exp(-raw_logit))` |
| `calibrated{,_dcmtk}.npy` | `(5,)` **fp64** | Post-calibration; preserve double precision downstream |
| `preview/{CC,MLO}_{L,R}.png` | 4 PNGs | Eyeball renders, pydicom path only |

### Pinned demo predictions (use as absolute baseline in Phases 4–8)

| Year | pydicom path | dcmtk path |
|---|---|---|
| 1 | 0.0314 | 0.0298 |
| 2 | 0.0505 | 0.0483 |
| 3 | 0.0711 | 0.0684 |
| 4 | 0.0935 | 0.09   |
| 5 | 0.1052 | 0.1016 |

### Captured-environment caveats

- Python 3.8 under Rosetta-translated `osx-64` conda env (`mirai-py38`); torch 1.9.0; pydicom 2.3.0. There is no Apple-Silicon arm64 wheel for torch 1.9.0 / Python 3.8.
- `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg`, `pylibjpeg-rle`, `pylibjpeg` **must be uninstalled** — their PyPI wheels advertise `macosx_*_x86_64` filenames but ship arm64 binaries, breaking pydicom imports under x86_64 Python. Demo DICOMs are uncompressed so pydicom's built-in handlers suffice.
- Tolerances are **machine-pinned** to the capture host. Cross-machine runs may drift at 1e-6; cross-machine reproducibility (Docker reference) is deferred — see §11.
- Determinism settings: `torch.set_num_threads(1)`, `torch.use_deterministic_algorithms(True, warn_only=True)`, fixed seeds.

### Four corrections to this plan that Phase 0 verified at runtime

These are the items §0.1, §3, §6, and §7 originally got wrong; the rest of this document has been updated to match:

1. **Preproc tensor shape is `(3, 2048, 1664)`**, not `(3, 1664, 2048)` — `Scale_2d` does `Resize((height=2048, width=1664))`.
2. **`risk_factor_vector` length is 100**, not 34 — 34 is the count of `risk_factor_keys`; each key produces multiple binary/one-hot dims. Read at runtime from `model.transformer.pool.length_risk_factor_vector`.
3. **The demo `risk_factor_vector` is NOT zeros.** With `pred_risk_factors=true` and `use_pred_risk_factors_at_test=true` (both set in `mirai_trained.json`), the pool concatenates **model-predicted** RFs.
4. **Batch slot order is `[(CC,L), (CC,R), (MLO,L), (MLO,R)]`** for the demo CLI input order, NOT `R-CC, R-MLO, L-CC, L-MLO`. Determined by dict iteration in `MiraiModel.run_model`. Future phases must read from `batch_order.json`, not hardcode any order.

### Reproducing

```bash
conda activate mirai-py38
cd /Users/jeya/Documents/projects/mirai-onnx
python -m tests.reference.capture_reference --regenerate   # only if you need to overwrite
pytest tests/reference/test_baseline.py -v                 # 56 passed
```

---

## 2. Phase 1 — Architecture Analysis and Export Plan

**Goal.** Produce the artifact that makes the ONNX export itself mechanical: a document that lists every tensor, every shape, every non-trivial control-flow branch, and the precise export-time settings for each.

**Inputs.** Phase 0 fixtures and the code in `onconet/models/mirai_full.py`, `onconet/models/hiddens_transfomer.py`, `onconet/models/resnet_base.py`, `onconet/models/pools/risk_factor_pool.py`, `onconet/models/cumulative_probability_layer.py`.

**Tasks.**
1. **Read `PHASE_0_REPORT.md` and `tests/reference/README.md` first.** They encode the four factual corrections to this plan (preproc shape, rf_dim, non-zero `risk_factor_vector`, batch order) and document subtle behaviors the architecture doc must reflect.
2. Write `docs/architecture.md` containing:
   - A shape table from DICOM input all the way to final calibrated probabilities. Use Phase 0 fixtures as the ground truth for every shape.
   - A dependency graph of the ten or so `nn.Module` subclasses involved at inference.
   - A list of every `if`/`else` branch in those modules that depends on `self.training`, `args.use_pred_*`, `args.survival_analysis_setup`, `args.mask_prob`, etc. For each, the resolution strategy for export (e.g. "set eval mode; trace with `mask_prob=0`; `training` branch is dead").
   - **The eval-mode situation.** Upstream Mirai never calls `.eval()` in `MiraiModel.run_model`. Empirically the outer `MiraiFull` is in train mode (default `nn.Module`), but the snapshot-loaded children (`image_encoder`, `transformer`, `pool`, `prob_of_failure_layer`) end up in eval mode because `torch.load` restores `_training` flags on full pickled objects. Phase 2+ MUST explicitly call `.eval()` on the entire model tree before tracing. Phase 0's `_hooks.py` records the actual training flag of every key submodule at inference (see `module_training_mode` in the captured dict) — refer to it.
   - **The per-key FC double-call.** In `aggregate_and_classify`, every per-key `*_fc` module is invoked twice per inference: (1) inside `RiskFactorPool.forward` with the raw internal-pool hidden — this is what feeds the concatenated `risk_factor_vector`; (2) inside `AllImageTransformer.forward → self.pool.get_pred_rf_loss(hidden, risk_factors)` with the post-relu image hidden, wrapped in `try/except` and silently a no-op because `risk_factors=None` in inference. Any tracing/export that touches this code path must capture the **first** call's output. Phase 0's `_hooks.py` does this with a `if k not in captured` guard.
   - **The XAI-hidden choice (pre-relu vs post-relu).** Document the choice (default post-relu; pre-relu also available as `pool_hidden{,_dcmtk}.npy`). Whichever is chosen drives the validation target in §4 (Phase 3).
3. Decide and document the **export surface**:
   - `image_encoder.onnx`: input `(N, 3, 2048, 1664)` fp32 (CHW: height before width), output `(N, 512)`. `N` is a dynamic axis.
   - `risk_model.onnx`: inputs
     - `img_feats : (B, 4, 512)` fp32
     - `view_seq  : (B, 4)` int64 — values in `{0,1,2}` (CC, MLO, PAD); PAD unused for single-exam inference
     - `side_seq  : (B, 4)` int64 — values in `{0,1,2}` (R, L, PAD); PAD unused for single-exam inference
     - `time_seq  : (B, 4)` int64 — all zeros for a single-exam use case
     - `rf_vector : (B, 100)` fp32 — user-provided RF vector, or all-zero to invoke model-predicted RF path. **`rf_dim=100`; do not hardcode 34 anywhere.** Read at construction from `model.transformer.pool.length_risk_factor_vector`.
     - `rf_known_mask : (B, 100)` fp32 — 1 where `rf_vector` is real, 0 where we want model fill-in
     - outputs
       - `logit : (B, 5)` fp32 — pre-sigmoid cumulative probs
       - `hidden_pre_hazard : (B, 612)` fp32 — **the XAI embedding** (post-relu by default; see Phase 1 decision above)
   - **Slot ordering convention.** Do not hardcode `R-CC, R-MLO, L-CC, L-MLO`. The order is determined by dict insertion in `MiraiModel.run_model` and depends on CLI input order. Phase 0 records the actual order in `tests/reference/fixtures/batch_order.json` (`[(CC,L),(CC,R),(MLO,L),(MLO,R)]` for the demo). The browser caller is responsible for choosing a slot order; the ONNX graph itself is order-agnostic given consistent `view_seq`/`side_seq`.
4. Identify and spell out the minimal code changes needed for a clean export:
   - In `AllImageTransformer.forward`, skip `mask_input` entirely in eval mode (or wrap with `if self.training`).
   - In `RiskFactorPool.forward`, replace the Python `if`/`np.random.random()` logic with a tensor-level blend `rf_used = rf_known_mask * rf_vector + (1 - rf_known_mask) * rf_predicted`. Make sure the export wrapper invokes each per-key FC **once** (matches Phase 0's first-call capture).
   - Return `hidden` as an extra output (or register a hook during tracing that adds an `Identity` node). Decide whether this is the pre-relu or post-relu hidden per the §0.2 decision.
   - Explicitly call `.eval()` on the full model tree before `torch.onnx.export(...)`.
5. Decide on `opset_version` (recommend 17+; `torch.onnx.export` with `dynamo_export` if the PyTorch version supports it cleanly — otherwise fall back to the classic tracer).

**Automated verification.** A lint-style `pytest tests/architecture/test_plan.py` that just checks `docs/architecture.md` exists, contains the required section headings, and lists the exact input/output names we committed to above. (The real verification of the plan is Phase 2+3 succeeding.)

**Manual verification.** Read `docs/architecture.md` top-to-bottom and agree with every decision. In particular, confirm the `hidden_pre_hazard` tensor is what you want for XAI and not e.g. the post-transformer, pre-pool `transformer_hidden`.

**Notes / pitfalls.**
- `RiskFactorPool` *mutates* `self.args.hidden_dim` and sets `self.args.img_only_dim` at construction time. Don't re-instantiate the model after loading weights — you'll desync dimensions.
- `Cumulative_Probability_Layer` registers an `upper_triagular_mask` as a non-trainable parameter; this serializes fine into ONNX as an initializer, but don't let tracing turn it into a constant tensor with a wrong shape if you change `max_followup`.
- `hiddens_transfomer.py` has an import cycle via `onconet.models.pools.factory` — keep that module order sane when extracting for export.
- `MiraiFull.forward` truncates hidden to `[:, :img_repr_dim]` only on the **image-encoder** branch; the risk-model branch keeps the full hidden. Don't apply that slice to the wrong branch when refactoring for export.
- The in-place `self.relu(hidden)` in `aggregate_and_classify` mutates the same storage as the `RiskFactorPool` output. Phase 0's `_hooks.py` defends against this with explicit `.copy()` after `.numpy()`; any export wrapper that splits the pre/post-relu tensors must do the same.

---

## 3. Phase 2 — Export the Image Encoder to ONNX

**Goal.** One `image_encoder.onnx` that, given `(N, 3, 2048, 1664)`, produces the same `(N, 512)` features as the Python model on the Phase 0 fixtures.

**Inputs.** The Phase 0 fixtures `preproc_tensor{,_dcmtk}/*.npy` and `image_encoder_out{,_dcmtk}.npy` (validate against **both** decode paths), plus snapshots referenced in `configs/mirai_trained.json` (the SHA-256s are pinned in `tests/reference/fixtures/MANIFEST.json`).

**Tasks.**
1. Write `scripts/export_image_encoder.py` that:
   - Loads `MiraiFull(args)` with `args.img_encoder_snapshot` and `args.transformer_snapshot` set, but only uses `model.image_encoder`.
   - Unwraps `nn.DataParallel` if present.
   - **Calls `.eval()` on the entire model tree** (upstream Mirai never does this — see §2 / Phase 0 finding) and wraps in a thin `nn.Module` that takes `(N,3,2048,1664)` and returns the 512-d image embedding (i.e. forward through the resnet + pool, then slice `[:, :img_repr_dim]` as `MiraiFull.forward` does).
   - `torch.onnx.export(...)` with:
     - dynamic axes: `{"input": {0: "N"}, "output": {0: "N"}}`
     - `opset_version >= 17`
     - `do_constant_folding=True`
2. Validate with `onnxruntime` (CPU provider first, WebGPU later):

```python
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("image_encoder.onnx")
out = sess.run(None, {"input": preproc_stack_f32})[0]  # (4, 512)
```

**Automated verification.** `pytest tests/onnx/test_image_encoder.py` parametrized over `["pydicom", "dcmtk"]` asserts:
- Output shape is `(4, 512)` for a 4-image batch.
- `np.allclose(out, np.load("image_encoder_out{,_dcmtk}.npy").reshape(4,512), atol=1e-6, rtol=0)` for **same-machine** validation (Phase 0 demonstrated this is achievable). Loosen to `atol=1e-5` only for cross-machine/CI sanity once a Docker reference exists.
- The ONNX file is smaller than 200 MB and passes `onnx.checker.check_model`.

**Manual verification.**
- Open `image_encoder.onnx` in [Netron](https://netron.app/) and confirm the input/output shapes and that the graph contains a ResNet-like structure ending in a GlobalMaxPool or equivalent.
- Confirm no `If` or `Loop` nodes are present (they'd indicate a dynamic Python branch leaked into the graph).

**Notes / pitfalls.**
- The pool layer chosen is `GlobalMaxPool` (from `mirai_trained.json` indirectly — see `custom_resnet` + `mirai_base.json`). Confirm via a forward hook during Phase 0. Don't guess.
- If the tracer complains about `args.use_precomputed_hiddens` or `args.cuda` attributes set on `model._model.args` at load-time (`load_model` in `factory.py` does this), set them explicitly to `False` before export.
- Input normalization (subtract mean, divide by std) is *not* part of the ONNX graph — it lives in the preprocessor. Keep it that way.

---

## 4. Phase 3 — Export the Risk Model to ONNX (with XAI embedding output)

**Goal.** One `risk_model.onnx` that takes the four 512-d image features + positional sequences + risk-factor vector/mask and produces both the 5-year logit *and* the pre-hazard hidden embedding.

**Inputs.** Phase 0 fixtures for `risk_factor_vector{,_dcmtk}.npy` (the **model-predicted** RF tensor — not zeros), `pool_hidden{,_dcmtk}.npy` (pre-relu) and/or `xai_hidden{,_dcmtk}.npy` (post-relu) per the §0.2 decision, `raw_logit{,_dcmtk}.npy`, `pred_risk_factors_per_key{,_dcmtk}/<key>.npy`, the slot ordering from `batch_order{,_dcmtk}.json`, and `image_encoder_out{,_dcmtk}.npy` (used as `img_feats` input).

**Tasks.**
1. Create `onconet/models/mirai_full_export.py` with a new `nn.Module` `RiskModelExport` that:
   - Owns the same `transformer` submodule as `MiraiFull`.
   - In `forward(img_feats, view_seq, side_seq, time_seq, rf_vector, rf_known_mask)`:
     - Constructs `batch = {'view_seq': view_seq, 'side_seq': side_seq, 'time_seq': time_seq}`.
     - Calls a patched version of `AllImageTransformer.forward` that (a) skips `mask_input` under `self.training == False` and (b) returns the `hidden` from `aggregate_and_classify` as well as the logit.
     - **Calls each per-key `*_fc` exactly once** (matches Phase 0's first-call capture; do not reproduce the upstream double-call).
     - In `RiskFactorPool`, replaces the Python branches with the tensor blend:
       ```python
       rf_predicted = torch.cat(pred_risk_factors, dim=1)  # (B, 100)
       rf_used = rf_known_mask * rf_vector + (1 - rf_known_mask) * rf_predicted
       hidden = torch.cat([image_hidden, rf_used], dim=1)  # (B, 612)
       ```
   - Returns `(logit, hidden_pre_hazard)` — `hidden_pre_hazard` shape `(B, 612)`. Pre-relu vs post-relu per the §0.2 / Phase 1 decision.
2. Write `scripts/export_risk_model.py` that instantiates `RiskModelExport` with loaded weights, **explicitly calls `.eval()` on the entire model tree** (upstream Mirai does not), and exports to ONNX:
   - Inputs as in §2, task 3.
   - Outputs: `logit`, `hidden_pre_hazard`.
   - `dynamic_axes` only on the batch axis `B`.
3. Validate numerically against Phase 0 fixtures (parametrized over both pydicom and dcmtk variants):
   - Inputs: `img_feats = np.load("image_encoder_out{,_dcmtk}.npy")` (shape `(1, 4, 512)`); `view_seq`/`side_seq` derived from `batch_order{,_dcmtk}.json` (NOT a hardcoded order — for the demo this happens to be `[(CC,L),(CC,R),(MLO,L),(MLO,R)]` but the source of truth is the JSON); `time_seq = zeros((1, 4))`; `rf_vector = zeros((1, 100))`; `rf_known_mask = zeros((1, 100))`.
   - Expected: `logit ≈ raw_logit{,_dcmtk}.npy` and `hidden_pre_hazard ≈ xai_hidden{,_dcmtk}.npy` (post-relu) OR `pool_hidden{,_dcmtk}.npy` (pre-relu) per the §0.2 decision. Tolerance `atol=1e-6, rtol=0` (same-machine) / `1e-5` (cross-machine).

**Automated verification.** `pytest tests/onnx/test_risk_model.py` parametrized over `["pydicom", "dcmtk"]`:
- Graph passes `onnx.checker.check_model`.
- Running it on the Phase 0 inputs produces outputs within `atol=1e-6` of `raw_logit{,_dcmtk}.npy` and the chosen hidden fixture.
- Running it with `rf_vector = real_values, rf_known_mask = ones` gives a `hidden_pre_hazard` whose last 100 entries are bit-for-bit equal to `rf_vector` if exporting pre-relu, or `relu(rf_vector)` if post-relu (proves the blend works).
- Running it with `rf_vector = zeros, rf_known_mask = zeros` produces a `hidden_pre_hazard` whose last 100 entries equal `risk_factor_vector{,_dcmtk}.npy` (pre-relu) or `relu(risk_factor_vector{,_dcmtk}.npy)` (post-relu). This catches the per-key FC double-call bug if it leaks into export.
- Per-key FC outputs concatenated into the model-predicted RF block must match the Phase 0 `pred_risk_factors_per_key{,_dcmtk}/<key>.npy` files (after sigmoid / softmax per key, per `risk_factor_key_to_num_class`).

**Manual verification.**
- Netron view shows two named outputs, not one.
- `hidden_pre_hazard` has shape `(B, 612)` — i.e. `(B, 512 + 100)`. Confirm `rf_dim=100` matches `model.transformer.pool.length_risk_factor_vector` from the Python model (the Phase 0 manifest also pins this).
- `pred_risk_factors`' per-key FCs are present in the graph (they're used whenever `rf_known_mask < 1`).

**Notes / pitfalls.**
- The positional embedding uses `padding_idx=-1` in PyTorch (`time_embed`, `view_embed`, `side_embed`). ONNX opsets don't fully respect `padding_idx` in `Gather`-based embeddings, so during export verify that the embedding values for `view=2` etc. haven't drifted. If they do, replace with a masked additive embedding.
- `self.prob_of_failure_layer.upper_triagular_mask` is registered as a non-trainable `Parameter` with `requires_grad=False`. This will serialize as an initializer — good. Do *not* reconstruct it in the export wrapper.
- The `.transpose(1,2).unsqueeze(-1)` reshape that turns the transformer's `(B, N, H)` output into an `(B, H, N, 1)` "image-like" hidden feeding the pool is easy to lose when you refactor. Keep it in the export wrapper.

---

## 5. Phase 4 — Extract the Calibrator to a Portable Format

**Goal.** Convert the pickled `MiraiCalibrator` for each of the 5 years into a single JSON file that the TS code can load without any Python runtime.

**Inputs.** The calibrator referenced in `configs/mirai_trained.json`. The exact path and SHA-256 are pinned in `tests/reference/fixtures/MANIFEST.json → snapshots.calibrator` so the export can be reproduced even if the snapshot URL changes.

**Tasks.**
1. Write `scripts/export_calibrator.py` that:
   - Loads the pickle. The Python file uses a dict of `{year_index → MiraiCalibrator}` (see `MiraiModel.process_image_joint`).
   - Serializes to `calibrator.json`:
     ```json
     {
       "years": [
         {"index": 0, "base_slope": ..., "base_offset": ..., "calibrator_slope": ..., "calibrator_offset": ...},
         ...
       ]
     }
     ```
2. Add a Python helper `calibrator_from_json.py` that reads the JSON and applies the formula:
   ```
   y = base_slope * p + base_offset
   y = calibrator_slope * y + calibrator_offset
   p_calibrated = 1 / (1 + exp(y))
   ```
   exactly as in `MiraiCalibrator.predict_proba(..., expand=False)`. This helper is what Phase 5 uses to compose the end-to-end pipeline in Python.

**Automated verification.** `pytest tests/calibrator/test_calibrator_json.py`:
- For each year, generate 100 random probabilities in `(0,1)`, apply the original `MiraiCalibrator.predict_proba`, apply `calibrator_from_json`, assert `np.allclose(..., atol=1e-9)` (this should be essentially bit-exact — same math, fp64).
- On the Phase 0 `raw_sigmoid{,_dcmtk}.npy` (parametrize over both decode paths), the JSON path reproduces `calibrated{,_dcmtk}.npy` within `atol=1e-9` — and the result rounded to 4 decimals matches the pinned demo predictions in §1 (e.g. pydicom Year 1 = 0.0314).

**Manual verification.** Open `calibrator.json`, confirm it has exactly 5 year entries with all four fp scalars populated and no NaNs.

**Notes / pitfalls.**
- The calibrator output is **fp64** (Phase 0 `calibrated{,_dcmtk}.npy` are fp64). The JSON helper, the eventual TS port, and any downstream serialization must preserve double precision — the demo predictions for pydicom and dcmtk paths only diverge starting at the 4th decimal, so casting through fp32 would mask real differences.
- The existing `MiraiCalibrator.predict_proba` returns a `(2,)` array when `expand=True`. Mirror that in both the Python helper and the TS port so the return type is predictable.

---

## 6. Phase 5 — Python-side End-to-End ONNX Pipeline

**Goal.** Prove that the two ONNX files + the JSON calibrator, composed *in Python with `onnxruntime`*, reproduce the full Mirai prediction on the demo DICOMs. This is the last phase before we cross the language barrier — if it passes here, everything broken later is in the TS port, not the ONNX export.

**Inputs.** `image_encoder.onnx` (Phase 2), `risk_model.onnx` (Phase 3), `calibrator.json` (Phase 4), Phase 0 fixtures.

**Tasks.**
1. Write `scripts/run_onnx_pipeline.py` that:
   - Loads the two ONNX models with `ort.InferenceSession(..., providers=['CPUExecutionProvider'])`.
   - Reads the four demo DICOMs, runs them through the *Python* preprocessor (unchanged from Phase 0), stacks into `(4, 3, 2048, 1664)`. Runs once per decode path (pydicom and dcmtk).
   - Runs `image_encoder.onnx` → `(4, 512)`.
   - Builds `view_seq`, `side_seq` from `tests/reference/fixtures/batch_order{,_dcmtk}.json` (do **not** hardcode any order); `time_seq = zeros((1, 4))`; `rf_vector = zeros((1, 100))`, `rf_known_mask = zeros((1, 100))`. The zero mask deliberately invokes the model-predicted RF path — the demo has no user-supplied RFs but the trained config has `use_pred_risk_factors_at_test=true`, so the resulting risk-factor block is non-zero and matches Phase 0's `risk_factor_vector{,_dcmtk}.npy`.
   - Runs `risk_model.onnx` → `(logit, hidden_pre_hazard)`.
   - Applies sigmoid, applies the JSON calibrator per year.
   - Writes `onnx_prediction{,_dcmtk}.json` and `onnx_embedding{,_dcmtk}.npy`.
2. Compare to Phase 0 fixtures (both decode paths).

**Automated verification.** `pytest tests/onnx/test_end_to_end_python.py` parametrized over `["pydicom", "dcmtk"]`:
- `onnx_prediction{,_dcmtk}.json` values are within `atol=1e-4, rtol=1e-3` of `predictions{,_dcmtk}.json` (looser than intermediate tolerances, because the rounding-to-4-dp in `MiraiModel.run_model` already costs us precision — but still tight).
- After 4-decimal rounding, the result must be **bit-equal** to the pinned demo predictions in §1 (pydicom: `0.0314, 0.0505, 0.0711, 0.0935, 0.1052`; dcmtk: `0.0298, 0.0483, 0.0684, 0.09, 0.1016`).
- `onnx_embedding{,_dcmtk}.npy` is within `atol=1e-5` of the chosen hidden fixture (`xai_hidden{,_dcmtk}.npy` post-relu, or `pool_hidden{,_dcmtk}.npy` pre-relu, per the §0.2 decision).

**Manual verification.** Print both JSONs side by side — they should look identical to 4 decimal places. Repeat for the dcmtk pair.

**Notes / pitfalls.** `ort.InferenceSession` is non-deterministic across providers in rare corner cases (esp. reductions). If you see `1e-6` noise, that's expected; if you see `1e-3`, something structural is wrong — check the view/side/time sequence construction first. **The slot ordering is whatever `batch_order{,_dcmtk}.json` says** (for the demo: `[(CC,L),(CC,R),(MLO,L),(MLO,R)]`); the older `R-CC, R-MLO, L-CC, L-MLO` from `csv_mammo_cancer.py` is for training-time CSV ingestion, not inference, and is not what the Python `mirai-predict` actually emits on the demo CLI.

---

## 7. Phase 6 — DICOM → Preprocessed Tensor in TypeScript

**Goal.** A pure-TS `preprocessDicom(file: ArrayBuffer): Float32Array` that reproduces the Python preprocessor (**pydicom path** — the dcmtk variant exists in Phase 0 fixtures only for differential debugging; the browser cannot run dcmtk) to within a tight pixel tolerance on the demo images.

**Inputs.** Phase 0 pydicom fixtures: the raw uint16 arrays (`dicom_raw_uint16/{CC,MLO}_{L,R}.npy`) and the final normalized fp32 tensors (`preproc_tensor/{CC,MLO}_{L,R}.npy`, shape `(3, 2048, 1664)`), per view.

**Tasks.**
1. Pick libraries. I recommend:
   - [`dicom-parser`](https://github.com/cornerstonejs/dicomParser) (Cornerstone) for DICOM parsing — small, battle-tested, no canvas dependency.
   - A small hand-rolled bilinear resampler for the 1664×2048 scale (don't use `<canvas>` — it does sRGB gamma correction on non-8-bit data which you don't want).
2. Implement a `windowing.ts` module that mirrors `onconet/utils/dicom.py:dicom_to_arr` for the pydicom path:
   - `apply_modality_lut` — intercept/slope from `(0028,1052)`/`(0028,1053)`.
   - `apply_voi_lut` when `(0028,3010)` exists (multi-sequence, indexed).
   - VOI type from `(0028,1056)` — `LINEAR` vs `SIGMOID`.
   - Windowing in `LINEAR` mode: exactly the clipping formula from `apply_windowing` in the Python file.
   - `minmax` fallback when no VOI LUT: `center = (min+max+1)/2`, `width = max-min+1`.
3. Implement `view_side.ts` that reads `(0018,5101)` ViewPosition and `(0020,0062)` ImageLaterality and maps to the integer codes `MiraiModel` uses:
   - `view: CC=0, MLO=1` (PAD=2 is unused in single-exam inference and can be omitted from the TS API)
   - `side: R=0, L=1` (PAD=2 unused, see above)
4. Implement `resize.ts` — a bilinear (not Lanczos) resample to `(2048, 1664)` in `(H, W)` order, matching `torchvision.transforms.Resize((height=2048, width=1664))` which uses bilinear by default.
5. Implement `align_to_left.ts`:
   - Compute the sum of pixel intensities in the left and right quarters of the *resized* image.
   - If right > left, horizontally flip.
6. Implement `normalize_and_channel_expand.ts`: grayscale → 3 identical channels, then `(x - 7047.99) / 12005.5`. Output `Float32Array` of length `3 * 2048 * 1664` in CHW order.
7. Wire up `preprocessDicom.ts` that calls all of the above.

**Automated verification.** A Jest/Vitest suite `tests/ts/preprocess.spec.ts`:
- For each demo DICOM, compare `preprocessDicom(buf)` to the corresponding `preproc_tensor/*.npy` fixture.
- Acceptance: `maxAbsDiff < 1e-3` *after* the normalization step (pre-normalization uint16 differences of a couple of units get amplified by `1/std ≈ 8.3e-5`, so this is essentially pixel-perfect).
- The view/side detection returns the same codes as the Python fixture.

**Manual verification.**
- Dump the preprocessed tensor back to a PNG and compare visually to the Python-side PNG. They should be indistinguishable.
- The L CC breast image should face left, same as Python.

**Notes / pitfalls.**
- Endianness: some DICOM transfer syntaxes pack pixels little-endian, others big-endian (rare). `dicom-parser` handles this, but confirm you read the `PixelRepresentation` and `BitsStored` tags correctly.
- Compressed transfer syntaxes (JPEG 2000, JPEG Lossless) are common and `dicom-parser` alone doesn't decode them — you may need `@cornerstonejs/codec-openjpeg` or similar. The demo DICOMs are uncompressed; check with `dcmdump` before assuming.
- The Python pydicom path uses `pillow=True` mode `'I'` (32-bit signed) for the resize input — it preserves full dynamic range. **Don't truncate to uint16 before resizing** in the TS port; Phase 0's pydicom fixtures preserve the full range through the resize step, and matching that is required to hit `1e-3` post-norm tolerance.
- `dcmtk` and `pydicom` agree to the pixel on the demo images *only* for the min-max branch. The two paths' Phase 0 predictions diverge at the 4th decimal (e.g. Year 1: 0.0314 vs 0.0298); the TS port targets the pydicom path, so dcmtk-parity is a non-goal here.

---

## 8. Phase 7 — Risk Factor Vectorizer in TypeScript

**Goal.** A pure-TS `vectorizeRiskFactors(input: Partial<RiskFactors>): { vector: Float32Array, knownMask: Float32Array }` that reproduces `RiskFactorVectorizer.transform` bit-for-bit. The vector has 100 dimensions (`rf_dim=100`), distributed across 34 keys (each key contributes between 1 and ~10 binary/one-hot dims). The key list and per-key dimensionality are pinned in `tests/reference/fixtures/MANIFEST.json` (`risk_factor_keys` and `risk_factor_key_to_num_class`).

**Inputs.** `onconet/utils/risk_factors.py`, plus a small new Python fixture generator that emits ~20 synthetic test cases (patient dicts → expected vector). Note that Phase 0's `pred_risk_factors_per_key{,_dcmtk}/<key>.npy` are **not** the same thing — they are the *model-predicted* per-key tensors, not the *user-supplied* vectorizer output. They share the same 34-key layout, so Phase 7 can borrow the key list and per-key dims for slot-alignment validation.

**Tasks.**
1. Write a Python helper `scripts/generate_rf_fixtures.py` that:
   - Builds ~20 synthetic patient+exam dicts covering edge cases (age unknown, menopause_age > exam_age, BRCA1 positive, various HRT combinations, mom with cancer, etc.).
   - Runs `RiskFactorVectorizer(args).transform(patient, exam)` on each and saves as `tests/rf/fixtures.json`.
2. Port each of the 34 transformer methods from `RiskFactorVectorizer` to TS, preserving the exact output order. Key translations:
   - `one_hot_vectorizor(value, cutoffs)` → a TS util.
   - `get_age_based_risk_factor_transformer` → compare against `exam.age`, return all-zeros if > exam age (the Python code treats it as missing in that case).
   - `transform_brca` → 4-way: `[never_or_unknown, negative, brca1, brca2]`.
   - `transform_menopausal_status` → 4-way: `[pre, peri, post, unknown]`, with the same exam-age logic.
   - `get_hrt_information_transformer('type' | 'duration' | 'years_ago_stopped')` — this one is gnarly; port it carefully and write extra unit tests.
3. Define the `knownMask`: for each output *position*, 1 if the user supplied any of the keys that determine it, 0 otherwise. For "binary" transformers (`binary_biopsy_*`, `brca`, etc.) it's one bit per output. For one-hot age/cut buckets, the whole group shares a known/unknown flag (either you know the age or you don't).
4. Expose a TypeScript interface:
   ```ts
   export interface MiraiRiskFactors {
     age?: number;
     density?: 1 | 2 | 3 | 4;
     binary_family_history?: boolean;
     // ... all 34
     relatives?: {
       M?: Array<{ breast_cancer?: 0|1; ovarian_cancer?: 0|1 }>;
       // ...
     };
   }
   ```
   with every field optional.

**Automated verification.** `tests/ts/rf.spec.ts`:
- For every fixture in `tests/rf/fixtures.json`, TS output matches Python output **bit-for-bit** (`===` on every float, not `allclose` — this is pure dictionary lookup, there's no floating-point nondeterminism).
- `knownMask` has exactly one `1` per output position when the corresponding input is supplied, `0` otherwise.
- `vectorizeRiskFactors({})` (all-unknown input) returns `vector = zeros(100), knownMask = zeros(100)` — this is the exact input shape Phase 3's ONNX risk model expects to invoke the model-predicted-RF path; reproducing the demo predictions in §1 from a `{}` RF input is the end-to-end gate.
- The output slot order matches Phase 0's `MANIFEST.json` `risk_factor_keys` list and per-key `num_class` map.

**Manual verification.**
- Skim the TS to confirm the feature-name order matches `RiskFactorVectorizer.get_feature_names()` exactly. Off-by-one here silently corrupts the model.
- For a fully-populated RF input, inspect the resulting vector in the browser dev tools and sanity-check a handful of fields.

**Notes / pitfalls.**
- `parse_risk_factors` in Python pulls some fields (`5yearcancer`, `prior_hist`, `years_to_cancer`, `bpe`) from a *separate* metadata JSON rather than patient-level RFs. For the browser use case these are all either outputs-of-the-model (`5yearcancer`, `years_to_cancer`) or clinician-provided (`prior_hist`, `bpe`). Only include them in the TS API if the downstream app actually needs to set them.
- The Python code has `MISSING_VALUE = -1` and `TREAT_MISSING_AS_NEGATIVE = False`. Replicate these exactly; changing `TREAT_MISSING_AS_NEGATIVE` is a silent behavior change.

---

## 9. Phase 8 — Calibrator in TypeScript and Browser ONNX Wiring

**Goal.** A `calibrator.ts` and a `runMirai.ts` that, given a `File[]` of four DICOMs plus an optional `MiraiRiskFactors`, returns `{ predictions: { year1..year5 }, embedding: Float32Array, metadata }`. All of it runs in the browser via `onnxruntime-web` + WebGPU.

**Inputs.** Phases 2, 3, 4, 6, 7 outputs.

**Tasks.**
1. Port `MiraiCalibrator.predict_proba` to TS — trivial, one file, ~15 lines:
   ```ts
   function calibrate(p: number, c: CalibratorYear): number {
     let y = c.base_slope * p + c.base_offset;
     y = c.calibrator_slope * y + c.calibrator_offset;
     return 1 / (1 + Math.exp(y));
   }
   ```
2. Set up `onnxruntime-web` with WebGPU:
   ```ts
   import * as ort from 'onnxruntime-web/webgpu';
   ort.env.wasm.numThreads = 4;
   const encoder = await ort.InferenceSession.create('/models/image_encoder.onnx', { executionProviders: ['webgpu'] });
   const risk = await ort.InferenceSession.create('/models/risk_model.onnx', { executionProviders: ['webgpu'] });
   ```
3. Implement `runMirai.ts` that:
   - Preprocesses four DICOMs → `Float32Array` × 4.
   - Stacks into `(4, 3, 2048, 1664)` and feeds `image_encoder.onnx` → `(4, 512)`.
   - Reshapes to `(1, 4, 512)` and feeds `risk_model.onnx` along with the RF vector/mask `(1, 100)` and the view/side/time sequences `(1, 4)`.
   - Returns `{ predictions, embedding: hidden_pre_hazard /* (1, 612) */, raw_logit, raw_sigmoid }`.
4. Expose a second function `getEmbedding(files, riskFactors?)` that is a thin wrapper around the same call and returns just the embedding (so downstream code doesn't pay attention to the prediction outputs if it doesn't need them).

**Automated verification.** `tests/ts/pipeline.spec.ts` (using Vitest + `happy-dom` or Playwright):
- Load the four demo DICOMs as `ArrayBuffer`s in Node (no browser needed for this check — `onnxruntime-web` has a Node-compatible path via `onnxruntime-node` or the WASM backend).
- Run `runMirai(...)`.
- Compare to Phase 0 fixtures: `predictions` within `atol=1e-3` (TS fp32 arithmetic vs Python is subtly different; you'll see 3rd-decimal-place noise), `embedding` within `atol=1e-4`.
- Separately, a Playwright test `tests/e2e/browser.spec.ts` runs the same in an actual Chromium instance with WebGPU enabled and checks the same tolerances.

**Manual verification.**
- Open DevTools, confirm WebGPU is being used (look for `executionProvider: 'webgpu'` and the absence of WASM fallback warnings).
- Inspect the embedding shape in the console; it should be `(1, 612)` — i.e. `(1, 512 + 100)`.
- Run on the four demo DICOMs (`ccl1.dcm`, `ccr1.dcm`, `mlol2.dcm`, `mlor2.dcm`) with no user-supplied RFs; predictions should match Phase 0's pinned pydicom values within ~1e-3: Year 1=0.0314, Year 2=0.0505, Year 3=0.0711, Year 4=0.0935, Year 5=0.1052.
- Run with and without user-supplied risk factors and eyeball that predictions shift in the expected direction (e.g., adding `brca=brca1` should raise risk).

**Notes / pitfalls.**
- `onnxruntime-web`'s WebGPU backend has gaps (as of early 2026 these are shrinking quickly but still exist). If an op is unsupported, it silently falls back to WASM — which is fine functionally but destroys latency. Watch the console.
- Large ONNX files (~100 MB for the image encoder, smaller for the risk model) must be served with proper `Content-Type` and ideally with `Cache-Control: immutable` and brotli/gzip. Don't put them in a service worker cache bucket that gets evicted on every redeploy.
- `ort.Tensor` with `float32` data is zero-copy from `Float32Array`. Don't copy.
- The WASM fallback needs cross-origin isolation (`Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp`) for threading. If your app doesn't have that, it still works single-threaded but is slow.

---

## 10. Phase 9 — End-to-End Validation, API Freeze, and Benchmarks

**Goal.** Declare victory, with evidence.

**Inputs.** Everything.

**Tasks.**
1. Write a one-shot end-to-end test `tests/e2e/parity.spec.ts` that:
   - Runs the Python Mirai pipeline on the four demo DICOMs, captures predictions + embedding.
   - Runs the TS pipeline on the same four DICOMs in a real browser.
   - Asserts predictions within `atol=1e-3` and embedding cosine similarity > 0.9999.
2. Freeze the public TS API in `src/index.ts` with JSDoc:
   - `predictMiraiRisk(files: File[], riskFactors?: Partial<MiraiRiskFactors>): Promise<MiraiResult>`
   - `getMiraiEmbedding(files: File[], riskFactors?: Partial<MiraiRiskFactors>): Promise<Float32Array>`
   - `MiraiResult = { predictions: Record<'year1'|...|'year5', number>, embedding: Float32Array, rawLogit: Float32Array, modelVersion: string }`
3. Add a simple HTML demo page that loads four local DICOMs and shows predictions + a t-SNE-style 2D projection of the embedding (useful for your XAI work).
4. Benchmark on a representative device matrix:
   - M-series Mac, Windows + RTX GPU, mid-range Android Chrome.
   - Record end-to-end latency from `File[]` → predictions (separate: DICOM parse, preprocessing, image_encoder, risk_model, calibration).
5. Document known limitations (compressed DICOMs, non-mammo DICOMs, devices other than Selenia/Selenia Dimensions/etc.).

**Automated verification.** The full CI pipeline runs and all `pytest`/`vitest`/Playwright suites pass.

**Manual verification.** A real user (you, or a colleague) loads the demo DICOMs into the demo page on a fresh machine, sees the expected predictions, and the numbers match what `mirai-predict` gives on the same DICOMs.

---

## 11. Cross-Cutting Infrastructure

These are not phases, but they need to exist from Phase 0 onward.

### 11.1 Reference harness
A single directory `tests/reference/fixtures/` holding every `.npy`/`.json` fixture captured in Phase 0. The authoritative inventory and reproduction commands live in `tests/reference/README.md`. Every phase validates against the same fixtures — phases don't "regenerate" fixtures for each other.

### 11.2 Versioning and reproducibility
`tests/reference/fixtures/MANIFEST.json` already exists (written by Phase 0) and records:
- Git SHA of upstream Mirai at capture time (`4af944449863966a5a9c66b44e56e3c141223897`).
- PyTorch / numpy / pydicom / pillow / dcmtk versions.
- SHA-256 of every snapshot file referenced by `mirai_trained.json` (image encoder, transformer, calibrator).
- SHA-256 / dtype / shape of every `.npy` and `.json` fixture file (per pydicom and dcmtk run).
- Environment fingerprint (platform, python version, python_arch=`x86_64`, hostname).
- Determinism settings used at capture (`torch_set_num_threads=1`, `use_deterministic_algorithms=True`, seeds).
- The list of cross-checks (`asserts_passed`) that held at capture time.

Future phases must treat MANIFEST as the source of truth and re-validate file hashes via `tests/reference/test_baseline.py::test_manifest_file_hashes` whenever they touch fixtures. If any of the recorded SHAs change, the whole fixture set is regenerated and the downstream tests are re-run. Do not paper over small numerical drifts by loosening tolerances.

### 11.3 Agent guardrails
When handing a phase to Claude Code:
- **Read `PHASE_0_REPORT.md` and `tests/reference/README.md` before starting any phase.** The four factual corrections to this plan (preproc shape `(3,2048,1664)`, `rf_dim=100`, non-zero `risk_factor_vector`, batch order from `batch_order.json`) are encoded in those files; the plan reflects them, but the fixtures are the source of truth.
- Pin the tolerance explicitly in the phase's agent prompt (e.g. "must pass `np.allclose(out, fixture, atol=1e-6)` for same-machine, `1e-5` for cross-machine"). Vague "matches closely" drifts.
- Forbid editing any file under `tests/reference/fixtures/`.
- Require the agent to produce a short `PHASE_N_REPORT.md` listing every file it created/modified and the exact commands it ran.

### 11.4 What can go wrong across phases
| Symptom | Likely cause | Where to look |
|---|---|---|
| ONNX encoder/risk-model off by 0.5%+ across the board | Model not in eval mode at export — upstream Mirai never calls `.eval()` in `MiraiModel.run_model` | Phase 1/2/3 — explicitly `.eval()` on the entire tree before export |
| ONNX encoder off by ~1e-3 | BatchNorm running stats not in eval mode (subset of the above) | Phase 2 — confirm `model.eval()` before export |
| Per-key risk-factor outputs match neither sigmoid nor softmax of the visible logits | Forward hook captured the **second** call (post-relu input from `get_pred_rf_loss`) instead of the first | Phase 3 — per-key `*_fc` modules are invoked twice; capture only the first call (Phase 0's `_hooks.py:make_hook` shows the `if k not in captured` pattern) |
| `risk_factor_vector` from ONNX is all zeros while Python has non-zero values | Either you skipped the model-predicted RF path, or you fed `rf_known_mask = ones` (which masks them out) | Phase 3 — for the demo, both `rf_vector` and `rf_known_mask` must be zeros to invoke the model-predicted path |
| Risk model off only when RFs are supplied | `rf_known_mask` not broadcasted the same way as in Python | Phase 3 — shapes of `rf_vector`, `rf_known_mask` |
| Slot 0 holds the wrong (view, side) | Hardcoded `R-CC, R-MLO, L-CC, L-MLO` instead of reading `batch_order.json` | Phases 3/5/8 — the demo's actual order is `[(CC,L),(CC,R),(MLO,L),(MLO,R)]`, driven by CLI input order |
| TS preprocessing off by ~1 unit pre-norm | Bilinear resize vs PIL bilinear | Phase 6 — consider using the exact PIL formula (fp32 intermediate) |
| TS preproc tensor has wrong dimensions | Used `(1664, 2048)` instead of `(2048, 1664)` for resize H,W | Phase 6 — `Resize((height=2048, width=1664))` |
| TS predictions off only for non-GE images | VOI LUT path ported incorrectly | Phase 6 — compare uint16 outputs to fixture for every device type |
| RF vector looks correct but risk prediction is wrong | Feature order mismatched | Phase 7 — diff `get_feature_names()` vs TS order; cross-check against Phase 0 MANIFEST `risk_factor_keys` |
| Browser ONNX 10× slower than expected | WebGPU falling back to WASM for one op | Phase 8 — check `ort.env.webgpu.profilingMode = 'default'` and the console |
| `pip install` fails on Apple Silicon Python 3.8 / torch 1.9.0 | No arm64 wheel; pylibjpeg wheels mislabeled (filename says x86_64 but binary is arm64) | `tests/reference/ENV.md` — use `osx-64` conda env under Rosetta; uninstall all 4 pylibjpeg packages |
| Capture or test reproduces noise at 1e-6 on a different machine | Machine-pinned tolerances; cross-machine drift expected until Docker reference exists | `tests/reference/README.md` — re-capture on the new machine, do not loosen tolerances |

---

## 12. Deliverables Summary

At the end of the plan you will have:

- `image_encoder.onnx`, `risk_model.onnx` — the two ONNX graphs, with the risk model exposing the XAI embedding as a named output.
- `calibrator.json` — 5-year Platt scaling parameters in a language-neutral format.
- `tests/reference/fixtures/` — the golden set of tensors Python and TS must both reproduce.
- `src/mirai/` — the TS package: DICOM preprocessing, risk-factor vectorizer, calibrator, ONNX Runtime wiring, and the two public functions `predictMiraiRisk` and `getMiraiEmbedding`.
- `docs/architecture.md` — the design record.
- `tests/` — Python + TS test suites validating each layer against the frozen fixtures.
- A demo HTML page proving end-to-end in a browser with WebGPU.

Every one of those deliverables has an automated test attached to it that an agent can run as its success criterion, and a manual check that you can run to audit the agent's work.
