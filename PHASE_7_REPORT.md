# Phase 7 Report — TypeScript RiskFactorVectorizer

**Status**: implementation + tests complete, not yet committed.
**Date**: 2026-04-23.
**Scope**: port Python `RiskFactorVectorizer` (`external/Mirai/onconet/utils/risk_factors.py`) to TypeScript, producing `{ rf_vector : Float32Array(100), rf_known_mask : Float32Array(100) }` from an optional `MiraiRiskFactors` user input — the two tensors that Phase 3's `risk_model.onnx` consumes via the blend `rf_used = rf_known_mask * rf_vector + (1 - rf_known_mask) * rf_predicted`.

## Files created

### Python (oracle)
- `scripts/generate_rf_fixtures.py` — drives the real `RiskFactorVectorizer` under `mirai-py38` with `parse_risk_factors` monkey-patched, emits `tests/rf/fixtures.json`. 39 synthetic cases covering every transformer branch. No model snapshots or metadata JSON files required.
- `tests/rf/fixtures.json` — 39 cases × 100-dim vector + 100-dim mask + feature-name manifest. Produced fresh by the script; reproducible.

### TypeScript
- `src/mirai/riskFactors/`:
  - `index.ts` — public barrel. Exports `vectorizeRiskFactors`, `FEATURE_NAMES`, `RISK_FACTOR_KEYS`, `RF_KEY_TO_NUM_CLASS`, `RF_KEY_TO_OFFSET`, `RF_DIM`, and all public types.
  - `keys.ts` — the 34 keys in inference-config order, num-class map, cumulative offsets, `RF_DIM = 100`.
  - `types.ts` — `MiraiRiskFactors` (flat user-facing API), `VectorizerResult`, internal `PatientFactors`/`ExamFactors` dicts that mirror the Python layout.
  - `missing.ts` — `MISSING_VALUE = -1` + `coerceInt()` (boolean/number/null/undefined → int, missing → -1).
  - `oneHot.ts` — shared `oneHotInto(value, cutoffs, out, offset)` + `oneHotFeatureNames(...)` matching Python's `one_hot_vectorizor` / `one_hot_feature_names`.
  - `factors.ts` — `buildInternalFactors(input)` (public API → internal dicts) and `deriveKnownByKey(input)` (per-key presence rules).
  - `vectorizer.ts` — orchestrator. Builds `vector`/`knownMask` once per call, dispatches to per-key transformers, stamps mask in whole-key blocks, returns `FEATURE_NAMES`.
  - `transformers/` (13 files) — one module per Python transformer family: `imageBiomarker.ts` (density), `examOneHot.ts` (age/weight/height), `ageBased.ts` (menarche/menopause/first-pregnancy/ovarian-cancer-age with the `exam_age < age_based_risk_factor → MISSING` guard), `binaryOccurrence.ts` (biopsy_* + ovarian_cancer), `binary.ts` (prior_hist, ashkenazi), `relative.ts` (6 relatives × 2 cancers), `familyHistory.ts`, `parous.ts`, `race.ts`, `menopausalStatus.ts` (pre/peri/post/unknown with `NEGATIVE_99` sentinel), `brca.ts`, `hrt.ts` (full three-piece state machine, ported verbatim from lines 263-329).

### Tests
- `tests/ts/riskFactors.spec.ts` — fixture-driven bit-for-bit parity (`===`) on every slot of both `vector` and `knownMask` across all 39 cases + fixture invariant assertions (schema, key order, num-class map, offsets, feature names).
- `tests/ts/riskFactors.integration.spec.ts` — RF constants sanity checks, empty-input zeroness, fresh-allocation independence, targeted slot probes (brca2 → 74, age=45 → 9, prior_hist true/false → 30).

## Decisions pinned in Phase 7

### 1. Keys list: trained config, not `parse_args` default

`mirai_trained.json` declares 34 `risk_factor_keys`. `parse_args`' default list (`parsing.py:258`) adds 4 extras (`brother_bc_cancer_history`, `father_bc_cancer_history`, `daughter_bc_cancer_history`, `daughter_oc_cancer_history`) that are not used at inference time. The TS port implements the 34-key list only. Sum of widths = 100 (`RF_DIM`).

### 2. Excluded transformers

`bpe`, `5yearcancer`, `years_to_cancer` are defined in `risk_factors.py` but not in the inference `risk_factor_keys`. The TS port does not implement them — they would be dead code.

### 3. Known-mask granularity: per-key, whole-block

The ONNX graph performs an element-wise blend (`scripts/export_risk_model.py:142`), so per-slot and per-group masking are mathematically equivalent *as long as* the vector's block is zero where mask is zero. Phase 7 enforces "per-key" at both the mask level (a whole `num_class` block is all-1 or all-0) and the vector level (block is zeroed unless the key is known). This cleanly decouples "unknown → fall back to model-predicted" from any latent Python quirks in the transformers' missing-value branches.

### 4. Zero-out on unknown — avoids two Python quirks leaking

Fixture generator and TS both zero the per-key block when `known=false`. This avoids two latent Python bugs showing up in the user vector:

- `get_binary_occurence_transformer` uses `if occurence:` which is truthy for `-1` — on missing input, biopsy slots would otherwise flip to `1`.
- `transform_race` does `race_vector[race-1] = 1` with no bounds check — on `race = -1`, Python wraps to `vector[-2]` and sets the "Hawaiian" slot to 1.

Neither quirk affects inference (mask=0 → ONNX substitutes predicted). But the TS port should not ship semantically-wrong values; fixture generator matches by zeroing the block when the mask block is zero (see `vector_from_python` in `scripts/generate_rf_fixtures.py`).

### 5. Public TS API is flat; internal mirror-of-Python dicts

The user-facing type `MiraiRiskFactors` collapses Python's patient/exam split into a single flat optional-field object, with explicit `age`, `density`, `weight`, `height`, `priorHist`, etc. Inside the vectorizer, `buildInternalFactors` constructs `{patient, exam}` dicts whose fields match `risk_factors.py` verbatim, so each transformer is a direct line-for-line port of its Python counterpart.

### 6. HRT transformer semantics preserved exactly

`transformers/hrt.ts:deriveHrtState` mirrors `get_hrt_information_transformer` (lines 263-329) branch-for-branch, including:
- the `last_age >= current_age and current_age != MISSING_VALUE` on-going guard,
- the two future-user branches (`first_age > current_age` and `last_age - extracted_duration > current_age`),
- the assertion `duration >= 0` (raised as a thrown `Error` on the TS side).

The three HRT output keys (`hrt_type`, `hrt_duration`, `hrt_years_ago_stopped`) share the state via a single `deriveHrtState` call per `vectorizeRiskFactors` invocation.

## Parity measurements

All 39 fixture cases pass with strict `===` equality on both `vector` and `knownMask`. Every slot of every case is a 0 or 1 integer-valued `Float32Array` element, so there is no floating-point rounding slack — this is bit-exact parity with the Python reference.

| Suite | Result |
|---|---|
| `npm run typecheck` | clean, zero errors |
| `npm test` (full TS suite, 7 files, Phase 6 + Phase 7) | 116 / 116 pass |
| `tests/ts/riskFactors.spec.ts` (fixture parity) | 83 / 83 pass |
| `tests/ts/riskFactors.integration.spec.ts` (invariants + probes) | 11 / 11 pass |
| `pytest tests/reference/test_baseline.py` (Phase 0 regression) | 56 / 56 pass (≈50 s) |

## What Phase 7 deliberately did not touch

- `tests/reference/fixtures/**` — not a single file changed. The 34 `.npy` files under `pred_risk_factors_per_key{,_dcmtk}/` are *model-predicted* outputs used on the other side of the ONNX blend; they are not the vectorizer's oracle and Phase 7 does not validate against them.
- `external/Mirai/**` — untouched.
- `src/mirai/types.ts` — kept minimal; all new types live in `src/mirai/riskFactors/types.ts`.
- `src/mirai/preprocess/**` — untouched (Phase 6 implementation).
- `docs/architecture.md` — not updated in this phase; the Phase 3 ONNX I/O spec is still authoritative for the `rf_vector`/`rf_known_mask` contract.

## What Phase 8 can assume

- `vectorizeRiskFactors(input?)` returns `{ vector, knownMask, featureNames }` where both tensors are `Float32Array(100)`, ready to feed `risk_model.onnx` as `rf_vector` and `rf_known_mask` (with a leading batch dim of 1 added by the caller).
- `vectorizeRiskFactors({})` returns two all-zero arrays, which is exactly the input Phase 5 used to reproduce the pinned demo predictions (pydicom Year 1 = 0.0314). Phase 8 can use this as the end-to-end gate.
- `FEATURE_NAMES[i]` is the human-readable label for slot `i` (matches Python's `RiskFactorVectorizer.get_feature_names()` verbatim). Useful for debug UI.
- `RF_KEY_TO_OFFSET[k]` is the cumulative slot offset for key `k`, and `RF_KEY_TO_NUM_CLASS[k]` is its width. Downstream code can reconstruct per-key views without hardcoding offsets.

## Reproduction

```bash
# Generate fixtures (one-time; commit the resulting tests/rf/fixtures.json).
conda activate mirai-py38
python scripts/generate_rf_fixtures.py   # writes tests/rf/fixtures.json

# Validate TS port.
npm run typecheck
npm test
```

## Open items / stretch goals for Phase 8

- Wire `vectorizeRiskFactors({})` into `scripts/run_onnx_pipeline.py` (Phase 5) or a browser-side probe and confirm end-to-end prediction reproducibility against the pinned pydicom demo predictions. Not strictly Phase 7 scope; documented as the integration gate for Phase 8.
- The Phase 8 browser wrapper will need to `new Float32Array(1, 100)`-reshape these to `(1, 100)` before feeding ORT; `onnxruntime-web`'s `ort.Tensor(type, data, dims)` accepts the existing `Float32Array` zero-copy.
