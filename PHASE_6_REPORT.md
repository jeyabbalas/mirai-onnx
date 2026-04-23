# Phase 6 Report — DICOM → Preprocessed Tensor in TypeScript

**Status:** complete (2026-04-23).

Phase 6 of `mirai-migration-plan.md` — ports the Python/PyTorch DICOM preprocessor to pure TypeScript (`src/mirai/preprocess/`) and proves end-to-end parity against the Phase 0 `preproc_tensor*` fixtures on all 4 demo DICOMs. The TS output matches Phase 0 to **~4.77e-7** (single-ULP fp32 noise), **~4000× inside the plan's `atol=1e-3` budget**.

## Files added

| Path | Purpose |
|---|---|
| `package.json` | npm manifest (private, `"type": "module"`). Deps: `dicom-parser@^1.8.21`. DevDeps: `typescript@^5.4`, `vitest@^1.6`, `@types/node@^20.11`. |
| `tsconfig.json` | Strict mode, ES2022, `moduleResolution: "bundler"`, `noUnused*` on. |
| `vitest.config.ts` | Tests glob `tests/ts/**/*.spec.ts`, 30 s per-test timeout. |
| `.nvmrc` | `20` — LTS target, forward-compatible with Node 25. |
| `package-lock.json` | Reproducible install. |
| `src/mirai/types.ts` | Public types: `View`, `Side`, `PreprocessResult`. |
| `src/mirai/util/npy.ts` | Minimal `.npy` v1/v3 parser for fixture tests (uint16, int32, float32, float64). |
| `src/mirai/preprocess/viewSide.ts` | Reads `(0018,5101)` + `(0020,0062)` → `(view, side)` codes. Mirrors `onconet/utils/dicom.py:187`. |
| `src/mirai/preprocess/windowing.ts` | `apply_windowing` LINEAR + SIGMOID and float64→uint16 truncation cast. Mirrors `dicom.py:18`. |
| `src/mirai/preprocess/voiLut.ts` | Reads and applies the VOI LUT Sequence `(0028,3010)`; includes the Mirai-specific `scaleToSixteenBit` bit-shift with uint16 wraparound semantics. |
| `src/mirai/preprocess/dicom.ts` | Orchestrates decode → modality LUT → VOI LUT (GE) or windowing (auto/minmax) → uint16 cast. |
| `src/mirai/preprocess/resize.ts` | Port of PIL `BILINEAR` resample for mode-'I' images from `libImaging/Resample.c` (9.0.0). `precomputeCoeffs` + separable horizontal/vertical passes, fp64 accumulation, `ROUND_UP` rounding. |
| `src/mirai/preprocess/alignToLeft.ts` | Quartile-sum comparison + horizontal flip. Mirrors `onconet/transformers/image.py:314`. |
| `src/mirai/preprocess/normalize.ts` | ToTensor (no `/255`) + `.expand(3, H, W)` materialization + per-channel `(x - μ) / σ`. |
| `src/mirai/preprocess/index.ts` | Public `preprocessDicom(buffer, opts)` pipeline wrapper. |
| `tests/ts/setup.ts` | Shared paths + pinned `DEMO_DICOMS` table with expected (view, side) codes. |
| `tests/ts/{smoke,npy,dicom,resize,preprocess}.spec.ts` | 5 spec files, 22 tests total. |
| `scripts/capture_post_resize_fixture.py` | New Python helper — isolates the PIL bilinear step by running `Scale_2d`'s torchvision.Resize on Phase 0's `dicom_raw_uint16*` arrays, writing int32 `(2048, 1664)` `.npy` files. Runs in `mirai-py38`. |
| `tests/reference/fixtures/post_resize/{CC,MLO}_{L,R}.npy` | New pydicom-path resize fixtures (int32). |
| `tests/reference/fixtures/post_resize_dcmtk/{CC,MLO}_{L,R}.npy` | New dcmtk-path resize fixtures (captured for completeness; not referenced by Phase 6 tests). |
| `tests/reference/fixtures/MANIFEST.json` | Appended 8 `post_resize*` entries with SHA-256 + dtype + shape. No existing entries were edited. |
| `.gitignore` | Added `node_modules/` and `dist/`. |
| `PHASE_6_REPORT.md` | This file. |

## Correction to the Phase 6 plan

The plan (`mirai-migration-plan.md` §7) and my initial approved plan both assumed the demo DICOMs hit the `minmax` windowing branch. They don't.

Empirical finding after probing the actual DICOMs:

- Manufacturer: `GE MEDICAL SYSTEMS`
- `(0028,3010)` VOI LUT Sequence present (3 items: NORMAL, HARDER, SOFTER)
- Transfer syntax: `1.2.840.10008.1.2.2` — **Explicit VR Big Endian**
- Bits allocated: 16, bits stored: 12, pixel representation: 0
- The `'GE' in manufacturer and (0x28,0x3010) in dicom` guard on `dicom.py:128` takes the GE branch, **not** the minmax branch.

Implementation follows: `dicom.ts` dispatches to `readVoiLutSequence` + `applyVoiLut` + `scaleToSixteenBit` for GE-with-VOILUT-sequence DICOMs; the minmax and auto branches are also implemented (for future non-GE DICOMs) but are not exercised by Phase 6 tests. The `usedBranch` field on the decode result confirms which path ran.

`readPixelDataUint16` also handles both endians correctly (big-endian for the demo TS, little-endian for the other two supported transfer syntaxes).

## Environment

Two environments, neither arm64 and neither affected by the Phase 0 Rosetta constraints.

**Node (new this phase):**

- Node 25.9.0 (current arm64 on this host; `.nvmrc` pins `20` LTS as the baseline).
- npm 11.12.1.
- `dicom-parser` 1.8.21, `typescript` 5.9.2, `vitest` 1.6.1, `@types/node` 20.19.9.
- `npm audit --omit=dev` reports 0 vulnerabilities in runtime deps. (The 4 moderate dev-only vulnerabilities are transitive through Vitest's tooling and do not ship.)

**Python (reused):** `mirai-py38` conda env (`osx-64`, Pillow 9.0.0, torchvision 0.10.0) — used only to capture the `post_resize*` fixtures and to re-run Phase 0's baseline.

## Commands run

```bash
# Phase 6 TS pipeline
npm install
npm run typecheck           # clean
npm test                    # 22 passed (5 spec files)

# Regenerate post_resize fixtures (one-time)
conda activate mirai-py38
python scripts/capture_post_resize_fixture.py

# Confirm Phase 0 baseline unaffected
pytest tests/reference/test_baseline.py -v   # 56 passed in 50.13s
```

All three suites pass; no regressions.

## Parity measurements

Empirical max absolute differences (no cherry-picking — reproduced by `npm test -- --reporter=verbose`).

### Per-stage, vs Phase 0 fixtures (pydicom path, all 4 demo DICOMs)

| Stage | Compared to | Budget | Measured |
|---|---|---:|---:|
| `decodeDicom` uint16 | `dicom_raw_uint16/*.npy` | bit-exact | **0 mismatched pixels / 7 330 428** |
| `readViewSide` | `batch_order.json` | exact | all 4 match |
| `resizeBilinearMode1` int32 | `post_resize/*.npy` (new) | ≤ 2 LSB | **0 mismatched pixels / 3 407 872** on all 4 DICOMs |
| `preprocessDicom` fp32 end-to-end | `preproc_tensor/*.npy` | atol 1e-3 | **max 4.768e-7**, RMS 3.2–4.0e-8 on all 4 DICOMs |

The resize and DICOM-decode stages are **fully bit-exact** to the Python reference. The only drift is in the normalize step's fp32 rounding of `(x - 7047.99) / 12005.5`, which is single-ULP noise.

### End-to-end detail (pydicom)

| DICOM | flipped | maxAbsDiff | RMS | worst idx |
|---|---|---:|---:|---:|
| CC_L  | true  | 4.768e-7 | 3.779e-8 | 1345163 |
| CC_R  | false | 4.768e-7 | 4.008e-8 | 1689347 |
| MLO_L | true  | 4.768e-7 | 3.219e-8 | 1105602 |
| MLO_R | false | 4.768e-7 | 3.199e-8 | 1227116 |

`flipped=true` for both L views and `flipped=false` for both R views, matching the expected "align so chest wall is on the left" semantics.

## Fixture provenance

New `.npy` files under `tests/reference/fixtures/` — all int32, shape `(2048, 1664)`.

| Path | SHA-256 |
|---|---|
| `post_resize/CC_L.npy` | `51a89880f7ae68ed7e88bd43057980709ed48400cf3b22815ff47042532f95fc` |
| `post_resize/CC_R.npy` | `1cb1950325d8410a97fed4224e69bc4df4618c1da0e10019e5ed4382b9f30764` |
| `post_resize/MLO_L.npy` | `adfc8f9fe8e7e99f2a507f2acd5f544edbcc9460156718847130450f68fc4120` |
| `post_resize/MLO_R.npy` | `209afd60e23827cdd761972365a5ed9b63ac5a3e787724aea00f447e3291f8ed` |
| `post_resize_dcmtk/CC_L.npy` | `3eae892cb55987f5135ef29c1925b4df7530b47e690b5659904c611f461d6179` |
| `post_resize_dcmtk/CC_R.npy` | `91ec6fa1ba2229819ed82788dc21a6785d87f61e9fd641d2073de25882d14e68` |
| `post_resize_dcmtk/MLO_L.npy` | `a7fa5ba12f48ce50653ed037b2884355803986a478a0ea43d68d3660f1def2b3` |
| `post_resize_dcmtk/MLO_R.npy` | `6f7ab2a11dc38121ed46328e005b4d83653dfb8c8fa7cdc4d8451a71e0a49b89` |

Reproduction: `conda activate mirai-py38 && python scripts/capture_post_resize_fixture.py`. The script also rewrites the MANIFEST block, idempotent on re-run.

## Known limitations (scope deferred)

None block the Phase 6 exit criteria. All are explicitly out-of-scope per the migration plan.

- **Compressed DICOMs.** `dicom.ts:readPixelDataUint16` throws on `encapsulatedPixelData`. JPEG 2000 / JPEG Lossless / RLE decoders (e.g. `@cornerstonejs/codec-openjpeg`) can be added at call sites when Phase 8 hits a compressed scanner.
- **PixelRepresentation = 1 (signed).** Demo DICOMs are unsigned; signed path throws explicitly.
- **BitsAllocated ≠ 16.** Mammography is almost exclusively 16-bit; other depths throw.
- **Non-GE VOI LUT Sequence.** The GE branch is gated on `manufacturer.includes("GE")`. Other vendors with VOI LUT sequences would hit the windowing branches instead, matching upstream behavior.
- **TypeScript DataParallel / batching.** `preprocessDicom` is per-image. Phase 8 will stack 4 results into `(4, 3, 2048, 1664)` for the ONNX encoder input.

## Deviations from the approved plan

1. **GE VOI LUT branch replaces minmax as the active path for the demo** — discovered pre-implementation and recorded in the "Correction" section above. The minmax and auto branches are still implemented, just unused by the demo tests. Tolerances and gates unchanged.
2. **Resize drift under budget.** Plan budgeted ≤ 2 LSB vs the new `post_resize` fixture; measured **0**. No code changes required.
3. **End-to-end tolerance.** Plan budgeted `atol < 1e-3`; measured **4.768e-7** (one-ULP fp32). No code changes required.
4. **Removed `docs/architecture.md` addendum from this phase.** The architecture doc is currently ONNX-centric; the TS pipeline is documented here and in-file JSDoc. Can be lifted into `docs/architecture.md` in Phase 8 alongside the browser-side wiring without re-running Phase 6.

## What Phase 6 does NOT yet prove

Intentional non-goals owned by later phases:

- **Browser execution.** Phase 6 runs in Node. Phase 8 wires `preprocessDicom` + `onnxruntime-web` + WebGPU.
- **DICOMs outside the 4 demos.** No multi-vendor / multi-bit-depth / compressed-transfer-syntax coverage. The scope is single-vendor GE + uncompressed Explicit VR Big Endian + 16-bit unsigned, which matches the demo set and the migration plan's Phase 6 scope.
- **Risk-factor vectorizer.** Phase 7.
- **Calibrator in TS.** Phase 8 (trivial port of `scripts/export_calibrator.py`'s JSON output).

## Next step

Phase 7 — TypeScript port of `RiskFactorVectorizer` (34 keys → 100-dim vector + known-mask). No overlap with Phase 6's preprocessor; independent module under `src/mirai/riskFactors/`.
