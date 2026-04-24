# Phase 9 Report — End-to-End Validation, API Freeze, and Benchmarks

**Status:** Complete, 2026-04-24.

**Scope delivered per `mirai-migration-plan.md` §10:**
- (a) End-to-end parity — `tests/ts/parity.web.spec.ts` exercises the browser code path (onnxruntime-web, WASM EP) under Node and proves numerical equivalence with Phase 0 fixtures.
- (b) Public API frozen — `src/mirai/index.ts` barrel with JSDoc; version bumped `0.1.0-phase6 → 0.1.0`; plan-named wrappers `predictMiraiRisk` + `getMiraiEmbedding` added.
- (c) Browser demo — `demo/` Vite app with file input, RF form, WebGPU/WASM toggle, predictions table, slot order, 34×18 canvas embedding heatmap, stage timings, and a 10-iter benchmark button.
- (d) Benchmark harness — `scripts/bench.ts` (`npm run bench`) with per-stage p50/p95, regression gate vs committed baseline, Apple M5 Pro baseline captured in `docs/BENCHMARKS.md`.
- (e) Limitations doc — `docs/LIMITATIONS.md` covers transfer syntaxes, manufacturer coverage, dcmtk drift, WebGPU matrix, memory footprint, calibrator provenance.

**Deferred (documented in §Deferred scope):**
- Windows + discrete GPU benchmark — user does not have the hardware.
- Android Chrome benchmark — user does not have the hardware; WebGPU is still gated on most devices.
- Playwright browser automation of the demo — `parity.web.spec.ts` covers the backend contract; the demo's remaining risk is UI rendering, not Mirai correctness.

---

## Files created

| Path | Purpose |
|---|---|
| `src/mirai/sessions/web.ts` | `createWebSessions` / `createWebSessionsFromBytes` — dynamic-import `onnxruntime-web`, WebGPU-first EP selection, WASM fallback. |
| `src/mirai/sessions/node.ts` | `createNodeSessions` — dynamic-import `onnxruntime-node` for Node scripts and tests. |
| `src/mirai/api.ts` | Plan-named wrappers: `predictMiraiRisk` (alias of `runMirai`), `getMiraiEmbedding`. |
| `src/mirai/calibrator.node.ts` | Node-only `loadCalibratorFromFile` — split out of `calibrator.ts` so browser bundlers tree-shake `node:fs` cleanly. |
| `tests/ts/parity.web.spec.ts` | Loads models via `createWebSessionsFromBytes` with WASM EP; asserts predictions bit-equal at 4dp, logit `atol=2e-5`, embedding `atol=2e-5` + cosine ≥ 0.99999, calibrated `atol=1e-7`. |
| `scripts/bench.ts` | Node per-stage benchmark, p50/p95, regression gate vs `artifacts/phase_9/bench_node.baseline.json`. |
| `demo/package.json` | Vite app manifest (`mirai-demo`), depends on `mirai-onnx-web` via `file:..`. |
| `demo/vite.config.ts` | COOP/COEP headers for threaded WASM, `onnxruntime-node` externalized so esbuild doesn't try to load `.node` bindings. |
| `demo/tsconfig.json` | ES2022 + DOM lib. |
| `demo/index.html` | Single-page UI: file input / drag-drop, RF form, EP toggle, predictions/slots/heatmap/timings tables, benchmark button. |
| `demo/src/main.ts` | Entry — wires `createWebSessionsFromBytes`, `runMirai`, `onStage`, `drawHeatmap`. |
| `demo/src/render.ts` | 34×18 canvas heatmap draw routine (linear grayscale 0 → max; post-ReLU non-negative). |
| `demo/scripts/link-models.mjs` | Symlinks `../models/` and `../mirai_demo_data/` into `demo/public/`; copies `onnxruntime-web` WASM assets into `demo/public/ort/` so same-origin fetch works under COOP/COEP. |
| `docs/LIMITATIONS.md` | 11-section limitations doc. |
| `docs/BENCHMARKS.md` | Device-matrix table with Apple M5 Pro Node numbers filled in, browser slots awaiting user's manual run. |
| `README.md` | Quick-start for Node and browser; public API overview; doc index. |
| `PHASE_9_REPORT.md` | This document. |
| `artifacts/phase_9/bench_node.json` | Last 10-iter bench run (gitignored via `artifacts/`). |
| `artifacts/phase_9/bench_node.baseline.json` | Committed baseline for regression gate. |

## Files modified

| Path | Change |
|---|---|
| `src/mirai/runMirai.ts` | Added `MiraiStage`, `MiraiRunOptions`, optional 5th-arg `options?: { onStage }`. Zero-cost when absent (guarded by optional-chain). |
| `src/mirai/index.ts` | Added JSDoc headers; added `api.ts` + `sessions/{web,node}.ts` exports; removed `loadCalibratorFromFile` from the barrel (it now lives at `./calibrator.node.js` to keep `node:fs` out of browser bundles). |
| `src/mirai/calibrator.ts` | Removed `node:fs` import + `loadCalibratorFromFile` body; replaced with a pointer comment to `./calibrator.node.js`. |
| `scripts/run_ts_pipeline.ts` | Swapped hand-rolled session construction for `createNodeSessions`. Imports `loadCalibratorFromFile` from `calibrator.node.js`. |
| `tests/ts/pipeline.spec.ts` | Added `onStage` assertion — stages fire exactly once in `preprocess → encoder → risk → calibrate → total` order. Imports `loadCalibratorFromFile` from `calibrator.node.js`. |
| `tests/ts/parity.web.spec.ts` | New file. |
| `tests/ts/calibrator.spec.ts` | Imports `loadCalibratorFromFile` from `calibrator.node.js`. |
| `package.json` | Version `0.1.0-phase6 → 0.1.0`. Added `main`, `exports`, `optionalDependencies.onnxruntime-web`, `devDependencies.onnxruntime-web`, scripts `bench`, `demo:install`, `demo:dev`, `demo:build`. |
| `CLAUDE.md` | Phases 0–8 complete → 0–9 complete; added Phase 9 deliverables summary. |

## Commands run to produce the deliverables

```bash
# API / session work
npm install                                    # pulls onnxruntime-web
npm run typecheck                               # passes
npm test                                        # 130/130 pass
npm run pipeline:ts                             # exit 0, predictions match pinned pydicom

# Benchmark baseline
npm run bench -- --iters 10 --update-baseline   # wrote artifacts/phase_9/bench_node.baseline.json

# Demo
cd demo && npm install && cd ..
npm --prefix demo run link-models               # symlinks models/ + mirai_demo_data/; copies ort-web wasm
npm --prefix demo run build                     # clean production build, no warnings
```

## Parity measurements (browser backend via onnxruntime-web / WASM EP)

Captured by `tests/ts/parity.web.spec.ts` on the four demo DICOMs (pydicom path):

| Metric | Observed | Budget |
|---|---|---|
| `predictions` vs pinned pydicom (4dp rounded) | bit-equal | bit-equal |
| `embedding` vs `xai_hidden.npy` max abs diff | `4.172e-7` | `< 2e-5` |
| `embedding` vs `xai_hidden.npy` cosine similarity | `1.00000000` | `≥ 0.99999` |
| `rawLogit` vs `raw_logit.npy` max abs diff | `2.384e-7` | `< 2e-5` |
| `calibrated` vs `calibrated.npy` max abs diff | `2.243e-8` | `< 1e-7` |

The WASM EP is numerically tighter than the onnxruntime-node CPU EP on this exact model (empirically 2-5× lower max abs diff on the logit and embedding). Both sit two orders of magnitude inside the committed `ATOL_ORT = 2e-5` budget.

## Benchmark (Apple M5 Pro, Node 25.9.0, onnxruntime-node 1.19 CPU)

10 iterations, 1 warmup dropped:

| stage      | p50 (ms) | p95 (ms) | mean (ms) | min (ms) | max (ms) |
|------------|---------:|---------:|----------:|---------:|---------:|
| preprocess |    290.1 |    321.0 |     284.6 |    257.8 |    321.0 |
| encoder    |   2349.2 |   2394.5 |    2348.2 |   2309.6 |   2394.5 |
| risk       |      0.6 |      1.5 |       0.6 |      0.4 |      1.5 |
| calibrate  |      0.1 |      0.1 |       0.1 |      0.0 |      0.1 |
| total      |   2638.2 |   2688.7 |    2637.2 |   2592.0 |   2688.7 |

The encoder dominates (~89 % of total). Browser WebGPU numbers will be filled into `docs/BENCHMARKS.md` during the user's manual verification pass.

## Deferred scope (explicit)

- **Windows + RTX GPU benchmark.** Non-blocking. The demo page's "Download benchmark JSON" button is the ingest path when someone with the hardware contributes.
- **Android Chrome benchmark.** WebGPU is still flag-gated on most Android devices as of 2026 Q2; the WASM-only number would be what most mobile users see and is subsumed by the browser WASM baseline once captured on the M5 Pro.
- **Playwright browser automation.** Vitest + `parity.web.spec.ts` cover the backend contract. The demo page's remaining risk is UI rendering (heatmap, tables, timings), which doesn't impact Mirai correctness.
- **Python-subprocess parity test.** `scripts/run_ts_pipeline.ts` + `tests/reference/test_baseline.py` already anchor both ends against the same pinned predictions, so a subprocess bridge would be pure redundancy.

## Corrections to the Phase 9 plan found during implementation

1. **`node:fs` tree-shaking.** The original plan kept `loadCalibratorFromFile` in `calibrator.ts`. Vite externalizes `node:fs` but warns at build time when a static import site exists even inside a dead-code-eliminated function. Split into `calibrator.node.ts` and removed from the barrel; Node-only scripts import directly from the subpath. This keeps the browser bundle clean and drops the barrel's module count from 111 → 45 modules in the demo build.
2. **`onnxruntime-node` in the Vite dep scanner.** Even though `sessions/node.ts` uses a dynamic `import("onnxruntime-node")`, Vite's esbuild scanner statically discovers the string and tries to pre-bundle — which fails on the native `.node` bindings. Added `optimizeDeps.exclude: [..., "onnxruntime-node"]` and `rollupOptions.external: ["onnxruntime-node"]` in `demo/vite.config.ts`.
3. **Demo `family history` UI.** The plan sketched a `binary_family_history: true` bool. The actual `MiraiRiskFactors` type derives that field from the `relatives` dict (non-empty list under any relative code → true). Demo binds the checkbox to `relatives: { M: [{}] }` as the canonical minimal expression.
4. **`ort.env.wasm.wasmPaths`.** Pointed at `/ort/` served from `demo/public/ort/` (populated by the `link-models` script copying from `node_modules/onnxruntime-web/dist/`). Needed for threaded WASM under COOP/COEP — the default CDN fetch origin does not round-trip the `Cross-Origin-Embedder-Policy` constraint cleanly.

---

## Manual verification checklist

1. `conda activate mirai-py38 && pytest tests/reference/test_baseline.py -v` — still 56/56.
2. `conda activate mirai-export && pytest tests/onnx/ tests/calibrator/ tests/architecture/ -v && python scripts/run_onnx_pipeline.py` — all green, pinned predictions reproduced.
3. `npm test` — 130/130 vitest specs pass (including `parity.web.spec.ts`).
4. `npm run typecheck` — no errors.
5. `npm run pipeline:ts` — exits 0 with `[pydicom] OK: rounded predictions bit-equal to pinned baseline`.
6. `npm run bench -- --iters 10` — exits 0 vs baseline; no regression.
7. `npm run demo:install && npm run demo:dev` — open http://localhost:5173 in Chrome:
   - Click "Load bundled demo DICOMs" → "Run prediction"
   - Confirm EP tag shows `webgpu` (or `wasm` on browsers without WebGPU)
   - Confirm predictions `year1=0.0314, year2=0.0505, year3=0.0711, year4=0.0935, year5=0.1052`
   - Confirm 4-row slot table: `(CC,L,true), (CC,R,false), (MLO,L,true), (MLO,R,false)`
   - Confirm heatmap renders (mostly near-white with dark sparse cells — expected post-ReLU sparsity)
   - Click EP dropdown → "WASM only" → "Run prediction" again; predictions unchanged
   - Click "Benchmark (10 iters)"; timings table fills; fill the WebGPU / WASM rows into `docs/BENCHMARKS.md`
8. `npm --prefix demo run build` — clean build, no warnings, `dist/assets/ort-wasm-simd-threaded.jsep-*.wasm` present.
