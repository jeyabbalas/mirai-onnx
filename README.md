# mirai-onnx

Browser/Node-friendly TypeScript pipeline for the [Mirai](https://github.com/reginabarzilaygroup/Mirai) 5-year mammogram-based breast-cancer risk model. Runs the ONNX graphs exported from the original PyTorch model and reproduces Python predictions bit-equal at 4 decimal places.

- **Two ONNX files** — `image_encoder.onnx` (per-view ResNet feature extractor) + `risk_model.onnx` (four-view transformer + risk-factor pool + hazard head, with the post-ReLU 612-dim XAI embedding as a named output).
- **One calibrator JSON** — the 5-year Platt-scaling parameters, language-neutral.
- **Pure-TS preprocessor** — DICOM decode (GE VOI LUT, Explicit VR BE), PIL-exact bilinear resize, align-to-left, normalization. Byte-exact on the decode and resize stages; single fp32 ULP on end-to-end output.
- **Backend-agnostic `runMirai`** — takes a `MiraiSessions` shape that both `onnxruntime-node` and `onnxruntime-web` (WebGPU + WASM) satisfy. No per-backend code in the core pipeline.
- **Privacy-preserving** — everything runs client-side; DICOMs never leave the browser.

Status: Phase 9 complete (2026-04-24). Nine-phase migration log in `PHASE_0_REPORT.md` through `PHASE_9_REPORT.md`.

## Live demo

A hosted build of the browser demo runs at <https://jeyabbalas.github.io/mirai-onnx/>. Click **Load bundled demo DICOMs** to run the end-to-end pipeline — four mammogram DICOMs, ONNX image encoder, risk model, calibration — entirely in the browser. No data leaves your machine. WebGPU on Chrome / Edge / Safari 17+ gives the fastest path; the demo falls back to threaded WASM (enabled via a `coi-serviceworker` that injects COOP/COEP headers) elsewhere.

Deployment is automated from `.github/workflows/deploy-pages.yml`; models and demo DICOMs are downloaded at build time from a pinned GitHub Release (`assets-v0.1.0`).

## Quick start — Node

```bash
# Prereqs: conda envs `mirai-py38` and `mirai-export` per tests/reference/ENV.md;
# models/ + mirai_demo_data/ populated per PHASE_0_REPORT.md + PHASE_2_REPORT.md.

npm install
npm test               # 130 vitest specs, TS ↔ Phase 0 fixture parity
npm run pipeline:ts    # end-to-end Node pipeline, exits 0 on pinned baseline
npm run bench          # 10-iter per-stage benchmark → artifacts/phase_9/
```

Programmatic use:

```ts
import fs from "node:fs";
import {
  runMirai,
  createNodeSessions,
  loadCalibrator,
} from "mirai-onnx-web";
// `loadCalibratorFromFile` is Node-only and lives at a dedicated subpath so
// browser bundlers stay clean. Import it directly when writing Node code:
import { loadCalibratorFromFile } from "mirai-onnx-web/src/mirai/calibrator.node.js";

const sessions = await createNodeSessions({
  encoder: "models/image_encoder.onnx",
  risk:    "models/risk_model.onnx",
});
const calibrator = loadCalibratorFromFile("models/calibrator.json");
const files = ["ccl1.dcm", "ccr1.dcm", "mlol2.dcm", "mlor2.dcm"].map(f =>
  fs.readFileSync(`mirai_demo_data/${f}`),
);
const result = await runMirai(files, sessions, calibrator);
// result.predictions = { year1: 0.0314, ..., year5: 0.1052 }
// result.embedding   = Float32Array(612)  // post-ReLU XAI embedding
// result.slotOrder   = [{view:0, side:1, flipped:true}, ...]
```

## Quick start — browser demo

```bash
npm run demo:install
npm run demo:dev
# open http://localhost:5173, click "Load bundled demo DICOMs" → "Run prediction"
```

The demo page:

- Accepts 4 DICOMs via file picker or drag-drop (or the "Load demo DICOMs" button).
- Optional risk factors: age, BI-RADS density, binary family history.
- Toggles between WebGPU (default, falls back to WASM) and WASM-only EP.
- Renders predictions (rounded + fp64), slot order, a 34×18 heatmap of the 612-dim post-ReLU embedding, and per-stage timings.
- "Benchmark (10 iters)" button runs a per-stage benchmark and reports p50/p95.
- "Download embedding as JSON" saves the 612-dim vector for downstream XAI tooling.

## Public API

```ts
// Pipeline orchestrator.
runMirai(files, sessions, calibrator, riskFactors?, options?): Promise<MiraiResult>;

// Plan-named convenience wrappers (see mirai-migration-plan.md §10.2).
predictMiraiRisk = runMirai;  // alias
getMiraiEmbedding(files, sessions, calibrator, riskFactors?, options?): Promise<Float32Array>;

// Session factories.
createNodeSessions({ encoder, risk }): Promise<MiraiSessions>;              // onnxruntime-node
createWebSessions({ encoder, risk }, { preferWebGPU?, … }): Promise<MiraiSessions>;        // onnxruntime-web
createWebSessionsFromBytes({ encoder, risk }, opts?): Promise<MiraiSessions>;              // pre-fetched bytes

// Individual stages.
preprocessDicom(buffer: ArrayBufferLike | Uint8Array): PreprocessResult;
vectorizeRiskFactors(input?: MiraiRiskFactors): { vector, knownMask, featureNames };
loadCalibrator(json: unknown): Calibrator;
calibrateAll(rawSigmoid, calibrator): Float64Array;
```

See the JSDoc on `src/mirai/index.ts` for detail on every export.

## Further reading

- [`mirai-migration-plan.md`](./mirai-migration-plan.md) — original 10-phase plan.
- [`docs/architecture.md`](./docs/architecture.md) — Phase 1 export design; every tensor shape and control-flow branch.
- [`docs/LIMITATIONS.md`](./docs/LIMITATIONS.md) — DICOM transfer syntaxes, manufacturer coverage, WebGPU matrix, calibrator caveats.
- [`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md) — device-matrix performance.
- [`PHASE_0_REPORT.md`](./PHASE_0_REPORT.md) … [`PHASE_9_REPORT.md`](./PHASE_9_REPORT.md) — per-phase execution logs, deviations, measured tolerances.
- [`CLAUDE.md`](./CLAUDE.md) — pinned facts and tolerances for AI-assisted work.

## License

The original Mirai model is licensed by the authors at [reginabarzilaygroup/Mirai](https://github.com/reginabarzilaygroup/Mirai). This repository's TypeScript port and tooling inherits upstream license terms for the model weights; the TS/ONNX conversion code in this repo is MIT.
