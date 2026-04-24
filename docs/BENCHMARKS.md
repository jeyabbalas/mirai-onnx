# Mirai ONNX Web — Benchmarks

Last updated: 2026-04-24 (Phase 9).

Per-stage wall-clock for a full pipeline run on **four 2048×1664×3 fp32 demo DICOMs**, no user-supplied risk factors (model-predicted RF path). Stages are reported as measured by `runMirai`'s `onStage` callback:

- **preprocess**: sum of four `preprocessDicom` calls (DICOM decode → windowing/VOI-LUT → bilinear resize → align-to-left → normalize + 3-channel expand).
- **encoder**: ONNX `image_encoder.onnx` forward, batch=4.
- **risk**: ONNX `risk_model.onnx` forward, batch=1.
- **calibrate**: sigmoid + 5-year Platt scaling.
- **total**: end-to-end wall-clock.

All timings are milliseconds. Baselines are 10-iteration runs (1 warmup dropped). Raw samples are in `artifacts/phase_9/bench_node.json` and `artifacts/phase_9/bench_node.baseline.json`; browser samples are emitted by the demo page's "Download benchmark JSON" button.

## Apple M5 Pro (MacBook Pro), macOS 15.x — Node

Device: Apple M5 Pro · 18 cores · macOS 25.x (Darwin/arm64).
Runtime: Node 25.9.0 (arm64) via `tsx` · `onnxruntime-node@1.19` (CPU EP).
Date: 2026-04-24.

| stage      | p50 (ms) | p95 (ms) | mean (ms) | stddev (ms) | min (ms) | max (ms) |
|------------|---------:|---------:|----------:|------------:|---------:|---------:|
| preprocess |    290.1 |    321.0 |     284.6 |        18.4 |    257.8 |    321.0 |
| encoder    |   2349.2 |   2394.5 |    2348.2 |        28.5 |   2309.6 |   2394.5 |
| risk       |      0.6 |      1.5 |       0.6 |         0.3 |      0.4 |      1.5 |
| calibrate  |      0.1 |      0.1 |       0.1 |         0.0 |      0.0 |      0.1 |
| total      |   2638.2 |   2688.7 |    2637.2 |        31.0 |   2592.0 |   2688.7 |

**Observations.** The image encoder is the overwhelming hot path (~89 % of total); risk + calibrate together are sub-millisecond. Preprocessing is dominated by the 2048×1664 bilinear resize of four 16-bit mammograms (memory-bound on a single JS event loop).

## Apple M5 Pro (MacBook Pro), Chrome 133 — browser (WebGPU)

Run via `npm run demo:dev`, open Chrome, click "Load demo DICOMs" then "Benchmark (10 iters)". Copy the numbers from the demo's timings table or click "Download benchmark JSON".

> _To be filled in by the user during manual verification._

## Apple M5 Pro (MacBook Pro), Chrome 133 — browser (WASM)

Select "WASM only" in the EP dropdown and repeat.

> _To be filled in by the user during manual verification._

## Windows / RTX GPU — browser

> _Deferred. Device unavailable at Phase 9 close. Contributions welcome._

## Android Chrome — browser

> _Deferred. Device unavailable at Phase 9 close. WebGPU is still gated on most Android devices as of 2026 Q2; WASM numbers would dominate._

---

## How to reproduce

```bash
# Node baseline (exits non-zero on >2× regression vs committed baseline)
npm run bench -- --iters 10

# Rotate baseline (e.g. after a real optimization)
npm run bench -- --iters 10 --update-baseline

# Browser (WebGPU or WASM, depending on demo page dropdown)
npm run demo:install
npm run demo:dev
#   open http://localhost:5173
#   → Load demo DICOMs → Benchmark (10 iters)
```
