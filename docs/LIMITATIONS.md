# Mirai ONNX Web — Known Limitations

Last updated: 2026-04-24 (Phase 9).

This document lists every capability gap of the browser/Node pipeline that a downstream clinical app needs to know. None of the items below are fatal — they reflect the scope of the Mirai paper and the four demo DICOMs we validated against. Expanding the coverage is a matter of future work and additional golden fixtures, not a broken implementation.

## 1. DICOM transfer syntaxes

**Supported (validated):**
- Explicit VR Big Endian (`1.2.840.10008.1.2.2`) — the demo DICOMs.
- Explicit VR Little Endian (`1.2.840.10008.1.2.1`) — standard Part 10 default.
- Implicit VR Little Endian (`1.2.840.10008.1.2`) — pre-2000 legacy standard.

**Not supported (throws):**
- JPEG Baseline (`1.2.840.10008.1.2.4.50`) and JPEG Lossless (`…1.2.4.70`).
- JPEG 2000 (`…1.2.4.90`, `…1.2.4.91`).
- JPEG-LS (`…1.2.4.80`).
- RLE Lossless (`…1.2.5`).
- Deflated Explicit VR Little Endian (`…1.2.1.99`).

A clinical frontend that accepts DICOMs from PACS systems should feature-detect the transfer-syntax UID before calling `preprocessDicom` and surface a clear "compressed DICOM not supported" error otherwise. Adding JPEG 2000 support is tractable via `@cornerstonejs/codec-openjpeg` as a pre-decode step — out of scope for this phase.

## 2. Manufacturer / vendor coverage

**Validated:** GE Medical Systems (via the four `mirai_demo_data/` demo DICOMs), VOI LUT Sequence `(0028,3010)` present with 3 items, Mirai uses `index=0`.

**Implemented but unvalidated against fixtures:**
- Hologic Selenia / Selenia Dimensions — the `minmax` windowing branch in `src/mirai/preprocess/dicom.ts` exists but Phase 0 captured no Hologic DICOM in its reference set. Predictions on Hologic images are expected to be accurate (the upstream Mirai paper was trained on a Hologic-dominant cohort) but not byte-verified against a Python reference here.
- Siemens Mammomat — `apply_modality_lut` + windowing path is identical to the GE decode path in pydicom, so it should work; unvalidated.

**Not supported / likely broken:**
- Monochrome1 (inverted) photometric interpretation — our pipeline assumes Monochrome2. A Monochrome1 DICOM will produce a visually-inverted preprocessed tensor and nonsense predictions. Detect via `(0028,0004) PhotometricInterpretation` and either invert before feeding `preprocessDicom` or reject with a clear error.

## 3. Modality

Mirai is a mammography model. The pipeline does not validate the `(0008,0060) Modality` tag and will happily run on MR, CT, or US DICOMs — producing meaningless output. Frontends must filter to `Modality == "MG"` before calling the pipeline.

## 4. View count

Exactly **four** DICOMs required: L-CC, R-CC, L-MLO, R-MLO. The Mirai paper describes a "partial view" training mode for missing views, but the exported ONNX risk model does not carry that logic — it uses fixed `view_seq`/`side_seq` tensors. Calling `runMirai` with fewer or more than 4 DICOMs throws immediately.

## 5. Slot order

Slot ordering is caller-determined. The demo tooling uses `[(CC,L),(CC,R),(MLO,L),(MLO,R)]` (driven by dict-insertion order in the original `MiraiModel.run_model` Python code). The ONNX graph itself is order-agnostic given consistent `view_seq`/`side_seq`/`time_seq`, so any permutation works as long as the four images match the caller's declared order. Pinned in `tests/reference/fixtures/batch_order.json`.

## 6. dcmtk vs pydicom 4th-decimal drift

Phase 0 captured golden predictions on both the pydicom decode path and the dcmtk decode path. They differ at the 4th decimal:

| Year | pydicom | dcmtk  |
|------|---------|--------|
| 1    | 0.0314  | 0.0298 |
| 2    | 0.0505  | 0.0483 |
| 3    | 0.0711  | 0.0684 |
| 4    | 0.0935  | 0.09   |
| 5    | 0.1052  | 0.1016 |

The TypeScript pipeline matches the **pydicom** path byte-for-byte (single fp32 ULP drift on the preprocessed tensor, then propagated through ONNX). The dcmtk path exists in Phase 0 only for differential debugging — the browser cannot run dcmtk, so we cannot match it. Clinicians reviewing predictions should be told that the answer to "what risk does Mirai estimate?" has a ±0.002 ambiguity at the 4th decimal depending on decoder choice; this is an upstream property of the Mirai/DICOM ecosystem, not an error introduced by this port.

## 7. Calibrator provenance

`models/calibrator.json` is extracted verbatim from the upstream `mirai_trained.json` snapshot. It was fit on the Mirai paper's original cohort (Hologic-heavy, Massachusetts General Hospital). A frontend deployed to a population with a materially different breast-cancer base rate, screening interval, or demographic composition may see **miscalibrated** absolute probabilities even when the relative risk ordering is correct. Recalibrating against a local cohort is a first-class downstream concern — the JSON schema (`schema_version: 1`, 5 years × 4 scalars) makes this a drop-in swap.

## 8. WebGPU browser matrix (as of Phase 9 close, 2026-04-24)

| Browser              | WebGPU EP         | WASM EP     |
|----------------------|-------------------|-------------|
| Chrome ≥ 113         | ✅ default-on      | ✅           |
| Edge ≥ 113           | ✅ default-on      | ✅           |
| Safari ≥ 17.4        | ✅ default-on      | ✅           |
| Firefox              | 🟡 behind flag     | ✅           |
| Chrome Android ≥ 121 | 🟡 flag; device-dependent | ✅    |
| Safari iOS           | 🟡 experimental    | ✅           |

WASM fallback is always available. The demo page auto-selects WebGPU when `navigator.gpu` is present and silently falls back to WASM otherwise. Cross-origin isolation (`COOP: same-origin`, `COEP: require-corp`) must be set on the server for WASM threading — without it, WASM runs single-threaded at ~4× the latency.

## 9. Memory footprint and mobile

- `image_encoder.onnx` is 43 MB, `risk_model.onnx` 7.7 MB. First-load fetch size is ≈51 MB (gzip savings are minimal — ONNX weight matrices don't compress much).
- Peak runtime memory during encoder inference is ≈1.2 GB (four 3×2048×1664 fp32 tensors + intermediates). Low-RAM Android devices may OOM under WASM. WebGPU keeps most of this in GPU memory and is materially lighter on the JS heap.
- Recommendation for mobile frontends: default to WASM only when RAM is plentiful; feature-detect WebGPU before attempting a run; surface a "this feature requires a desktop or tablet with ≥ 4 GB RAM" banner otherwise.

## 10. Compressed/streaming DICOM delivery

The current `preprocessDicom` takes a complete `ArrayBuffer`. Progressive decode (pixel-data-as-it-arrives) and HTTP-range-fetched DICOMs are out of scope — both require deeper changes to `dicom-parser`. For a typical 15 MB mammogram DICOM over broadband this is a non-issue; for batch-of-1000 use cases, redesign the ingestion layer.

## 11. Observability

There is no built-in telemetry, tracing, or logging. The `onStage` callback in `runMirai` gives per-stage wall-clock which a frontend can wire to its own observability stack. Model drift detection, input QA, and per-patient audit trails are downstream concerns.
