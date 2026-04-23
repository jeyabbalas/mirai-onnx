# Phase 5 Report — Python-side End-to-End ONNX Pipeline

**Status: COMPLETE (2026-04-23).** The two ONNX files + JSON calibrator, composed with `onnxruntime` CPU in Python, reproduce the pinned Mirai demo predictions bit-for-bit after 4-decimal rounding on **both** decode paths.

Contract source: `mirai-migration-plan.md` §6.

## Files created / modified

| Path | Role |
|---|---|
| `scripts/run_onnx_pipeline.py` | End-to-end composer; reads Phase 0 `preproc_tensor*` + `batch_order.json`, runs both ONNX sessions, applies sigmoid + `apply_calibrator_json`, rounds to 4 dp, writes `artifacts/phase_5/onnx_prediction*.json` + `onnx_embedding*.npy`, returns non-zero on any mismatch vs `PINNED_PREDICTIONS`. |
| `tests/onnx/test_end_to_end_python.py` | 10 parametrized pytest cases (`pydicom`, `dcmtk`): logit parity, embedding parity, calibrated parity, 4-dp rounded bit-equality vs pinned baseline, `predictions.json` schema equality. |
| `PHASE_5_REPORT.md` | This file. |
| `.gitignore` | Appended `artifacts/` (Phase 5+ outputs are reproducible; not committed). |

No other files were touched. `tests/reference/fixtures/` is intact (verified via `tests/reference/test_baseline.py::test_manifest_file_hashes` — all 56 cases pass).

## Source-of-inputs decision

Per the Phase 5 plan (approved before implementation), the script and tests use Phase 0's `preproc_tensor{,_dcmtk}/*.npy` fixtures as the image input, **not** live DICOMs. Those fixtures are the bit-exact output of upstream Mirai's DICOM→tensor path, pinned by SHA-256 in `tests/reference/fixtures/MANIFEST.json` and reproducible via `tests/reference/capture_reference.py`. Substituting them changes nothing about what Phase 5 proves (ONNX composition reproduces Python predictions), keeps the pipeline hermetic (no `onconet`/`MiraiFull` import), and defers the DICOM→tensor check to Phase 6 (TS port) where it naturally belongs.

A `--source=dicoms` mode was considered but not built; it's trivial to add later if a DICOM-in oracle is needed for Phase 8's browser-demo cross-check (load `MiraiFull`, install `_hooks.install_collate_capture`, capture stack, feed to ONNX).

## Commands run

```bash
conda activate mirai-export
python scripts/run_onnx_pipeline.py --decode both         # produces artifacts/phase_5/*
pytest tests/onnx/test_end_to_end_python.py -v            # 10 passed in 4.66s
pytest tests/onnx/ tests/calibrator/ tests/architecture/ -v   # 95 passed in 15.21s

conda activate mirai-py38
pytest tests/reference/test_baseline.py -v                # 56 passed in 51.00s
```

All three suites pass; no regressions.

## Parity measurements (ORT CPU vs Phase 0 fixtures)

Empirical max absolute differences, measured on this host/env (no cherry-picking — script below reproduces):

| Quantity | Target fixture | atol budget | pydicom max abs | dcmtk max abs |
|---|---|---:|---:|---:|
| `logit` (ORT) | `raw_logit{,_dcmtk}.npy` | `2e-5` | **2.980e-07** | **4.172e-07** |
| `hidden_pre_hazard` (ORT) | `xai_hidden{,_dcmtk}.npy` | `2e-5` | **4.172e-07** | **3.874e-07** |
| `calibrated` (fp64) | `calibrated{,_dcmtk}.npy` | `1e-4` (atol), `1e-3` (rtol) | **2.190e-08** | **4.249e-08** |

All three metrics are **1–2 orders of magnitude inside their committed tolerance**. The feared image-encoder-drift-propagation-through-risk-model effect did not materialize: the encoder's measured ~1e-5 drift (Phase 2) does *not* compound through the attention + RF blend; it damps to the same ~4e-7 order as the risk-model-alone measurement (Phase 3).

### Why the ORT tolerance stays at 2e-5

The measured drift is ~50× inside the bound. I chose not to tighten in line with CLAUDE.md's tolerance-stability principle: the committed `ATOL_ORT=2e-5` already survived Phase 2/3 on the same host, and tightening now would make Phase 6/8 (different envs, different providers) unnecessarily fragile without empirical justification.

### Embedding tolerance — plan's 1e-5 vs committed 2e-5

The migration plan §6 originally suggested `atol=1e-5` for the embedding. I committed `2e-5` (= `ATOL_ORT`) on the expectation that image-encoder drift would propagate. In practice drift was ~4e-7 on both paths, so `1e-5` would also have passed easily. Leaving the code at `2e-5` to keep a single project-wide ORT bound rather than introducing a per-test value that would just be re-bumped if a future provider change nudged drift.

## 4-decimal rounding stability

All 10 calibrated values are comfortably off their rounding boundaries:

| Decode | Year | calibrated (fp64) | rounded | distance to nearest 4-dp boundary |
|---|---:|---:|---:|---:|
| pydicom | 1 | 0.0313617 | 0.0314 | 1.17e-05 |
| pydicom | 2 | 0.0505131 | 0.0505 | 3.69e-05 |
| pydicom | 3 | 0.0711243 | 0.0711 | 2.57e-05 |
| pydicom | 4 | 0.0934583 | 0.0935 | 8.34e-06 |
| pydicom | 5 | 0.1051542 | 0.1052 | 4.17e-06 |
| dcmtk   | 1 | 0.0298495 | 0.0298 | **5.05e-07** |
| dcmtk   | 2 | 0.0483406 | 0.0483 | 9.39e-06 |
| dcmtk   | 3 | 0.0684051 | 0.0684 | 4.49e-05 |
| dcmtk   | 4 | 0.0900143 | 0.0900 | 3.57e-05 |
| dcmtk   | 5 | 0.1016078 | 0.1016 | 4.22e-05 |

**Robustness note**: dcmtk/Year 1 has only a 5.05e-07 margin — the unrounded value is 0.0298495, one ULP below the rounding tie at 0.02985. The measured calibrated drift is 4.25e-08 (12× smaller than the margin) so it is safe on this host, but if a future environment (different BLAS, different provider) drifts by more than ~5e-7 on dcmtk/Year 1 specifically, `round(x, 4)` could flip from `0.0298` to `0.0299` and break the bit-equal test. If that happens, the intended remediation is to re-investigate whether it's a true bug (tolerance leak somewhere) before changing the pinned value. The pydicom path and all other dcmtk years have ≥4e-6 margins, 10–100× safer.

## Output artifacts

Generated by `python scripts/run_onnx_pipeline.py --decode both`:

| File | Size | SHA-256 |
|---|---:|---|
| `artifacts/phase_5/onnx_prediction.json`       | 164 B | `4c3e763bf4443f30ff17ffe2ab77768a24be19c99d7464ff43d345bf5224cf0b` |
| `artifacts/phase_5/onnx_prediction_dcmtk.json` | 162 B | `1a603e13b5766a4d2d00274f4f34671defd03ab30ff782b8f14431b0a818c475` |
| `artifacts/phase_5/onnx_embedding.npy`         | 2576 B | `7ad5959e967d3b32453e22bdb186e5097cad464b95a2c176e7a27814f7257767` |
| `artifacts/phase_5/onnx_embedding_dcmtk.npy`   | 2576 B | `ef05518f75c67f8fcaad78d0bd1ead3bb544d24a7cb16d434581978c8953feef` |

Embeddings are `(1, 612)` fp32. JSON files mirror `predictions{,_dcmtk}.json`'s schema exactly — `diff -u` between them is empty on both decode paths.

## Environment

| Component | Version |
|---|---|
| platform | Darwin 25.4.0 x86_64 (Rosetta, conda env `mirai-export`) |
| python | 3.8.20 |
| torch | 2.2.2 (present for test infra; Phase 5 runtime itself does not invoke torch) |
| numpy | 1.24.4 |
| onnx | 1.17.0 |
| onnxruntime | 1.19.2 |

Phase 0 env (`mirai-py38`) is used only to re-verify `tests/reference/test_baseline.py` didn't drift; it's not involved in Phase 5 runtime.

## Deviations from the plan

None that changed behavior or tolerance.

- Script uses `--source=fixtures` implicitly (the only mode implemented). Approved pre-implementation; rationale in "Source-of-inputs decision" above.
- Embedding tolerance committed at `2e-5` rather than the plan text's `1e-5`. Approved pre-implementation; empirical drift is ~4e-7, so either value would pass.

## What Phase 5 does NOT yet prove

These are intentional non-goals for Phase 5 and are owned by later phases:

- **DICOM → preproc tensor equivalence outside Python.** Phase 6 (TS port) will re-implement that stage and compare against `preproc_tensor*.npy`.
- **Cross-host reproducibility of the tolerances above.** All measurements in this report are machine-pinned (same host that captured Phase 0). A Docker reference env is deferred per `mirai-migration-plan.md` §11.2.
- **GPU / WebGPU parity.** Phase 5 uses `CPUExecutionProvider` only. WebGPU parity is a Phase 8 check.

## Next step

Phase 6 — TypeScript DICOM → preproc-tensor port. Phase 5's script becomes the Python oracle Phase 6's TS output is diffed against (indirectly — via the shared `preproc_tensor*.npy` fixtures and the ONNX encoder input contract).
