# Phase 0 — Golden Reference Fixtures

This directory holds the frozen tensors that every later phase of the Mirai → ONNX migration validates against. The flow:

1. `capture_reference.py` runs the upstream Python Mirai pipeline once on the four demo DICOMs (twice, actually — once via pydicom and once via dcmtk) and writes every intermediate tensor as a `.npy` under `fixtures/`.
2. `test_baseline.py` re-runs the same pipeline and asserts every captured tensor matches the on-disk fixtures within tight tolerances.

## Where things live

| Path | What it is |
|---|---|
| `capture_reference.py` | One-shot capture script. Refuses to overwrite fixtures without `--regenerate`. |
| `test_baseline.py` | Pytest module that re-runs and asserts. |
| `_hooks.py` | Forward-hook + monkey-patch helpers used by both scripts. |
| `conftest.py` | Path constants, tolerances. |
| `fixtures/MANIFEST.json` | Versions, snapshot/file SHA-256s, shapes, asserts that passed at capture time. |
| `fixtures/predictions.json` | Bit-equal to `mirai-predict --use-pydicom` on the demo DICOMs. |
| `fixtures/batch_order.json` | Records the (view, side) per batch slot — needed by every later phase to reconstruct the input ordering. |
| `fixtures/dicom_raw_uint16/{CC,MLO}_{L,R}.npy` | Raw uint16 numpy arrays after windowing (pydicom path). |
| `fixtures/dicom_raw_uint16_dcmtk/...` | Same, but for the dcmtk path. |
| `fixtures/preproc_tensor/{CC,MLO}_{L,R}.npy` | `(3, 2048, 1664)` fp32 — the per-image tensor the encoder consumes (pydicom path). |
| `fixtures/preproc_tensor_dcmtk/...` | Same, dcmtk path. |
| `fixtures/image_encoder_out{,_dcmtk}.npy` | `(1, 4, 512)` fp32. |
| `fixtures/risk_factor_vector{,_dcmtk}.npy` | `(1, rf_dim)` fp32. **Not zeros** — model-predicted RFs because the demo has none and the trained config sets `use_pred_risk_factors_at_test=true`. |
| `fixtures/pred_risk_factors_per_key{,_dcmtk}/{key}.npy` | One file per `risk_factor_keys` entry, holding that key's per-key sigmoid/softmax tensor before concatenation. |
| `fixtures/xai_hidden{,_dcmtk}.npy` | `(1, 512+rf_dim)` fp32 — the **post-relu** hidden that flows into `prob_of_failure_layer`. This is the XAI embedding. |
| `fixtures/raw_logit{,_dcmtk}.npy` | `(1, 5)` fp32, output of `Cumulative_Probability_Layer`. |
| `fixtures/raw_sigmoid{,_dcmtk}.npy` | `(1, 5)` fp32, sigmoid of `raw_logit`. |
| `fixtures/calibrated{,_dcmtk}.npy` | `(5,)` fp64, post-calibration probabilities. |
| `fixtures/preview/{CC,MLO}_{L,R}.png` | Eyeball renders of `preproc_tensor/` (pydicom path). Breast tissue should face left in all four. |

## Reproducing the capture

You need the conda env from `ENV.md`. Once active:

```bash
cd /Users/jeya/Documents/projects/mirai-onnx
python -m tests.reference.capture_reference          # refuses if fixtures exist
python -m tests.reference.capture_reference --regenerate  # overwrite
pytest tests/reference/test_baseline.py -v
```

`capture_reference.py` does two end-to-end pipeline runs (pydicom + dcmtk), so it takes 30-60 seconds on a Rosetta-translated x86_64 Python 3.8.

## Why these tests are machine-pinned

We run torch 1.9.0 under Rosetta on macOS. fp32 reductions on CPU are deterministic to within an ULP **when** thread count, op order, and library build are all fixed. We pin:

- single thread (`torch.set_num_threads(1)`),
- deterministic algorithms (`torch.use_deterministic_algorithms(True, warn_only=True)`),
- CPU-only execution (`args.cuda = False`),
- the exact wheel set captured in `MANIFEST.env`.

If you re-run on a different machine and a fp32 fixture drifts at the 1e-6 level, that is **expected**, not a bug — but it does mean `test_baseline.py` should only be trusted on the capture machine until a Docker reference is added (planned for after Phase 1). Do **not** loosen the tolerance to make a different machine pass; instead capture a new fixture set on that machine.

## Notes from Phase 0 that contradict the migration plan

The migration plan §1/§3/§6 had four factual errors that this fixture set resolves:

1. Preproc tensor shape is `(3, 2048, 1664)`, not `(3, 1664, 2048)`. (`Scale_2d` does `Resize((height, width))` with `width, height = args.img_size = [1664, 2048]`.)
2. `risk_factor_vector` length is captured at runtime from `model.transformer.pool.length_risk_factor_vector` (= 100). 34 was the count of *keys*, not vector dimensions.
3. The demo `risk_factor_vector` is **not** zeros. With `use_pred_risk_factors_at_test=true`, the pool concatenates model-predicted RFs.
4. Batch slot order is `[(CC,L), (CC,R), (MLO,L), (MLO,R)]` for the demo CLI input order, not `R-CC, R-MLO, L-CC, L-MLO`.

`xai_hidden` is captured **post-relu** (because `RiskFactorPool.replaces_fc()` returns False). If a downstream phase wants the pre-relu version, that's a Phase 1 question.
