# Phase 0 — Execution Report

Captured `2026-04-22` on `Jeyas-MacBook-Pro.local` (macOS 26.4.1, Apple Silicon).

## Summary
- Stood up upstream Mirai (`onconet 0.14.1`, git SHA `4af9444`) under a fresh Python 3.8 / Rosetta x86_64 conda env.
- Captured every Phase 0 tensor on the four demo DICOMs, **twice** (pydicom and dcmtk decoding paths).
- 109 fixture files (~425 MB) under `tests/reference/fixtures/`.
- 56 pytest assertions, all green, idempotent across two consecutive runs (50.2s and 50.4s).
- Predictions match upstream `mirai-predict` to 4 decimal places on both paths.

## Files created (relative to project root)
| Path | Purpose |
|---|---|
| `tests/__init__.py` | empty package marker |
| `tests/reference/__init__.py` | empty package marker |
| `tests/reference/conftest.py` | path constants + tolerances (`ATOL_FP32 = 1e-6`, `ATOL_FP64 = 1e-9`) |
| `tests/reference/_hooks.py` | `install_dicom_capture`, `install_collate_capture`, `install_post_load_hooks`, `derive_pred_risk_factors_per_key` |
| `tests/reference/capture_reference.py` | one-shot capture script with `--regenerate` and `--only` flags |
| `tests/reference/test_baseline.py` | 56 parametrized pytest assertions |
| `tests/reference/README.md` | what the fixtures are, how to regenerate, machine-pinned caveat |
| `tests/reference/ENV.md` | env install + pylibjpeg workaround + verification commands |
| `tests/reference/fixtures/MANIFEST.json` | versions, snapshot/file SHAs, shapes, asserts that passed |
| `tests/reference/fixtures/predictions.json`, `predictions_dcmtk.json` | bit-equal to upstream demo |
| `tests/reference/fixtures/batch_order.json`, `batch_order_dcmtk.json` | (view, side) per slot |
| `tests/reference/fixtures/dicom_raw_uint16{,_dcmtk}/{CC,MLO}_{L,R}.npy` | uint16 windowed DICOM arrays |
| `tests/reference/fixtures/preproc_tensor{,_dcmtk}/{CC,MLO}_{L,R}.npy` | `(3, 2048, 1664)` fp32 |
| `tests/reference/fixtures/image_encoder_out{,_dcmtk}.npy` | `(1, 4, 512)` fp32 |
| `tests/reference/fixtures/image_hidden_in_pool{,_dcmtk}.npy` | `(1, 512)` fp32 |
| `tests/reference/fixtures/pool_hidden{,_dcmtk}.npy` | `(1, 612)` fp32 — pre-relu hidden returned by RiskFactorPool |
| `tests/reference/fixtures/risk_factor_vector{,_dcmtk}.npy` | `(1, 100)` fp32 — concatenated model-predicted RFs |
| `tests/reference/fixtures/pred_risk_factors_per_key{,_dcmtk}/<key>.npy` | one per `risk_factor_keys` entry (×34 each path) |
| `tests/reference/fixtures/xai_hidden{,_dcmtk}.npy` | `(1, 612)` fp32 — **post-relu** hidden, the input to `prob_of_failure_layer` |
| `tests/reference/fixtures/raw_logit{,_dcmtk}.npy` | `(1, 5)` fp32, output of `Cumulative_Probability_Layer` |
| `tests/reference/fixtures/raw_sigmoid{,_dcmtk}.npy` | `(1, 5)` fp32 |
| `tests/reference/fixtures/calibrated{,_dcmtk}.npy` | `(5,)` fp64 |
| `tests/reference/fixtures/preview/{CC,MLO}_{L,R}.png` | eyeball renders (pydicom path only) |
| `PHASE_0_REPORT.md` | this file |
| `.gitignore` | extended to cover demo data, snapshots, pyc, pytest cache, etc. |

## Commands run (in order)
```bash
brew install dcmtk
brew install --cask miniforge
softwareupdate --install-rosetta --agree-to-license
rm -rf /Users/jeya/Documents/projects/mirai-onnx/.venv
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
CONDA_SUBDIR=osx-64 conda create -n mirai-py38 python=3.8 -y
conda activate mirai-py38
conda config --env --set subdir osx-64
pip install -e /Users/jeya/Documents/projects/mirai-onnx/external/Mirai[test]
pip uninstall -y pylibjpeg-openjpeg pylibjpeg-libjpeg pylibjpeg-rle pylibjpeg
mirai-predict --dry-run         # downloads ~/.mirai/snapshots
cd /Users/jeya/Documents/projects/mirai-onnx
curl -sLO https://github.com/reginabarzilaygroup/Mirai/releases/latest/download/mirai_demo_data.zip
unzip -o mirai_demo_data.zip -d mirai_demo_data
mirai-predict --output-path demo_prediction.json --use-pydicom \
    mirai_demo_data/{ccl1,ccr1,mlol2,mlor2}.dcm
mirai-predict --output-path demo_prediction_dcmtk.json \
    mirai_demo_data/{ccl1,ccr1,mlol2,mlor2}.dcm
python -m tests.reference.capture_reference --regenerate
pytest tests/reference/test_baseline.py -v   # twice
```

## Findings worth flagging for later phases

### 1. Migration plan §1/§3/§6 had several factual errors (now resolved)
- Preproc tensor shape is `(3, 2048, 1664)`, not `(3, 1664, 2048)`. (`Scale_2d` reads `width, height = args.img_size` then calls `Resize((height, width))`.)
- `risk_factor_vector` length is `100`, not `34`. 34 is the count of *risk_factor_keys*; each key produces multiple binary/one-hot dims. Captured at runtime from `model.transformer.pool.length_risk_factor_vector`.
- The demo `risk_factor_vector` is **NOT** zeros. With `pred_risk_factors=true` and `use_pred_risk_factors_at_test=true` (both set in `mirai_trained.json`), the pool concatenates **model-predicted** RFs.
- Batch slot order is `[(CC,L), (CC,R), (MLO,L), (MLO,R)]` for the demo CLI input order, NOT `R-CC, R-MLO, L-CC, L-MLO`. The order is determined by dict iteration in `MiraiModel.run_model`.

### 2. Upstream Mirai never calls `.eval()` in inference
`MiraiModel.process_image_joint` does not put the model into eval mode. Whether the children are in eval depends on what mode the snapshot was saved in. Empirically:
- Outer `MiraiFull` is in **train mode** (freshly constructed; `nn.Module` default).
- `image_encoder`, `transformer`, `pool`, `prob_of_failure_layer` are in **eval mode** (loaded as full pickled objects via `torch.load`, which restores `_training` flags).

This works because the eval-mode children dominate inference correctness (BatchNorm running stats, dropout disabled). But Phase 2+ ONNX export should explicitly call `.eval()` on the entire model before tracing.

### 3. Per-key `*_fc` modules are called TWICE per inference
In `aggregate_and_classify`, the path is:
1. `RiskFactorPool.forward` calls each `key_fc` once, with the raw internal-pool hidden. The result feeds `risk_factor_vector`.
2. `self.relu(hidden)` then runs **in place** on the pool output.
3. `AllImageTransformer.forward` then calls `self.pool.get_pred_rf_loss(hidden, risk_factors)`, which iterates the same `key_fc`s again — but with the post-relu image hidden slice. (This is wrapped in `try/except` and silently fails because `risk_factors=None` in inference.)

A naive `register_forward_hook` on each `*_fc` captures the SECOND call. The capture script captures only the FIRST per key; if Phase 2+ touches this code, beware.

### 4. pylibjpeg wheels are mislabeled on PyPI
`pylibjpeg-libjpeg==2.2.0` and `pylibjpeg-openjpeg==2.3.0` ship wheels whose filenames advertise `macosx_*_x86_64` but the `.so` inside is arm64. Under our x86_64 Python 3.8, pydicom imports fail. Workaround: uninstall all four pylibjpeg packages. The demo DICOMs are uncompressed and pydicom handles them with built-in handlers. If a future DICOM uses JPEG 2000 transfer syntax, we'll need either `python-gdcm` or older (correctly-tagged) pylibjpeg wheels.

### 5. `xai_hidden` is captured POST-relu
Because `RiskFactorPool.replaces_fc()` returns False, `aggregate_and_classify` runs `self.relu(hidden)` (in place) before passing to `prob_of_failure_layer`. So `xai_hidden` (the input to `prob_of_failure_layer`) is the post-relu tensor. If a downstream phase wants the pre-relu version, that's `pool_hidden` (also captured).

## Manual verification checklist for the user
1. `cat tests/reference/fixtures/predictions.json` — should show Year 1 = 0.0314, increasing through Year 5 = 0.1052.
2. `cat tests/reference/fixtures/predictions_dcmtk.json` — Year 1 = 0.0298 → Year 5 = 0.1016.
3. Open `tests/reference/fixtures/preview/{CC,MLO}_{L,R}.png` — breast tissue clearly faces left in all four. (Already eyeballed during execution — pass.)
4. Inspect `tests/reference/fixtures/MANIFEST.json` — confirm `git.mirai_sha == 4af944449863966a5a9c66b44e56e3c141223897`, `shapes.rf_dim == 100`, `shapes.preproc_batch_x == [1,3,4,2048,1664]`, `batch_order` matches `[(CC,L),(CC,R),(MLO,L),(MLO,R)]`, `env.python_arch == "x86_64"`.
5. `diff <(jq -S '.predictions' demo_prediction.json) <(jq -S '.predictions' tests/reference/fixtures/predictions.json)` — empty.
6. `pytest tests/reference/test_baseline.py -v` — 56 passed. (Confirmed; ran twice in 50s each.)

## Not done (deferred per plan)
- ONNX export — Phase 2+.
- TS port — Phase 6+.
- RF vectorizer fixtures (`tests/rf/`) — Phase 7.
- Docker reference for cross-machine reproducibility — added in Phase 1.

## Open question for next phase
Whether downstream phases want the **post-relu** `xai_hidden` (currently captured) or the **pre-relu** `pool_hidden` as the public XAI embedding. Both are saved.
