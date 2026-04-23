# Phase 1 — Execution Report

Completed `2026-04-22` on `Jeyas-MacBook-Pro.local`.

## Summary
- Produced `docs/architecture.md` (41,693 bytes, 15 H2 sections) — the design document that drives Phase 2/3 ONNX exports and Phase 4 calibrator extraction.
- Produced `tests/architecture/test_plan.py` — stdlib-only pytest module; 44 parametrized assertions, all green in 0.03 s.
- Produced root `CLAUDE.md` — pinned Phase 0 corrections, tolerances, and Phase 1 decisions.
- No edits under `external/Mirai/`, `tests/reference/fixtures/`, or `mirai-migration-plan.md`.
- Re-ran Phase 0 baseline (`tests/reference/test_baseline.py`) — still 56/56 green, no regression.

## Files created (relative to project root)

| Path | Size | Purpose |
|---|---|---|
| `docs/architecture.md` | 41,693 bytes | Full architecture inventory: shape table, module graph, control-flow branches, eval-mode commitment, per-key FC double-call analysis, XAI-hidden decision (post-relu), export surface spec, refactor targets, export-time settings, .eval() sequencing, pitfalls, validation checklist |
| `tests/architecture/__init__.py` | 0 bytes | Package marker |
| `tests/architecture/test_plan.py` | 6,840 bytes | Lint test: doc exists, 15 H2 headings, ONNX I/O names/shapes/dtypes, opset=17 commitment, XAI=post-relu commitment, rf_dim dynamic (not hardcoded to 34), batch_order.json cited (no hardcoded slot order), .eval() + upstream + eval mentioned in §6, per-key _fc rule with "exactly once", 6 Phase 0 fixture stems referenced, ≥3 file:line refs under external/Mirai/onconet/ in §11 |
| `CLAUDE.md` | — | Repo-level guardrail: project one-liner, required reading list, pinned Phase 0 facts, pinned Phase 1 decisions, environment notes, "do not" list |
| `PHASE_1_REPORT.md` | this file | this report |

## User decisions captured (via AskUserQuestion)

| Question | Answer |
|---|---|
| Which hidden tensor for `hidden_pre_hazard`? | Post-ReLU (recommended) |
| ONNX `opset_version`? | 17 (recommended) |
| ONNX export path? | Classic `torch.onnx.export` (recommended) |
| Create a top-level `CLAUDE.md`? | Yes |

## Commands run (in order)
```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate mirai-py38
mkdir -p docs tests/architecture
# (file writes via Claude Code's Write tool)
pytest tests/architecture/test_plan.py -v
# → 44 passed in 0.03s
pytest tests/reference/test_baseline.py -v
# → 56 passed in 51.05s
git status
# → only Phase 1 additions (CLAUDE.md, docs/, tests/architecture/) plus pre-existing Phase 0 untracked items
```

## Key architectural decisions recorded in `docs/architecture.md`

1. **Export surface.** `image_encoder.onnx` takes `(N, 3, 2048, 1664)` fp32 → `(N, 512)` fp32, dynamic axis `N`. `risk_model.onnx` takes 6 inputs (`img_feats`, `view_seq`, `side_seq`, `time_seq`, `rf_vector`, `rf_known_mask`) and produces 2 outputs (`logit`, `hidden_pre_hazard`), dynamic axis `B`.
2. **XAI-hidden = post-relu.** The `hidden_pre_hazard` output of `risk_model.onnx` is validated against `xai_hidden{,_dcmtk}.npy`. Pre-relu remains available as `pool_hidden` fixtures for downstream consumers who prefer it.
3. **opset_version=17** with `do_constant_folding=True` and classic `torch.onnx.export`.
4. **Eval-mode.** Phase 2/3 export wrappers must call `.eval()` on the entire model tree. Upstream Mirai never does this; the outer `MiraiFull` lives in train mode while children are in eval mode via `torch.load`.
5. **Per-key FC exactly once.** Export wrappers must invoke each `{key}_fc` exactly once (first-call semantics), matching Phase 0's `if k not in captured` hook guard. `get_pred_rf_loss` is not called from the wrapper.
6. **Slot ordering is caller-chosen.** ONNX graph is order-agnostic given consistent `view_seq`/`side_seq`; Phase 5/8 must read `tests/reference/fixtures/batch_order.json` and not hardcode any order.
7. **Refactor targets (R1–R8)** recorded with upstream `file:line` references — all refactors live in new export-wrapper modules, not in `external/Mirai/`.

## Verification

### Automated
- `pytest tests/architecture/test_plan.py -v` — **44 passed in 0.03s.**
- `pytest tests/reference/test_baseline.py -v` — **56 passed in 51.05s** (Phase 0 baseline unchanged).

### Manual (for the user)
1. Open `docs/architecture.md` and read top-to-bottom. Confirm every committed decision reads correctly (§8 post-relu, §12 opset_version=17, §9.1 and §9.2 I/O names and shapes).
2. Spot-check any 3 refactor targets in §11 against `external/Mirai/onconet/...py` — confirm the cited code is at the line number given in the pinned Mirai SHA `4af944449863966a5a9c66b44e56e3c141223897`.
3. Spot-check any 3 shape rows in §3 against `tests/reference/fixtures/MANIFEST.json`.
4. Read `CLAUDE.md`; confirm the four Phase 0 corrections and the pinned tolerances (`ATOL_FP32=1e-6`, `ATOL_FP64=1e-9`) are present.
5. `git status` — expected untracked new entries: `CLAUDE.md`, `docs/`, `tests/architecture/`, `PHASE_1_REPORT.md`. No modifications to `external/Mirai/` or `tests/reference/fixtures/`.

## Not done (deferred per plan)
- ONNX image-encoder export — Phase 2.
- ONNX risk-model export with XAI output — Phase 3.
- Calibrator extraction to JSON — Phase 4.
- Python end-to-end ONNX pipeline — Phase 5.
- TypeScript preprocessor / RF vectorizer / browser wiring — Phases 6–8.
- Cross-machine Docker reference — deferred per migration plan §11.2.

## No git commit performed
Per the user's instruction, no git commits were made. User will review the changes and commit manually.
