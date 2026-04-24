/**
 * Public API barrel for the Mirai ONNX pipeline.
 *
 * This package wraps the two ONNX graphs exported from the original PyTorch Mirai
 * model (image encoder + risk transformer) and the portable calibrator JSON, so
 * consumers can run 5-year breast-cancer risk inference in the browser (via
 * `onnxruntime-web` + WebGPU) or on Node (via `onnxruntime-node`) with numerical
 * parity against the reference Python pipeline.
 *
 * Pinned numerical contract (Phase 0 golden fixtures):
 *   - Four demo DICOMs with no user-supplied risk factors produce 4-dp rounded
 *     predictions `[0.0314, 0.0505, 0.0711, 0.0935, 0.1052]` (pydicom decode path).
 *   - The 612-dim post-ReLU embedding matches `xai_hidden.npy` within `atol=2e-5`.
 *
 * Quick start:
 * ```ts
 * import {
 *   createNodeSessions,
 *   loadCalibratorFromFile,
 *   runMirai,
 * } from "mirai-onnx-web";
 *
 * const sessions = await createNodeSessions({
 *   encoder: "models/image_encoder.onnx",
 *   risk:    "models/risk_model.onnx",
 * });
 * const calibrator = loadCalibratorFromFile("models/calibrator.json");
 * const files = dicomPaths.map(p => fs.readFileSync(p));      // 4 DICOMs
 * const result = await runMirai(files, sessions, calibrator);
 * // result.predictions = { year1: 0.0314, ..., year5: 0.1052 }
 * // result.embedding   = Float32Array(612)
 * ```
 *
 * @packageDocumentation
 */

// ── Preprocessing ────────────────────────────────────────────────────────────
export {
  preprocessDicom,
  MIRAI_INPUT_ROWS,
  MIRAI_INPUT_COLS,
  MIRAI_IMG_MEAN,
  MIRAI_IMG_STD,
} from "./preprocess/index.js";
export type { PreprocessOptions } from "./preprocess/index.js";
export type { PreprocessResult, View, Side } from "./types.js";

// ── Risk factor vectorizer ───────────────────────────────────────────────────
export {
  vectorizeRiskFactors,
  RISK_FACTOR_KEYS,
  RF_KEY_TO_NUM_CLASS,
  RF_KEY_TO_OFFSET,
  RF_DIM,
  FEATURE_NAMES,
} from "./riskFactors/index.js";
export type {
  MiraiRiskFactors,
  VectorizerResult,
  Relative,
  RelativeCode,
  RaceCode,
  DensityCode,
  HrtInfo,
  HrtType,
  RiskFactorKey,
} from "./riskFactors/index.js";

// ── Calibrator ───────────────────────────────────────────────────────────────
// Browser-safe APIs. Node callers can additionally import `loadCalibratorFromFile`
// from "mirai-onnx-web/calibrator-node" (or via the TS path `./calibrator.node.js`
// inside this repo) to read a calibrator JSON directly from disk.
export {
  loadCalibrator,
  calibrateYear,
  calibrateAll,
  CALIBRATOR_SCHEMA_VERSION,
  CALIBRATOR_N_YEARS,
} from "./calibrator.js";
export type { Calibrator, CalibratorYear } from "./calibrator.js";

// ── Core pipeline ────────────────────────────────────────────────────────────
export { runMirai, MIRAI_MODEL_VERSION } from "./runMirai.js";
export type {
  MiraiSessions,
  MiraiResult,
  MiraiPredictions,
  MiraiSlot,
  MiraiStage,
  MiraiRunOptions,
  OrtSession,
  OrtTensor,
  OrtTensorCtor,
  OrtTensorData,
} from "./runMirai.js";

// ── Plan-named convenience wrappers ──────────────────────────────────────────
export { predictMiraiRisk, getMiraiEmbedding } from "./api.js";

// ── Session factories ────────────────────────────────────────────────────────
// Web: browsers + WebGPU/WASM via `onnxruntime-web`. Dynamic-imports the runtime
// so consumers who only use preprocess/riskFactors/calibrator don't pay for it.
export {
  createWebSessions,
  createWebSessionsFromBytes,
} from "./sessions/web.js";
export type { CreateWebSessionsOptions } from "./sessions/web.js";

// Node: `onnxruntime-node` for local benchmarks, CI parity, and this repo's
// `scripts/run_ts_pipeline.ts`. Native bindings — do not import from browser code.
export { createNodeSessions } from "./sessions/node.js";
