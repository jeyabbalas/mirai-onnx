// Phase 9: browser-backend parity.
//
// Loads the same ONNX graphs via `onnxruntime-web` on the WASM EP — the code path
// the browser demo will take when WebGPU is unavailable — and asserts numerical
// parity with the Phase 0 fixtures and the pinned pydicom predictions.
//
// WebGPU can't be exercised under Node; the demo page carries that manual gate.
//
// Skip the spec when SKIP_WEB_PARITY=1 so environments without usable WASM
// threading can still run `npm test`.

import { describe, it, expect, beforeAll } from "vitest";
import fs from "node:fs";
import path from "node:path";

import { REPO_ROOT, FIXTURES_DIR, DEMO_DATA_DIR, DEMO_DICOMS } from "./setup.js";
import { loadNpy } from "../../src/mirai/util/npy.js";
import { loadCalibratorFromFile } from "../../src/mirai/calibrator.node.js";
import type { Calibrator } from "../../src/mirai/calibrator.js";
import { runMirai } from "../../src/mirai/runMirai.js";
import type { MiraiSessions } from "../../src/mirai/runMirai.js";
import { createWebSessionsFromBytes } from "../../src/mirai/sessions/web.js";

const MODELS_DIR = path.join(REPO_ROOT, "models");
const CALIBRATOR_PATH = path.join(MODELS_DIR, "calibrator.json");
const ENCODER_PATH = path.join(MODELS_DIR, "image_encoder.onnx");
const RISK_PATH = path.join(MODELS_DIR, "risk_model.onnx");

const PINNED_PYDICOM: readonly number[] = [0.0314, 0.0505, 0.0711, 0.0935, 0.1052];

const SKIP = process.env.SKIP_WEB_PARITY === "1";

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) {
    throw new Error(`length mismatch ${a.length} vs ${b.length}`);
  }
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > max) max = d;
  }
  return max;
}

function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) {
    throw new Error(`length mismatch ${a.length} vs ${b.length}`);
  }
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

let sessions: MiraiSessions;
let calibrator: Calibrator;

(SKIP ? describe.skip : describe)("runMirai — browser backend (onnxruntime-web, WASM EP)", () => {
  beforeAll(async () => {
    calibrator = loadCalibratorFromFile(CALIBRATOR_PATH);
    const encoderBytes = fs.readFileSync(ENCODER_PATH);
    const riskBytes = fs.readFileSync(RISK_PATH);
    sessions = await createWebSessionsFromBytes(
      {
        encoder: encoderBytes.buffer.slice(
          encoderBytes.byteOffset,
          encoderBytes.byteOffset + encoderBytes.byteLength,
        ) as ArrayBuffer,
        risk: riskBytes.buffer.slice(
          riskBytes.byteOffset,
          riskBytes.byteOffset + riskBytes.byteLength,
        ) as ArrayBuffer,
      },
      // WebGPU is not available under Node. WASM-only keeps the spec deterministic.
      { preferWebGPU: false, numThreads: 4 },
    );
  }, 180_000);

  it("predictions bit-equal pinned pydicom baseline; embedding cos ≥ 0.99999", async () => {
    const files = DEMO_DICOMS.map((d) =>
      fs.readFileSync(path.join(DEMO_DATA_DIR, d.file)),
    );
    const result = await runMirai(files, sessions, calibrator);

    const pred = [
      result.predictions.year1,
      result.predictions.year2,
      result.predictions.year3,
      result.predictions.year4,
      result.predictions.year5,
    ];
    expect(pred).toEqual([...PINNED_PYDICOM]);

    // Embedding vs Phase 0 xai_hidden.npy (post-ReLU, shape (1, 612)).
    const hiddenFx = loadNpy(path.join(FIXTURES_DIR, "xai_hidden.npy"));
    expect(hiddenFx.shape).toEqual([1, 612]);
    const hiddenFxData = hiddenFx.data as Float32Array;
    const embDiff = maxAbsDiff(result.embedding, hiddenFxData);
    const embCos = cosineSimilarity(result.embedding, hiddenFxData);
    console.log(
      `  web/WASM: embedding maxAbsDiff=${embDiff.toExponential(3)} cos=${embCos.toFixed(8)}`,
    );
    expect(embDiff).toBeLessThan(2e-5);
    expect(embCos).toBeGreaterThanOrEqual(0.99999);

    // Raw logit vs raw_logit.npy.
    const logitFx = loadNpy(path.join(FIXTURES_DIR, "raw_logit.npy"));
    expect(logitFx.shape).toEqual([1, 5]);
    const logitDiff = maxAbsDiff(result.rawLogit, logitFx.data as Float32Array);
    console.log(`  web/WASM: logit maxAbsDiff=${logitDiff.toExponential(3)}`);
    expect(logitDiff).toBeLessThan(2e-5);

    // Calibrated fp64 vs calibrated.npy (Phase 5 empirically ≤ 4e-8).
    const calibFx = loadNpy(path.join(FIXTURES_DIR, "calibrated.npy"));
    expect(calibFx.shape).toEqual([5]);
    const calibDiff = maxAbsDiff(result.calibrated, calibFx.data as Float64Array);
    console.log(`  web/WASM: calibrated maxAbsDiff=${calibDiff.toExponential(3)}`);
    expect(calibDiff).toBeLessThan(1e-7);

    expect(result.modelVersion).toBe("0.14.1");
  }, 240_000);
});
