import { describe, it, expect, beforeAll } from "vitest";
import fs from "node:fs";
import path from "node:path";
import * as ort from "onnxruntime-node";

import { REPO_ROOT, FIXTURES_DIR, DEMO_DATA_DIR, DEMO_DICOMS } from "./setup.js";
import { loadNpy } from "../../src/mirai/util/npy.js";
import { calibrateAll } from "../../src/mirai/calibrator.js";
import type { Calibrator } from "../../src/mirai/calibrator.js";
import { loadCalibratorFromFile } from "../../src/mirai/calibrator.node.js";
import { runMirai } from "../../src/mirai/runMirai.js";
import type { MiraiSessions, MiraiStage, OrtTensorCtor } from "../../src/mirai/runMirai.js";

const MODELS_DIR = path.join(REPO_ROOT, "models");
const CALIBRATOR_PATH = path.join(MODELS_DIR, "calibrator.json");
const ENCODER_PATH = path.join(MODELS_DIR, "image_encoder.onnx");
const RISK_PATH = path.join(MODELS_DIR, "risk_model.onnx");

const PINNED_PYDICOM: readonly number[] = [0.0314, 0.0505, 0.0711, 0.0935, 0.1052];

// Phase 6 measured flip decisions — L side always flipped, R side never flipped.
const EXPECTED_FLIPS: Record<"CC_L" | "CC_R" | "MLO_L" | "MLO_R", boolean> = {
  CC_L: true,
  CC_R: false,
  MLO_L: true,
  MLO_R: false,
};

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): { max: number; idx: number } {
  if (a.length !== b.length) {
    throw new Error(`maxAbsDiff: length mismatch ${a.length} vs ${b.length}`);
  }
  let max = 0;
  let idx = -1;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > max) {
      max = d;
      idx = i;
    }
  }
  return { max, idx };
}

let encoder: ort.InferenceSession;
let riskSession: ort.InferenceSession;
let calibrator: Calibrator;
let sessions: MiraiSessions;

beforeAll(async () => {
  calibrator = loadCalibratorFromFile(CALIBRATOR_PATH);
  encoder = await ort.InferenceSession.create(ENCODER_PATH);
  riskSession = await ort.InferenceSession.create(RISK_PATH);
  sessions = {
    encoder,
    risk: riskSession,
    Tensor: ort.Tensor as unknown as OrtTensorCtor,
  };
}, 120_000);

describe("runMirai — Mode A: fixtures-fed ONNX + calibrator wiring", () => {
  it("logit and hidden_pre_hazard match fixtures within atol=2e-5; predictions bit-equal at 4dp", async () => {
    const batchOrder = JSON.parse(
      fs.readFileSync(path.join(FIXTURES_DIR, "batch_order.json"), "utf8"),
    ) as Array<{ slot: number; view: 0 | 1; view_str: "CC" | "MLO"; side: 0 | 1; side_str: "L" | "R" }>;
    expect(batchOrder).toHaveLength(4);

    const perImage = 3 * 2048 * 1664;
    const stack = new Float32Array(4 * perImage);
    for (let s = 0; s < 4; s++) {
      const entry = batchOrder[s];
      const label = `${entry.view_str}_${entry.side_str}`;
      const fx = loadNpy(path.join(FIXTURES_DIR, "preproc_tensor", `${label}.npy`));
      expect(fx.shape).toEqual([3, 2048, 1664]);
      stack.set(fx.data as Float32Array, s * perImage);
    }

    // Encoder.
    const encInTensor = new ort.Tensor("float32", stack, [4, 3, 2048, 1664]);
    const encOut = await encoder.run({ input: encInTensor });
    const encFeats = encOut["output"].data as Float32Array;
    expect(encFeats.length).toBe(4 * 512);

    // Risk model.
    const imgFeats = new ort.Tensor("float32", encFeats, [1, 4, 512]);
    const viewSeq = BigInt64Array.from(batchOrder.map((e) => BigInt(e.view)));
    const sideSeq = BigInt64Array.from(batchOrder.map((e) => BigInt(e.side)));
    const timeSeq = new BigInt64Array(4);
    const rfVector = new Float32Array(100);
    const rfMask = new Float32Array(100);
    const riskOut = await riskSession.run({
      img_feats: imgFeats,
      view_seq: new ort.Tensor("int64", viewSeq, [1, 4]),
      side_seq: new ort.Tensor("int64", sideSeq, [1, 4]),
      time_seq: new ort.Tensor("int64", timeSeq, [1, 4]),
      rf_vector: new ort.Tensor("float32", rfVector, [1, 100]),
      rf_known_mask: new ort.Tensor("float32", rfMask, [1, 100]),
    });
    const logit = riskOut["logit"].data as Float32Array;
    const hidden = riskOut["hidden_pre_hazard"].data as Float32Array;
    expect(logit.length).toBe(5);
    expect(hidden.length).toBe(612);

    const logitFx = loadNpy(path.join(FIXTURES_DIR, "raw_logit.npy"));
    const hiddenFx = loadNpy(path.join(FIXTURES_DIR, "xai_hidden.npy"));
    expect(logitFx.shape).toEqual([1, 5]);
    expect(hiddenFx.shape).toEqual([1, 612]);

    const logitDiff = maxAbsDiff(logit, logitFx.data as Float32Array);
    const hiddenDiff = maxAbsDiff(hidden, hiddenFx.data as Float32Array);
    console.log(
      `  Mode A: logit maxAbsDiff=${logitDiff.max.toExponential(3)} @${logitDiff.idx}; hidden maxAbsDiff=${hiddenDiff.max.toExponential(3)} @${hiddenDiff.idx}`,
    );
    // ORT-CPU vs PyTorch tier (CLAUDE.md tolerances table).
    expect(logitDiff.max).toBeLessThan(2e-5);
    expect(hiddenDiff.max).toBeLessThan(2e-5);

    // Sigmoid + calibrator.
    const sigmoid = new Float64Array(5);
    for (let i = 0; i < 5; i++) sigmoid[i] = 1.0 / (1.0 + Math.exp(-logit[i]));

    const calibrated = calibrateAll(sigmoid, calibrator);
    const calibFx = loadNpy(path.join(FIXTURES_DIR, "calibrated.npy"));
    expect(calibFx.shape).toEqual([5]);
    const calibDiff = maxAbsDiff(calibrated, calibFx.data as Float64Array);
    console.log(`  Mode A: calibrated maxAbsDiff=${calibDiff.max.toExponential(3)} @${calibDiff.idx}`);
    // Calibrator amplifies via base_slope ~5x; matches Phase 5's ATOL_CALIBRATED.
    expect(calibDiff.max).toBeLessThan(1e-4);

    const rounded = Array.from(calibrated).map((v) => Math.round(v * 1e4) / 1e4);
    expect(rounded).toEqual([...PINNED_PYDICOM]);
  }, 180_000);
});

describe("runMirai — Mode B: real DICOMs end-to-end", () => {
  it("predictions bit-equal pinned pydicom values at 4dp; slotOrder follows input order", async () => {
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

    // slotOrder must reflect the caller's input order, with view/side auto-extracted
    // from each DICOM's tags, and Phase 6's measured flip decisions.
    expect(result.slotOrder).toEqual(
      DEMO_DICOMS.map((d) => ({
        view: d.view,
        side: d.side,
        flipped: EXPECTED_FLIPS[d.label],
      })),
    );

    expect(result.embedding.length).toBe(612);
    expect(result.rawLogit.length).toBe(5);
    expect(result.rawSigmoid.length).toBe(5);
    expect(result.calibrated.length).toBe(5);
    expect(result.modelVersion).toBe("0.14.1");
  }, 240_000);

  it("onStage callback fires once per stage in order", async () => {
    const files = DEMO_DICOMS.map((d) =>
      fs.readFileSync(path.join(DEMO_DATA_DIR, d.file)),
    );
    const stages: MiraiStage[] = [];
    const timings: Record<string, number> = {};
    await runMirai(files, sessions, calibrator, {}, {
      onStage: (stage, ms) => {
        stages.push(stage);
        timings[stage] = ms;
      },
    });
    expect(stages).toEqual(["preprocess", "encoder", "risk", "calibrate", "total"]);
    for (const stage of stages) {
      expect(timings[stage]).toBeGreaterThanOrEqual(0);
      expect(Number.isFinite(timings[stage])).toBe(true);
    }
    // total bounds each component — redundant but catches reversed timestamps.
    expect(timings.total).toBeGreaterThanOrEqual(timings.preprocess);
    expect(timings.total).toBeGreaterThanOrEqual(timings.encoder);
    expect(timings.total).toBeGreaterThanOrEqual(timings.risk);
    expect(timings.total).toBeGreaterThanOrEqual(timings.calibrate);
  }, 240_000);
});
