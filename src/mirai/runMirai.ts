import { preprocessDicom } from "./preprocess/index.js";
import { vectorizeRiskFactors } from "./riskFactors/index.js";
import type { MiraiRiskFactors } from "./riskFactors/index.js";
import { calibrateAll } from "./calibrator.js";
import type { Calibrator } from "./calibrator.js";
import type { Side, View } from "./types.js";

const ENCODER_INPUT_NAME = "input";
const ENCODER_OUTPUT_NAME = "output";
const RISK_OUTPUT_LOGIT = "logit";
const RISK_OUTPUT_HIDDEN = "hidden_pre_hazard";

const NUM_IMAGES = 4;
const ENCODER_FEATURE_DIM = 512;
const RF_DIM_EXPECTED = 100;
const HIDDEN_DIM = 612;
const N_YEARS = 5;
const INPUT_ROWS = 2048;
const INPUT_COLS = 1664;
const INPUT_CHANNELS = 3;
const PER_IMAGE_ELEMENTS = INPUT_CHANNELS * INPUT_ROWS * INPUT_COLS;

// Phase 0's predictions.json carries this tag; runMirai echoes it so downstream callers
// can version-gate without re-reading predictions.json. Update in lockstep with Phase 0.
export const MIRAI_MODEL_VERSION = "0.14.1";

// Minimal structural types that both onnxruntime-node and onnxruntime-web satisfy, so this
// module stays backend-agnostic. Phase 9's browser entry will inject onnxruntime-web
// sessions through the same shape; no imports from either ORT package belong here.
export interface OrtTensor {
  readonly data: unknown;
  readonly dims: readonly number[];
  readonly type: string;
}

export interface OrtSession {
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
}

export type OrtTensorData = Float32Array | BigInt64Array | Float64Array | Uint8Array;

export type OrtTensorCtor = new (
  type: "float32" | "int64" | "float64" | "bool",
  data: OrtTensorData,
  dims: readonly number[],
) => OrtTensor;

export interface MiraiSessions {
  encoder: OrtSession;
  risk: OrtSession;
  Tensor: OrtTensorCtor;
}

export interface MiraiSlot {
  view: View;
  side: Side;
  flipped: boolean;
}

export interface MiraiPredictions {
  year1: number;
  year2: number;
  year3: number;
  year4: number;
  year5: number;
}

export interface MiraiResult {
  predictions: MiraiPredictions;
  embedding: Float32Array;   // (HIDDEN_DIM,) post-ReLU — matches xai_hidden.npy.
  rawLogit: Float32Array;    // (N_YEARS,)
  rawSigmoid: Float64Array;  // (N_YEARS,)
  calibrated: Float64Array;  // (N_YEARS,) full fp64; predictions above are this rounded to 4dp.
  slotOrder: MiraiSlot[];
  modelVersion: string;
}

export type MiraiStage = "preprocess" | "encoder" | "risk" | "calibrate" | "total";

export interface MiraiRunOptions {
  /**
   * Per-stage wall-clock callback (milliseconds). Fires once per stage in order:
   * preprocess → encoder → risk → calibrate → total. Omit for zero-cost runs;
   * when present, uses `performance.now()` at stage boundaries.
   */
  onStage?: (stage: MiraiStage, ms: number) => void;
}

function stackEncoderInput(images: ReadonlyArray<Float32Array>): Float32Array {
  const out = new Float32Array(images.length * PER_IMAGE_ELEMENTS);
  for (let i = 0; i < images.length; i++) {
    const img = images[i];
    if (img.length !== PER_IMAGE_ELEMENTS) {
      throw new Error(
        `runMirai: image ${i} has ${img.length} elements, expected ${PER_IMAGE_ELEMENTS}`,
      );
    }
    out.set(img, i * PER_IMAGE_ELEMENTS);
  }
  return out;
}

function expectTensor(
  t: OrtTensor | undefined,
  name: string,
  expectedType: string,
  expectedDims: readonly number[],
): Float32Array {
  if (!t) {
    throw new Error(`runMirai: missing output '${name}'`);
  }
  if (t.type !== expectedType) {
    throw new Error(`runMirai: output '${name}' type ${t.type}, expected ${expectedType}`);
  }
  if (t.dims.length !== expectedDims.length || t.dims.some((d, i) => d !== expectedDims[i])) {
    throw new Error(
      `runMirai: output '${name}' dims ${JSON.stringify(t.dims)}, expected ${JSON.stringify(expectedDims)}`,
    );
  }
  if (!(t.data instanceof Float32Array)) {
    throw new Error(`runMirai: output '${name}' data is not a Float32Array`);
  }
  return t.data;
}

function round4dp(x: number): number {
  // Matches Python's `round(float(x), 4)` for non-tie values. Demo predictions
  // are all safely away from .xxxx5 ties; a cross-language sanity test gates parity.
  return Math.round(x * 1e4) / 1e4;
}

export async function runMirai(
  files: ReadonlyArray<ArrayBufferLike | Uint8Array>,
  sessions: MiraiSessions,
  calibrator: Calibrator,
  riskFactors: MiraiRiskFactors = {},
  options?: MiraiRunOptions,
): Promise<MiraiResult> {
  if (files.length !== NUM_IMAGES) {
    throw new Error(`runMirai: expected ${NUM_IMAGES} DICOMs, got ${files.length}`);
  }

  const onStage = options?.onStage;
  const tTotal = onStage ? performance.now() : 0;

  // Preprocess each DICOM. preprocessDicom also extracts view/side from DICOM tags,
  // so the caller's file order directly determines slot order (matches Phase 5,
  // which reads batch_order.json populated by dict-iteration of the caller's input).
  const tPre = onStage ? performance.now() : 0;
  const preprocessed = files.map((buf) => preprocessDicom(buf));
  const slotOrder: MiraiSlot[] = preprocessed.map(({ view, side, flipped }) => ({
    view,
    side,
    flipped,
  }));
  onStage?.("preprocess", performance.now() - tPre);

  const encoderInput = new sessions.Tensor(
    "float32",
    stackEncoderInput(preprocessed.map((p) => p.pixels)),
    [NUM_IMAGES, INPUT_CHANNELS, INPUT_ROWS, INPUT_COLS],
  );
  const tEnc = onStage ? performance.now() : 0;
  const encoderOut = await sessions.encoder.run({ [ENCODER_INPUT_NAME]: encoderInput });
  const encoderFeats = expectTensor(
    encoderOut[ENCODER_OUTPUT_NAME],
    ENCODER_OUTPUT_NAME,
    "float32",
    [NUM_IMAGES, ENCODER_FEATURE_DIM],
  );
  onStage?.("encoder", performance.now() - tEnc);

  // Reshape (N, 512) -> (1, N, 512). Same buffer, new dims tensor.
  const imgFeatsTensor = new sessions.Tensor(
    "float32",
    encoderFeats,
    [1, NUM_IMAGES, ENCODER_FEATURE_DIM],
  );

  // int64 sequences — onnxruntime requires BigInt64Array for int64 tensors.
  const viewSeqData = new BigInt64Array(NUM_IMAGES);
  const sideSeqData = new BigInt64Array(NUM_IMAGES);
  const timeSeqData = new BigInt64Array(NUM_IMAGES); // zeros
  for (let i = 0; i < NUM_IMAGES; i++) {
    viewSeqData[i] = BigInt(preprocessed[i].view);
    sideSeqData[i] = BigInt(preprocessed[i].side);
  }

  const rf = vectorizeRiskFactors(riskFactors);
  if (rf.vector.length !== RF_DIM_EXPECTED || rf.knownMask.length !== RF_DIM_EXPECTED) {
    throw new Error(
      `runMirai: vectorizeRiskFactors returned vector.length=${rf.vector.length}, knownMask.length=${rf.knownMask.length}; expected ${RF_DIM_EXPECTED}`,
    );
  }

  const tRisk = onStage ? performance.now() : 0;
  const riskOut = await sessions.risk.run({
    img_feats: imgFeatsTensor,
    view_seq: new sessions.Tensor("int64", viewSeqData, [1, NUM_IMAGES]),
    side_seq: new sessions.Tensor("int64", sideSeqData, [1, NUM_IMAGES]),
    time_seq: new sessions.Tensor("int64", timeSeqData, [1, NUM_IMAGES]),
    rf_vector: new sessions.Tensor("float32", rf.vector, [1, RF_DIM_EXPECTED]),
    rf_known_mask: new sessions.Tensor("float32", rf.knownMask, [1, RF_DIM_EXPECTED]),
  });
  const logitFlat = expectTensor(riskOut[RISK_OUTPUT_LOGIT], RISK_OUTPUT_LOGIT, "float32", [1, N_YEARS]);
  const hiddenFlat = expectTensor(riskOut[RISK_OUTPUT_HIDDEN], RISK_OUTPUT_HIDDEN, "float32", [1, HIDDEN_DIM]);

  // Copy out of ORT-owned buffers so the result owns independent storage and does not
  // hold a session reference alive (session can be disposed after run returns).
  const rawLogit = new Float32Array(N_YEARS);
  rawLogit.set(logitFlat);
  const embedding = new Float32Array(HIDDEN_DIM);
  embedding.set(hiddenFlat);
  onStage?.("risk", performance.now() - tRisk);

  // sigmoid(-logit) on fp32 values widened to fp64 on read — matches Phase 5's
  // `1.0 / (1.0 + np.exp(-logit.astype(np.float32)))`.
  const tCal = onStage ? performance.now() : 0;
  const rawSigmoid = new Float64Array(N_YEARS);
  for (let i = 0; i < N_YEARS; i++) {
    rawSigmoid[i] = 1.0 / (1.0 + Math.exp(-rawLogit[i]));
  }

  const calibrated = calibrateAll(rawSigmoid, calibrator);
  const predictions: MiraiPredictions = {
    year1: round4dp(calibrated[0]),
    year2: round4dp(calibrated[1]),
    year3: round4dp(calibrated[2]),
    year4: round4dp(calibrated[3]),
    year5: round4dp(calibrated[4]),
  };
  onStage?.("calibrate", performance.now() - tCal);
  onStage?.("total", performance.now() - tTotal);

  return {
    predictions,
    embedding,
    rawLogit,
    rawSigmoid,
    calibrated,
    slotOrder,
    modelVersion: MIRAI_MODEL_VERSION,
  };
}
