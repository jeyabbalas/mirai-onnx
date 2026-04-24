// Phase 9: plan-named convenience wrappers over `runMirai`. These exist so
// downstream callers can use the names from `mirai-migration-plan.md` §10.2
// (`predictMiraiRisk`, `getMiraiEmbedding`) without forcing a module-level
// singleton. Sessions and calibrator stay caller-managed.

import { runMirai } from "./runMirai.js";
import type {
  MiraiResult,
  MiraiRunOptions,
  MiraiSessions,
} from "./runMirai.js";
import type { Calibrator } from "./calibrator.js";
import type { MiraiRiskFactors } from "./riskFactors/index.js";

/**
 * Run the full Mirai pipeline on four DICOMs and return predictions, the
 * pre-hazard embedding (XAI surface), and raw intermediates.
 *
 * Alias of {@link runMirai}. See that function for full documentation of the
 * inputs and outputs. This name mirrors `mirai-migration-plan.md` §10.2.
 *
 * @public
 */
export const predictMiraiRisk: typeof runMirai = runMirai;

/**
 * Return only the 612-dimensional post-ReLU pre-hazard embedding for four DICOMs.
 * Convenience for downstream tooling (e.g. XAI, cohort projection) that does not
 * need the calibrated predictions.
 *
 * The embedding has shape `(612,)` fp32 and matches Phase 0's `xai_hidden.npy`
 * within `atol=2e-5` when run against the same ONNX models on CPU/WASM. It is the
 * concatenation of the 512-d image feature (post-transformer, post-pool) and the
 * 100-d risk-factor vector (user-supplied where `knownMask=1`, else
 * model-predicted), after an in-place ReLU.
 *
 * @param files - Exactly four DICOM byte buffers in the caller's chosen slot order.
 * @param sessions - ONNX sessions + Tensor constructor. Build via
 *   {@link createNodeSessions} or {@link createWebSessions}.
 * @param calibrator - The loaded calibrator JSON. Not used for the embedding
 *   itself but required because {@link runMirai} runs the full pipeline.
 * @param riskFactors - Optional user-supplied risk factors. Missing keys fall back
 *   to the model-predicted path.
 * @returns The 612-d embedding as an independent Float32Array (the session's
 *   output buffer is copied out).
 *
 * @public
 * @see PHASE_0_REPORT.md for the provenance of the 612-d post-ReLU definition.
 */
export async function getMiraiEmbedding(
  files: ReadonlyArray<ArrayBufferLike | Uint8Array>,
  sessions: MiraiSessions,
  calibrator: Calibrator,
  riskFactors?: MiraiRiskFactors,
  options?: MiraiRunOptions,
): Promise<Float32Array> {
  const result: MiraiResult = await runMirai(
    files,
    sessions,
    calibrator,
    riskFactors ?? {},
    options,
  );
  return result.embedding;
}
