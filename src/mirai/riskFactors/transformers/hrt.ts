import type { ExamFactors, PatientFactors } from "../types.js";
import { MISSING_VALUE } from "../missing.js";
import { oneHotFeatureNames, oneHotInto } from "../oneHot.js";

export const HRT_YEAR_CUTOFFS = [1, 3, 5, 7] as const;

export interface HrtState {
  type: number;              // 0=combined, 1=estrogen, 2=unknown; -1 if none/future
  duration: number;          // -1 if unresolvable
  yearsAgoStopped: number;   // -1 if unresolvable
}

// Direct port of get_hrt_information_transformer in risk_factors.py lines 263-329.
// The three "pieces" (type/duration/years_ago_stopped) all derive from the same
// state machine; we compute it once per call.
export function deriveHrtState(patient: PatientFactors, exam: ExamFactors): HrtState {
  const currentAge = exam.age;

  let type = MISSING_VALUE;
  let duration = MISSING_VALUE;
  let yearsAgoStopped = MISSING_VALUE;

  let firstAge = MISSING_VALUE;
  let lastAge = MISSING_VALUE;
  let extractedDuration = MISSING_VALUE;
  let hasBranch = false;

  if (patient.combined_hrt === 1) {
    type = 0;
    firstAge = patient.combined_hrt_first_age;
    lastAge = patient.combined_hrt_last_age;
    extractedDuration = patient.combined_hrt_duration;
    hasBranch = true;
  } else if (patient.estrogen_hrt === 1) {
    type = 1;
    firstAge = patient.estrogen_hrt_first_age;
    lastAge = patient.estrogen_hrt_last_age;
    extractedDuration = patient.estrogen_hrt_duration;
    hasBranch = true;
  } else if (patient.unknown_hrt === 1) {
    type = 2;
    firstAge = patient.unknown_hrt_first_age;
    lastAge = patient.unknown_hrt_last_age;
    extractedDuration = patient.unknown_hrt_duration;
    hasBranch = true;
  }

  if (hasBranch) {
    if (lastAge >= currentAge && currentAge !== MISSING_VALUE) {
      if (firstAge !== MISSING_VALUE && firstAge > currentAge) {
        // future_user: started in the future
        type = MISSING_VALUE;
      } else if (
        extractedDuration !== MISSING_VALUE &&
        lastAge - extractedDuration > currentAge
      ) {
        // future_user: inconsistent — would start after exam
        type = MISSING_VALUE;
      } else {
        // currentAge is guaranteed != MISSING here; branch reduces to:
        duration = firstAge !== MISSING_VALUE ? currentAge - firstAge : extractedDuration;
      }
    } else if (lastAge !== MISSING_VALUE) {
      yearsAgoStopped = currentAge - lastAge;
      if (extractedDuration !== MISSING_VALUE) {
        duration = extractedDuration;
      } else if (firstAge !== MISSING_VALUE) {
        // lastAge is != MISSING by the elif guard
        duration = lastAge - firstAge;
        if (duration < 0) {
          throw new Error(
            `HRT duration assertion: expected duration >= 0, got ${duration} ` +
              `(firstAge=${firstAge}, lastAge=${lastAge})`,
          );
        }
      }
    } else {
      duration = extractedDuration !== MISSING_VALUE ? extractedDuration : MISSING_VALUE;
    }
  }

  return { type, duration, yearsAgoStopped };
}

export function hrtTypeInto(state: HrtState, out: Float32Array, offset: number): void {
  if (state.type > MISSING_VALUE) {
    out[offset + state.type] = 1;
  }
}

export function hrtDurationInto(state: HrtState, out: Float32Array, offset: number): void {
  oneHotInto(state.duration, HRT_YEAR_CUTOFFS, out, offset);
}

export function hrtYearsAgoStoppedInto(state: HrtState, out: Float32Array, offset: number): void {
  oneHotInto(state.yearsAgoStopped, HRT_YEAR_CUTOFFS, out, offset);
}

export const hrtTypeFeatureNames = () => ["hrt_combined", "hrt_estrogen", "hrt_unknown"];
export const hrtDurationFeatureNames = () =>
  oneHotFeatureNames("hrt_duration", HRT_YEAR_CUTOFFS);
export const hrtYearsAgoStoppedFeatureNames = () =>
  oneHotFeatureNames("hrt_years_ago_stopped", HRT_YEAR_CUTOFFS);
