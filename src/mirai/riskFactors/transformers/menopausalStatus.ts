import type { ExamFactors, PatientFactors } from "../types.js";
import { MISSING_VALUE } from "../missing.js";

// Python uses NEGATIVE_99 = -99 internally as a sentinel distinct from -1
// (the MISSING_VALUE). When menopause_age is -1, age_at_menopause becomes
// -99 and the transformer falls through to the default (index 3 = unknown).
const NEGATIVE_99 = -99;

// Python's TREAT_MISSING_AS_NEGATIVE is False at module-load time in risk_factors.py;
// do not flip this without re-running Phase 0.
const TREAT_MISSING_AS_NEGATIVE = false;

export function menopausalStatusInto(
  patient: PatientFactors,
  exam: ExamFactors,
  out: Float32Array,
  offset: number,
): void {
  const examAge = exam.age;
  const rawMenopauseAge = patient.menopause_age;
  const ageAtMenopause = rawMenopauseAge !== MISSING_VALUE ? rawMenopauseAge : NEGATIVE_99;

  let status = 3;  // default: unknown
  if (ageAtMenopause !== NEGATIVE_99) {
    if (ageAtMenopause < examAge) status = 2;  // post
    else if (ageAtMenopause === examAge) status = 1;  // peri
    else if (ageAtMenopause > examAge) status = 0;  // pre
  } else if (TREAT_MISSING_AS_NEGATIVE) {
    status = 0;  // pre
  }

  out[offset + status] = 1;
}

export function menopausalStatusFeatureNames(): string[] {
  return ["pre", "peri", "post", "unknown"];
}
