import type { ExamFactors, PatientFactors } from "../types.js";
import { MISSING_VALUE } from "../missing.js";

export function parousInto(
  patient: PatientFactors,
  exam: ExamFactors,
  out: Float32Array,
  offset: number,
): void {
  // Port of transform_parous: `num_births != -1` sets [0]=1; then if
  // `first_pregnancy_age != -1`, OVERWRITE with (first_pregnancy_age < exam_age).
  // Note: the second branch overwrites unconditionally — can flip a 1 back to 0.
  let bit = 0;
  if (patient.num_births !== MISSING_VALUE) bit = 1;
  if (patient.first_pregnancy_age !== MISSING_VALUE) {
    bit = patient.first_pregnancy_age < exam.age ? 1 : 0;
  }
  if (bit === 1) out[offset] = 1;
}
