import type { ExamFactors, PatientFactors } from "../types.js";

export function priorHistInto(
  _patient: PatientFactors,
  exam: ExamFactors,
  out: Float32Array,
  offset: number,
): void {
  if (exam.prior_hist === 1) out[offset] = 1;
}

export function ashkenaziInto(
  patient: PatientFactors,
  _exam: ExamFactors,
  out: Float32Array,
  offset: number,
): void {
  if (patient.ashkenazi === 1) out[offset] = 1;
}
