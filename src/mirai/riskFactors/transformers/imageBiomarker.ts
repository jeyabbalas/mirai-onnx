import type { ExamFactors } from "../types.js";
import { MISSING_VALUE } from "../missing.js";

export function densityInto(exam: ExamFactors, out: Float32Array, offset: number): void {
  const v = exam.density;
  if (v === MISSING_VALUE) return;
  if (v < 1 || v > 4) return;
  out[offset + (v - 1)] = 1;
}

export function densityFeatureNames(): string[] {
  return ["density_1", "density_2", "density_3", "density_4"];
}
