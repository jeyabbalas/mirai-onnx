import type { ExamFactors } from "../types.js";
import { oneHotInto, oneHotFeatureNames } from "../oneHot.js";

export const AGE_CUTOFFS = [40, 50, 60, 70, 80] as const;
export const WEIGHT_CUTOFFS = [100, 130, 160, 190, 220, 250] as const;
export const HEIGHT_CUTOFFS = [50, 55, 60, 65, 70, 75] as const;

export function ageInto(exam: ExamFactors, out: Float32Array, offset: number): void {
  oneHotInto(exam.age, AGE_CUTOFFS, out, offset);
}

export function weightInto(exam: ExamFactors, out: Float32Array, offset: number): void {
  oneHotInto(exam.weight, WEIGHT_CUTOFFS, out, offset);
}

export function heightInto(exam: ExamFactors, out: Float32Array, offset: number): void {
  oneHotInto(exam.height, HEIGHT_CUTOFFS, out, offset);
}

export const ageFeatureNames = () => oneHotFeatureNames("age", AGE_CUTOFFS);
export const weightFeatureNames = () => oneHotFeatureNames("weight", WEIGHT_CUTOFFS);
export const heightFeatureNames = () => oneHotFeatureNames("height", HEIGHT_CUTOFFS);
