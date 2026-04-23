import type { ExamFactors, PatientFactors } from "../types.js";
import { MISSING_VALUE } from "../missing.js";
import { oneHotFeatureNames, oneHotInto } from "../oneHot.js";

export const MENARCHE_CUTOFFS = [10, 12, 14, 16] as const;
export const MENOPAUSE_CUTOFFS = [45, 50, 55, 60] as const;
export const FIRST_PREGNANCY_CUTOFFS = [20, 25, 30, 35, 40] as const;
export const OVARIAN_CANCER_AGE_CUTOFFS = [30, 40, 50, 60, 70] as const;

function ageBasedInto(
  value: number,
  examAge: number,
  cutoffs: readonly number[],
  out: Float32Array,
  offset: number,
): void {
  let v = value;
  if (examAge !== MISSING_VALUE && examAge < v) {
    v = MISSING_VALUE;
  }
  oneHotInto(v, cutoffs, out, offset);
}

export function menarcheAgeInto(
  patient: PatientFactors,
  exam: ExamFactors,
  out: Float32Array,
  offset: number,
): void {
  ageBasedInto(patient.menarche_age, exam.age, MENARCHE_CUTOFFS, out, offset);
}

export function menopauseAgeInto(
  patient: PatientFactors,
  exam: ExamFactors,
  out: Float32Array,
  offset: number,
): void {
  ageBasedInto(patient.menopause_age, exam.age, MENOPAUSE_CUTOFFS, out, offset);
}

export function firstPregnancyAgeInto(
  patient: PatientFactors,
  exam: ExamFactors,
  out: Float32Array,
  offset: number,
): void {
  ageBasedInto(patient.first_pregnancy_age, exam.age, FIRST_PREGNANCY_CUTOFFS, out, offset);
}

export function ovarianCancerAgeInto(
  patient: PatientFactors,
  exam: ExamFactors,
  out: Float32Array,
  offset: number,
): void {
  ageBasedInto(patient.ovarian_cancer_age, exam.age, OVARIAN_CANCER_AGE_CUTOFFS, out, offset);
}

export const menarcheFeatureNames = () => oneHotFeatureNames("menarche_age", MENARCHE_CUTOFFS);
export const menopauseAgeFeatureNames = () => oneHotFeatureNames("menopause_age", MENOPAUSE_CUTOFFS);
export const firstPregnancyFeatureNames = () =>
  oneHotFeatureNames("first_pregnancy_age", FIRST_PREGNANCY_CUTOFFS);
export const ovarianCancerAgeFeatureNames = () =>
  oneHotFeatureNames("ovarian_cancer_age", OVARIAN_CANCER_AGE_CUTOFFS);
