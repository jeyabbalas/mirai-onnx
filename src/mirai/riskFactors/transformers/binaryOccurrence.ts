import type { ExamFactors, PatientFactors } from "../types.js";
import { MISSING_VALUE } from "../missing.js";

function binaryOccurrenceInto(
  occurrence: number,
  occurrenceAge: number,
  examAge: number,
  out: Float32Array,
  offset: number,
): void {
  // Python's condition is `if occurence and (occurence_age == MISSING_VALUE or
  // exam_age >= occurence_age)`. `occurence` is the raw int — truthy for any
  // non-zero value including -1. Our orchestrator only calls this when the
  // user supplied the boolean field, so occurrence is guaranteed to be 0 or 1.
  if (occurrence === 1 && (occurrenceAge === MISSING_VALUE || examAge >= occurrenceAge)) {
    out[offset] = 1;
  }
}

export function biopsyBenignInto(
  patient: PatientFactors,
  exam: ExamFactors,
  out: Float32Array,
  offset: number,
): void {
  binaryOccurrenceInto(
    patient.biopsy_hyperplasia,
    patient.biopsy_hyperplasia_age,
    exam.age,
    out,
    offset,
  );
}

export function biopsyLCISInto(
  patient: PatientFactors,
  exam: ExamFactors,
  out: Float32Array,
  offset: number,
): void {
  binaryOccurrenceInto(
    patient.biopsy_LCIS,
    patient.biopsy_LCIS_age,
    exam.age,
    out,
    offset,
  );
}

export function biopsyAtypicalHyperplasiaInto(
  patient: PatientFactors,
  exam: ExamFactors,
  out: Float32Array,
  offset: number,
): void {
  binaryOccurrenceInto(
    patient.biopsy_atypical_hyperplasia,
    patient.biopsy_atypical_hyperplasia_age,
    exam.age,
    out,
    offset,
  );
}

export function ovarianCancerInto(
  patient: PatientFactors,
  exam: ExamFactors,
  out: Float32Array,
  offset: number,
): void {
  binaryOccurrenceInto(
    patient.ovarian_cancer,
    patient.ovarian_cancer_age,
    exam.age,
    out,
    offset,
  );
}
