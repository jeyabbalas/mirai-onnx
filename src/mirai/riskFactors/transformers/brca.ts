import type { PatientFactors } from "../types.js";

export function brcaInto(patient: PatientFactors, out: Float32Array, offset: number): void {
  // Python logic: brca2==1 → 3; elif brca1==1 → 2; elif brca1==0 → 1; else 0.
  let idx = 0;
  if (patient.brca2 === 1) idx = 3;
  else if (patient.brca1 === 1) idx = 2;
  else if (patient.brca1 === 0) idx = 1;
  out[offset + idx] = 1;
}

export function brcaFeatureNames(): string[] {
  return ["never or unknown", "negative result", "brca1", "brca2"];
}
