import type { PatientFactors } from "../types.js";

export function binaryFamilyHistoryInto(
  patient: PatientFactors,
  out: Float32Array,
  offset: number,
): void {
  // Python iterates relatives_dict.items() and sets [0]=1 if any list is
  // non-empty. Our PatientFactors.relatives is fully populated with every
  // code, some possibly empty lists.
  for (const code of Object.keys(patient.relatives)) {
    const list = patient.relatives[code as keyof typeof patient.relatives];
    if (list.length > 0) {
      out[offset] = 1;
      return;
    }
  }
}
