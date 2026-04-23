import type { PatientFactors, Relative, RelativeCode } from "../types.js";

type Cancer = "breastCancer" | "ovarianCancer";

function relativeHasCancer(rel: Relative, cancer: Cancer): boolean {
  return rel[cancer] === true;
}

export function relativeCancerHistoryInto(
  patient: PatientFactors,
  code: RelativeCode,
  cancer: Cancer,
  out: Float32Array,
  offset: number,
): void {
  const list = patient.relatives[code];
  for (const rel of list) {
    if (relativeHasCancer(rel, cancer)) {
      out[offset] = 1;
      return;
    }
  }
}
