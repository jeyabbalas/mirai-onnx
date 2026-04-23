import { MISSING_VALUE } from "./missing.js";

export function oneHotInto(
  value: number,
  cutoffs: readonly number[],
  out: Float32Array,
  offset: number,
): void {
  if (value === MISSING_VALUE) return;
  for (let i = 0; i < cutoffs.length; i++) {
    const cutoff = cutoffs[i] as number;
    if (value <= cutoff) {
      out[offset + i] = 1;
      return;
    }
  }
  out[offset + cutoffs.length] = 1;
}

export function oneHotFeatureNames(
  name: string,
  cutoffs: readonly number[],
): string[] {
  const n = cutoffs.length + 1;
  const out = new Array<string>(n);
  out[0] = `${name}_lt_${cutoffs[0]}`;
  out[n - 1] = `${name}_gt_${cutoffs[cutoffs.length - 1]}`;
  for (let i = 1; i < cutoffs.length; i++) {
    out[i] = `${name}_${cutoffs[i - 1]}_${cutoffs[i]}`;
  }
  return out;
}
