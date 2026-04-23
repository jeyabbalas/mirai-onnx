import type { PatientFactors } from "../types.js";

export const RACE_CODE_TO_NAME: Record<number, string> = {
  1: "White",
  2: "African American",
  3: "American Indian, Eskimo, Aleut",
  4: "Asian or Pacific Islander",
  5: "Other Race",
  6: "Caribbean/West Indian",
  7: "Unknown",
  8: "Hispanic",
  9: "Chinese",
  10: "Japanese",
  11: "Filipino",
  12: "Hawaiian",
  13: "Other Asian",
};

export function raceInto(patient: PatientFactors, out: Float32Array, offset: number): void {
  const r = patient.race;
  if (r < 1 || r > 13) return;
  out[offset + (r - 1)] = 1;
}

export function raceFeatureNames(): string[] {
  return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13].map((i) => RACE_CODE_TO_NAME[i] as string);
}
