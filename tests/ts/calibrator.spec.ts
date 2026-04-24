import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";

import { REPO_ROOT, FIXTURES_DIR } from "./setup.js";
import { loadNpy } from "../../src/mirai/util/npy.js";
import {
  loadCalibrator,
  calibrateAll,
  CALIBRATOR_N_YEARS,
  CALIBRATOR_SCHEMA_VERSION,
} from "../../src/mirai/calibrator.js";
import { loadCalibratorFromFile } from "../../src/mirai/calibrator.node.js";

const CALIBRATOR_PATH = path.join(REPO_ROOT, "models", "calibrator.json");

describe("loadCalibrator shape + validation", () => {
  const calibrator = loadCalibratorFromFile(CALIBRATOR_PATH);

  it("has the expected schema_version and 5 year entries", () => {
    expect(calibrator.schema_version).toBe(CALIBRATOR_SCHEMA_VERSION);
    expect(calibrator.years).toHaveLength(CALIBRATOR_N_YEARS);
  });

  it("exposes year indices 0..4 in order", () => {
    for (let i = 0; i < CALIBRATOR_N_YEARS; i++) {
      expect(calibrator.years[i].index).toBe(i);
    }
  });

  it("all four Platt scalars are finite per year", () => {
    for (const y of calibrator.years) {
      expect(Number.isFinite(y.base_slope)).toBe(true);
      expect(Number.isFinite(y.base_offset)).toBe(true);
      expect(Number.isFinite(y.calibrator_slope)).toBe(true);
      expect(Number.isFinite(y.calibrator_offset)).toBe(true);
    }
  });

  it("rejects wrong schema_version", () => {
    const base = JSON.parse(fs.readFileSync(CALIBRATOR_PATH, "utf8"));
    expect(() => loadCalibrator({ ...base, schema_version: 99 })).toThrow(
      /schema_version/,
    );
  });

  it("rejects truncated years array", () => {
    const base = JSON.parse(fs.readFileSync(CALIBRATOR_PATH, "utf8"));
    expect(() =>
      loadCalibrator({ ...base, years: base.years.slice(0, 3) }),
    ).toThrow(/expected 5 year entries/);
  });

  it("rejects duplicate year indices", () => {
    const base = JSON.parse(fs.readFileSync(CALIBRATOR_PATH, "utf8"));
    const dup = { ...base, years: [base.years[0], base.years[0], base.years[2], base.years[3], base.years[4]] };
    expect(() => loadCalibrator(dup)).toThrow(/duplicate year index/);
  });
});

describe("calibrateAll vs Phase 0 fixture", () => {
  const calibrator = loadCalibratorFromFile(CALIBRATOR_PATH);
  const rawSigmoid = loadNpy(path.join(FIXTURES_DIR, "raw_sigmoid.npy"));
  const calibratedFixture = loadNpy(path.join(FIXTURES_DIR, "calibrated.npy"));

  it("fixtures have the expected dtypes and shapes", () => {
    expect(rawSigmoid.dtype).toBe("<f4");
    expect(rawSigmoid.shape).toEqual([1, 5]);
    expect(calibratedFixture.dtype).toBe("<f8");
    expect(calibratedFixture.shape).toEqual([5]);
  });

  it("reproduces calibrated fixture within atol=1e-9 (same-math fp64 parity)", () => {
    const sigmoid = rawSigmoid.data as Float32Array;
    // raw_sigmoid.npy is (1, 5); feed the single row.
    const row = new Float64Array(5);
    for (let i = 0; i < 5; i++) row[i] = sigmoid[i];

    const got = calibrateAll(row, calibrator);
    const want = calibratedFixture.data as Float64Array;
    expect(got.length).toBe(5);

    let maxAbs = 0;
    let worstIdx = -1;
    for (let i = 0; i < 5; i++) {
      const d = Math.abs(got[i] - want[i]);
      if (d > maxAbs) {
        maxAbs = d;
        worstIdx = i;
      }
    }
    console.log(`  calibrator maxAbsDiff=${maxAbs.toExponential(3)} @year${worstIdx + 1}`);
    expect(maxAbs).toBeLessThan(1e-9);
  });

  it("rounded to 4dp matches predictions.json pinned pydicom baseline", () => {
    const sigmoid = rawSigmoid.data as Float32Array;
    const row = new Float64Array(5);
    for (let i = 0; i < 5; i++) row[i] = sigmoid[i];
    const got = calibrateAll(row, calibrator);
    const rounded = Array.from(got).map((v) => Math.round(v * 1e4) / 1e4);
    expect(rounded).toEqual([0.0314, 0.0505, 0.0711, 0.0935, 0.1052]);
  });

  it("calibrateAll rejects length mismatch", () => {
    expect(() => calibrateAll(new Float64Array(4), calibrator)).toThrow(
      /does not match/,
    );
  });
});
