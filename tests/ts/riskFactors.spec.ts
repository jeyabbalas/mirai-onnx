import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";

import { REPO_ROOT } from "./setup.js";
import {
  vectorizeRiskFactors,
  RISK_FACTOR_KEYS,
  RF_KEY_TO_NUM_CLASS,
  RF_KEY_TO_OFFSET,
  RF_DIM,
  FEATURE_NAMES,
  type MiraiRiskFactors,
} from "../../src/mirai/riskFactors/index.js";

interface FixtureCase {
  name: string;
  input: MiraiRiskFactors;
  expected_vector: number[];
  expected_known_mask: number[];
}

interface FixtureFile {
  schema_version: number;
  rf_dim: number;
  risk_factor_keys: string[];
  rf_key_to_num_class: Record<string, number>;
  rf_key_to_offset: Record<string, number>;
  feature_names: string[];
  cases: FixtureCase[];
}

const FIXTURE_PATH = path.join(REPO_ROOT, "tests", "rf", "fixtures.json");
const fixtures = JSON.parse(fs.readFileSync(FIXTURE_PATH, "utf8")) as FixtureFile;

describe("fixture file invariants", () => {
  it("uses the expected schema and dims", () => {
    expect(fixtures.schema_version).toBe(1);
    expect(fixtures.rf_dim).toBe(RF_DIM);
    expect(fixtures.feature_names.length).toBe(RF_DIM);
    expect(fixtures.cases.length).toBeGreaterThanOrEqual(20);
  });

  it("risk_factor_keys match the TS constant (same order)", () => {
    expect(fixtures.risk_factor_keys).toEqual([...RISK_FACTOR_KEYS]);
  });

  it("num_class map matches", () => {
    for (const k of RISK_FACTOR_KEYS) {
      expect(fixtures.rf_key_to_num_class[k]).toBe(RF_KEY_TO_NUM_CLASS[k]);
    }
  });

  it("offset map matches", () => {
    for (const k of RISK_FACTOR_KEYS) {
      expect(fixtures.rf_key_to_offset[k]).toBe(RF_KEY_TO_OFFSET[k]);
    }
  });

  it("feature_names match FEATURE_NAMES verbatim", () => {
    expect([...FEATURE_NAMES]).toEqual(fixtures.feature_names);
  });
});

describe.each(fixtures.cases)("vectorizeRiskFactors($name)", (c) => {
  const result = vectorizeRiskFactors(c.input);

  it("vector matches Python bit-for-bit", () => {
    expect(result.vector.length).toBe(RF_DIM);
    for (let i = 0; i < RF_DIM; i++) {
      if (result.vector[i] !== c.expected_vector[i]) {
        throw new Error(
          `case=${c.name} slot=${i} (${FEATURE_NAMES[i]}): ` +
            `ts=${result.vector[i]} py=${c.expected_vector[i]}`,
        );
      }
    }
  });

  it("known mask matches per-key rule", () => {
    expect(result.knownMask.length).toBe(RF_DIM);
    for (let i = 0; i < RF_DIM; i++) {
      if (result.knownMask[i] !== c.expected_known_mask[i]) {
        throw new Error(
          `case=${c.name} slot=${i} (${FEATURE_NAMES[i]}): ` +
            `ts=${result.knownMask[i]} py=${c.expected_known_mask[i]}`,
        );
      }
    }
  });
});
