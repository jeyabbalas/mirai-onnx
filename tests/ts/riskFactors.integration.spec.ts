import { describe, it, expect } from "vitest";
import {
  vectorizeRiskFactors,
  RISK_FACTOR_KEYS,
  RF_KEY_TO_NUM_CLASS,
  RF_KEY_TO_OFFSET,
  RF_DIM,
  FEATURE_NAMES,
} from "../../src/mirai/riskFactors/index.js";

describe("RF constants", () => {
  it("RF_DIM = 100 and num_class sums to 100", () => {
    expect(RF_DIM).toBe(100);
    const sum = RISK_FACTOR_KEYS.reduce((s, k) => s + RF_KEY_TO_NUM_CLASS[k], 0);
    expect(sum).toBe(RF_DIM);
  });

  it("RF_KEY_TO_OFFSET is cumulative over RISK_FACTOR_KEYS", () => {
    let off = 0;
    for (const k of RISK_FACTOR_KEYS) {
      expect(RF_KEY_TO_OFFSET[k]).toBe(off);
      off += RF_KEY_TO_NUM_CLASS[k];
    }
    expect(off).toBe(RF_DIM);
  });

  it("FEATURE_NAMES has 100 entries", () => {
    expect(FEATURE_NAMES.length).toBe(RF_DIM);
  });

  it("RISK_FACTOR_KEYS contains exactly 34 entries", () => {
    expect(RISK_FACTOR_KEYS.length).toBe(34);
  });
});

describe("vectorizeRiskFactors — empty/default input", () => {
  it("returns all-zero vector and mask on no input (overloaded)", () => {
    const r = vectorizeRiskFactors();
    for (let i = 0; i < RF_DIM; i++) {
      expect(r.vector[i]).toBe(0);
      expect(r.knownMask[i]).toBe(0);
    }
  });

  it("returns all-zero vector and mask on {} input", () => {
    const r = vectorizeRiskFactors({});
    for (let i = 0; i < RF_DIM; i++) {
      expect(r.vector[i]).toBe(0);
      expect(r.knownMask[i]).toBe(0);
    }
  });

  it("each call returns a fresh Float32Array", () => {
    const a = vectorizeRiskFactors();
    const b = vectorizeRiskFactors();
    expect(a.vector).not.toBe(b.vector);
    expect(a.knownMask).not.toBe(b.knownMask);
    a.vector[0] = 1;
    expect(b.vector[0]).toBe(0);
  });
});

describe("vectorizeRiskFactors — targeted slot probes", () => {
  it("brca2: true sets slot 74 and marks 4-wide brca block known", () => {
    const r = vectorizeRiskFactors({ brca2: true });
    const off = RF_KEY_TO_OFFSET["brca"];
    expect(off).toBe(71);
    expect(r.vector[off + 3]).toBe(1);
    expect(r.vector[off + 0]).toBe(0);
    expect(r.vector[off + 1]).toBe(0);
    expect(r.vector[off + 2]).toBe(0);
    for (let i = off; i < off + 4; i++) expect(r.knownMask[i]).toBe(1);
  });

  it("age: 45 sets age_40_50 bucket", () => {
    const r = vectorizeRiskFactors({ age: 45 });
    const off = RF_KEY_TO_OFFSET["age"];
    expect(off).toBe(8);
    expect(r.vector[off + 1]).toBe(1);  // age_40_50
    for (let i = off; i < off + 6; i++) expect(r.knownMask[i]).toBe(1);
  });

  it("priorHist: true sets slot 30", () => {
    const r = vectorizeRiskFactors({ priorHist: true });
    expect(RF_KEY_TO_OFFSET["prior_hist"]).toBe(30);
    expect(r.vector[30]).toBe(1);
    expect(r.knownMask[30]).toBe(1);
  });

  it("priorHist: false keeps slot 30 at zero but marks it known", () => {
    const r = vectorizeRiskFactors({ priorHist: false });
    expect(r.vector[30]).toBe(0);
    expect(r.knownMask[30]).toBe(1);
  });
});
