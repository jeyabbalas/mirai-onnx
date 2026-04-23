import { describe, it, expect } from "vitest";
import fs from "node:fs";
import { DEMO_DATA_DIR, FIXTURES_DIR, DEMO_DICOMS } from "./setup.js";

describe("test harness", () => {
  it("can see demo DICOMs", () => {
    for (const d of DEMO_DICOMS) {
      const p = `${DEMO_DATA_DIR}/${d.file}`;
      expect(fs.existsSync(p), `missing demo DICOM: ${p}`).toBe(true);
    }
  });

  it("can see Phase 0 fixtures", () => {
    for (const d of DEMO_DICOMS) {
      const p = `${FIXTURES_DIR}/preproc_tensor/${d.label}.npy`;
      expect(fs.existsSync(p), `missing fixture: ${p}`).toBe(true);
    }
  });
});
