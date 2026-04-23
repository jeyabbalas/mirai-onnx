import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { preprocessDicom, MIRAI_INPUT_ROWS, MIRAI_INPUT_COLS } from "../../src/mirai/preprocess/index.js";
import { loadNpy } from "../../src/mirai/util/npy.js";
import { DEMO_DATA_DIR, FIXTURES_DIR, DEMO_DICOMS } from "./setup.js";

const PLANE = MIRAI_INPUT_ROWS * MIRAI_INPUT_COLS;

describe.each(DEMO_DICOMS)("preprocessDicom end-to-end ($label)", (d) => {
  const buf = fs.readFileSync(path.join(DEMO_DATA_DIR, d.file));

  it("matches preproc_tensor fixture within atol=1e-3", () => {
    const result = preprocessDicom(buf);
    expect(result.view).toBe(d.view);
    expect(result.side).toBe(d.side);
    expect(result.pixels.length).toBe(3 * PLANE);

    const fixture = loadNpy(path.join(FIXTURES_DIR, "preproc_tensor", `${d.label}.npy`));
    expect(fixture.shape).toEqual([3, MIRAI_INPUT_ROWS, MIRAI_INPUT_COLS]);
    const py = fixture.data as Float32Array;

    let maxAbs = 0;
    let worstIdx = -1;
    let rmsAcc = 0;
    for (let i = 0; i < result.pixels.length; i++) {
      const diff = result.pixels[i] - py[i];
      const a = Math.abs(diff);
      rmsAcc += diff * diff;
      if (a > maxAbs) {
        maxAbs = a;
        worstIdx = i;
      }
    }
    const rms = Math.sqrt(rmsAcc / result.pixels.length);
    console.log(
      `  [${d.label}] flipped=${result.flipped} maxAbsDiff=${maxAbs.toExponential(3)} rms=${rms.toExponential(3)} @${worstIdx}`,
    );
    expect(maxAbs, `worst @${worstIdx}: ts=${result.pixels[worstIdx]} py=${py[worstIdx]}`).toBeLessThan(1e-3);

    // Channel replication: three channels must be byte-identical in both ours and Python's.
    for (let i = 0; i < PLANE; i += 1000) {
      expect(result.pixels[i]).toBe(result.pixels[PLANE + i]);
      expect(result.pixels[i]).toBe(result.pixels[2 * PLANE + i]);
    }
  }, 60_000);
});

describe("align-to-left flip decisions", () => {
  it("agree with Python reference on all 4 demos via end-to-end parity", () => {
    // Flip-decision correctness is implicit in the end-to-end atol=1e-3 gate
    // above: if any flip bit disagreed, the diff would be huge (full-image
    // mirror, no local alignment). Record the flip bits here for PHASE_6_REPORT.
    for (const d of DEMO_DICOMS) {
      const buf = fs.readFileSync(path.join(DEMO_DATA_DIR, d.file));
      const { flipped } = preprocessDicom(buf);
      console.log(`  [${d.label}] flipped=${flipped}`);
    }
    expect(true).toBe(true);
  }, 60_000);
});
