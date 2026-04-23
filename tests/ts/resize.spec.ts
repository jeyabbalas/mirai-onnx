import { describe, it, expect } from "vitest";
import path from "node:path";
import { loadNpy } from "../../src/mirai/util/npy.js";
import { resizeBilinearMode1, uint16ToInt32 } from "../../src/mirai/preprocess/resize.js";
import { FIXTURES_DIR, DEMO_DICOMS } from "./setup.js";

const IN_ROWS = 3062;
const IN_COLS = 2394;
const OUT_ROWS = 2048;
const OUT_COLS = 1664;

describe.each(DEMO_DICOMS)("resizeBilinearMode1 parity ($label)", (d) => {
  it("matches PIL bilinear within ≤2 LSB vs post_resize fixture", () => {
    const rawU16 = loadNpy(path.join(FIXTURES_DIR, "dicom_raw_uint16", `${d.label}.npy`));
    expect(rawU16.shape).toEqual([IN_ROWS, IN_COLS]);
    const rawI32 = uint16ToInt32(rawU16.data as Uint16Array);

    const result = resizeBilinearMode1(rawI32, IN_ROWS, IN_COLS, OUT_ROWS, OUT_COLS);

    const fixture = loadNpy(path.join(FIXTURES_DIR, "post_resize", `${d.label}.npy`));
    expect(fixture.shape).toEqual([OUT_ROWS, OUT_COLS]);
    expect(fixture.data).toBeInstanceOf(Int32Array);
    const py = fixture.data as Int32Array;
    expect(result.length).toBe(py.length);

    let maxAbs = 0;
    let mismatch = 0;
    let worstIdx = -1;
    for (let i = 0; i < result.length; i++) {
      const d = result[i] - py[i];
      if (d !== 0) mismatch++;
      const a = Math.abs(d);
      if (a > maxAbs) {
        maxAbs = a;
        worstIdx = i;
      }
    }
    // Report even on success so we can eyeball drift in --reporter=verbose.
    console.log(
      `  [${d.label}] maxAbsDiff=${maxAbs} mismatches=${mismatch}/${result.length} ratio=${((mismatch / result.length) * 100).toFixed(3)}%`,
    );
    expect(maxAbs, `worst @${worstIdx}: ts=${result[worstIdx]} py=${py[worstIdx]}`).toBeLessThanOrEqual(2);
  }, 30_000);
});
