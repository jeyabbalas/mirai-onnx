import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { decodeDicom } from "../../src/mirai/preprocess/dicom.js";
import { loadNpy } from "../../src/mirai/util/npy.js";
import { DEMO_DATA_DIR, FIXTURES_DIR, DEMO_DICOMS } from "./setup.js";

describe.each(DEMO_DICOMS)("decodeDicom($label)", (d) => {
  const buf = fs.readFileSync(path.join(DEMO_DATA_DIR, d.file));

  it("produces bit-exact uint16 output vs dicom_raw_uint16 fixture", () => {
    const result = decodeDicom(buf);
    expect(result.rows).toBe(3062);
    expect(result.cols).toBe(2394);
    expect(result.usedBranch).toBe("ge_voi_lut");

    const fixture = loadNpy(path.join(FIXTURES_DIR, "dicom_raw_uint16", `${d.label}.npy`));
    expect(fixture.data).toBeInstanceOf(Uint16Array);
    expect(result.pixels.length).toBe(fixture.data.length);

    // Count mismatches; for a bit-exact gate, must be zero.
    let mismatch = 0;
    let worstIdx = -1;
    let worstDiff = 0;
    const ts = result.pixels;
    const py = fixture.data as Uint16Array;
    for (let i = 0; i < ts.length; i++) {
      const d = ts[i] - py[i];
      if (d !== 0) {
        mismatch++;
        if (Math.abs(d) > worstDiff) {
          worstDiff = Math.abs(d);
          worstIdx = i;
        }
      }
    }
    expect(
      mismatch,
      mismatch === 0
        ? ""
        : `mismatches=${mismatch}/${ts.length} worstDiff=${worstDiff} worstIdx=${worstIdx} ts=${ts[worstIdx]} py=${py[worstIdx]}`,
    ).toBe(0);
  });

  it("detects view and side matching the fixture", () => {
    const result = decodeDicom(buf);
    expect({ view: result.view, side: result.side, viewStr: result.viewStr, sideStr: result.sideStr })
      .toEqual({ view: d.view, side: d.side, viewStr: d.viewStr, sideStr: d.sideStr });
  });
});

describe("batch_order.json consistency", () => {
  it("DEMO_DICOMS order matches batch_order.json verbatim", () => {
    const batchOrder = JSON.parse(
      fs.readFileSync(path.join(FIXTURES_DIR, "batch_order.json"), "utf8"),
    ) as Array<{ slot: number; view: number; side: number; view_str: string; side_str: string }>;
    expect(batchOrder.length).toBe(DEMO_DICOMS.length);
    for (let i = 0; i < batchOrder.length; i++) {
      expect(batchOrder[i].slot).toBe(i);
      expect(batchOrder[i].view).toBe(DEMO_DICOMS[i].view);
      expect(batchOrder[i].side).toBe(DEMO_DICOMS[i].side);
      expect(batchOrder[i].view_str).toBe(DEMO_DICOMS[i].viewStr);
      expect(batchOrder[i].side_str).toBe(DEMO_DICOMS[i].sideStr);
    }
  });
});
