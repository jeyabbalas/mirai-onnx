import { describe, it, expect } from "vitest";
import path from "node:path";
import { loadNpy } from "../../src/mirai/util/npy.js";
import { FIXTURES_DIR, DEMO_DICOMS } from "./setup.js";

describe("npy loader", () => {
  it("loads preproc_tensor/CC_L.npy with expected shape and dtype", () => {
    const arr = loadNpy(path.join(FIXTURES_DIR, "preproc_tensor", "CC_L.npy"));
    expect(arr.dtype).toBe("<f4");
    expect(arr.shape).toEqual([3, 2048, 1664]);
    expect(arr.fortranOrder).toBe(false);
    expect(arr.data).toBeInstanceOf(Float32Array);
    expect(arr.data.length).toBe(3 * 2048 * 1664);
    // Channel replication: channel 0 and channel 1 should be byte-identical
    // (torch's .expand is a zero-copy view, so torch.save serializes each
    // channel independently to produce identical bytes).
    const c0 = (arr.data as Float32Array).subarray(0, 2048 * 1664);
    const c1 = (arr.data as Float32Array).subarray(2048 * 1664, 2 * 2048 * 1664);
    for (let i = 0; i < 16; i++) expect(c0[i]).toBe(c1[i]);
  });

  it("loads dicom_raw_uint16 fixtures for all 4 DICOMs", () => {
    for (const d of DEMO_DICOMS) {
      const arr = loadNpy(path.join(FIXTURES_DIR, "dicom_raw_uint16", `${d.label}.npy`));
      expect(arr.dtype, d.label).toBe("<u2");
      expect(arr.shape, d.label).toEqual([3062, 2394]);
      expect(arr.data).toBeInstanceOf(Uint16Array);
      expect(arr.data.length).toBe(3062 * 2394);
    }
  });
});
