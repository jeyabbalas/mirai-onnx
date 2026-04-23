import type { DataSet } from "dicom-parser";

export interface VoiLutItem {
  numEntries: number;
  firstMap: number;
  numBits: number;
  lutData: Uint16Array;
}

export function readVoiLutSequence(ds: DataSet, index = 0): VoiLutItem {
  const seq = ds.elements["x00283010"];
  if (!seq) throw new Error("VOI LUT Sequence (0028,3010) not present");
  if (!seq.items || seq.items.length === 0) throw new Error("VOI LUT Sequence empty");
  if (index < 0 || index >= seq.items.length) {
    throw new Error(`VOI LUT index ${index} out of range [0, ${seq.items.length})`);
  }

  const itemDs = seq.items[index].dataSet;
  if (!itemDs) throw new Error(`VOI LUT item ${index} has no nested dataSet`);

  const d0 = itemDs.uint16("x00283002", 0);
  const d1 = itemDs.uint16("x00283002", 1);
  const d2 = itemDs.uint16("x00283002", 2);
  if (d0 === undefined || d1 === undefined || d2 === undefined) {
    throw new Error("VOI LUT Descriptor (0028,3002) incomplete");
  }
  const numEntries = d0 === 0 ? 65536 : d0;
  const firstMap = d1;
  const numBits = d2;

  const lutDataEl = itemDs.elements["x00283006"];
  if (!lutDataEl) throw new Error("VOI LUT Data (0028,3006) missing");
  const expectedBytes = numEntries * 2;
  if (lutDataEl.length < expectedBytes) {
    throw new Error(
      `VOI LUT Data too short: expected ${expectedBytes} bytes, got ${lutDataEl.length}`,
    );
  }

  const parser = itemDs.byteArrayParser;
  const lutData = new Uint16Array(numEntries);
  for (let i = 0; i < numEntries; i++) {
    lutData[i] = parser.readUint16(itemDs.byteArray, lutDataEl.dataOffset + i * 2);
  }

  return { numEntries, firstMap, numBits, lutData };
}

export function applyVoiLut(pixels: Uint16Array, item: VoiLutItem): Uint16Array {
  const { numEntries, firstMap, lutData } = item;
  const lastMap = firstMap + numEntries - 1;
  const out = new Uint16Array(pixels.length);
  for (let i = 0; i < pixels.length; i++) {
    let p = pixels[i];
    if (p < firstMap) p = firstMap;
    else if (p > lastMap) p = lastMap;
    out[i] = lutData[p - firstMap];
  }
  return out;
}

/** Multiply in-place by 2^(16 - numBits), wrapping in uint16 per numpy semantics. */
export function scaleToSixteenBit(image: Uint16Array, numBits: number): void {
  const factor = 1 << (16 - numBits);
  for (let i = 0; i < image.length; i++) {
    image[i] = (image[i] * factor) & 0xffff;
  }
}
