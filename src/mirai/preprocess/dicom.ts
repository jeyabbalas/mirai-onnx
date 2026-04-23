import dicomParser, { type DataSet } from "dicom-parser";
import { readViewSide } from "./viewSide.js";
import { applyWindowing, castFloat64ToUint16, type VoiType } from "./windowing.js";
import { readVoiLutSequence, applyVoiLut, scaleToSixteenBit } from "./voiLut.js";
import type { View, Side } from "../types.js";

export type WindowMethod = "minmax" | "auto";

export interface DecodedDicom {
  pixels: Uint16Array; // shape (rows, cols) row-major
  rows: number;
  cols: number;
  view: View;
  side: Side;
  viewStr: string;
  sideStr: string;
  usedBranch: "ge_voi_lut" | "auto_window" | "minmax_window";
}

export interface DecodeOptions {
  windowMethod?: WindowMethod;
  voiLutIndex?: number;
}

const EXPLICIT_VR_LITTLE = "1.2.840.10008.1.2.1";
const IMPLICIT_VR_LITTLE = "1.2.840.10008.1.2";
const EXPLICIT_VR_BIG = "1.2.840.10008.1.2.2";

function readPixelDataUint16(ds: DataSet): Uint16Array {
  const el = ds.elements["x7fe00010"];
  if (!el) throw new Error("PixelData (7FE0,0010) missing");
  if (el.encapsulatedPixelData) {
    throw new Error("Compressed/encapsulated pixel data is not supported in Phase 6");
  }

  const tsUid = ds.string("x00020010");
  const littleEndian = tsUid !== EXPLICIT_VR_BIG;

  const bitsAllocated = ds.uint16("x00280100");
  if (bitsAllocated !== 16) {
    throw new Error(`Unsupported BitsAllocated=${bitsAllocated}; only 16 is supported in Phase 6`);
  }
  const pixelRep = ds.uint16("x00280103");
  if (pixelRep !== 0) {
    throw new Error(`Unsupported PixelRepresentation=${pixelRep}; only unsigned is supported in Phase 6`);
  }

  const numPixels = el.length / 2;
  const out = new Uint16Array(numPixels);
  const byteArray = ds.byteArray;
  const buffer = byteArray.buffer as ArrayBuffer;
  const baseOffset = byteArray.byteOffset + el.dataOffset;

  if (littleEndian && baseOffset % 2 === 0) {
    // Zero-copy view for the common path.
    return new Uint16Array(buffer, baseOffset, numPixels).slice();
  }

  const dv = new DataView(buffer, baseOffset, el.length);
  for (let i = 0; i < numPixels; i++) {
    out[i] = dv.getUint16(i * 2, littleEndian);
  }
  return out;
}

function applyModalityLut(pixels: Uint16Array, slope: number, intercept: number): Float64Array {
  const out = new Float64Array(pixels.length);
  if (slope === 1 && intercept === 0) {
    for (let i = 0; i < pixels.length; i++) out[i] = pixels[i];
  } else {
    for (let i = 0; i < pixels.length; i++) out[i] = pixels[i] * slope + intercept;
  }
  return out;
}

export function decodeDicom(buffer: ArrayBufferLike | Uint8Array, opts: DecodeOptions = {}): DecodedDicom {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const ds = dicomParser.parseDicom(bytes);

  const tsUid = ds.string("x00020010");
  if (tsUid !== undefined && tsUid !== EXPLICIT_VR_LITTLE && tsUid !== IMPLICIT_VR_LITTLE && tsUid !== EXPLICIT_VR_BIG) {
    throw new Error(`Unsupported TransferSyntaxUID=${tsUid}; only uncompressed LE/BE are supported in Phase 6`);
  }

  const rows = ds.uint16("x00280010");
  const cols = ds.uint16("x00280011");
  if (rows === undefined || cols === undefined) throw new Error("Rows/Cols tags missing");

  const { view, side, viewStr, sideStr } = readViewSide(ds);

  const rawPixels = readPixelDataUint16(ds);
  if (rawPixels.length !== rows * cols) {
    throw new Error(`PixelData length ${rawPixels.length} != rows*cols ${rows * cols}`);
  }

  const slope = ds.floatString("x00281053") ?? 1;
  const intercept = ds.floatString("x00281052") ?? 0;
  const modalityOut = applyModalityLut(rawPixels, slope, intercept);

  const voiTypeStr = ds.string("x00281056");
  const voiType: VoiType = voiTypeStr === "SIGMOID" ? "SIGMOID" : "LINEAR";

  const manufacturer = ds.string("x00080070") ?? "";
  const isGe = manufacturer.includes("GE");
  const hasVoiLutSeq = !!ds.elements["x00283010"];

  let pixelsU16: Uint16Array;
  let usedBranch: DecodedDicom["usedBranch"];

  if (isGe && hasVoiLutSeq) {
    const preLut = castFloat64ToUint16(modalityOut);
    const item = readVoiLutSequence(ds, opts.voiLutIndex ?? 0);
    pixelsU16 = applyVoiLut(preLut, item);
    scaleToSixteenBit(pixelsU16, item.numBits);
    usedBranch = "ge_voi_lut";
  } else {
    const windowMethod: WindowMethod = opts.windowMethod ?? "minmax";
    if (windowMethod === "auto") {
      let center = -600;
      let width = 1500;
      const wc = ds.floatString("x00281050");
      const ww = ds.floatString("x00281051");
      if (wc !== undefined && ww !== undefined) {
        center = wc;
        width = ww;
      }
      applyWindowing(modalityOut, center, width, voiType);
      usedBranch = "auto_window";
    } else {
      let mn = Infinity;
      let mx = -Infinity;
      for (let i = 0; i < modalityOut.length; i++) {
        const v = modalityOut[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      const center = (mn + mx + 1) / 2;
      const width = mx - mn + 1;
      applyWindowing(modalityOut, center, width, voiType);
      usedBranch = "minmax_window";
    }
    pixelsU16 = castFloat64ToUint16(modalityOut);
  }

  return { pixels: pixelsU16, rows, cols, view, side, viewStr, sideStr, usedBranch };
}
