import { decodeDicom, type DecodeOptions } from "./dicom.js";
import { resizeBilinearMode1, uint16ToInt32 } from "./resize.js";
import { alignToLeft } from "./alignToLeft.js";
import { normalizeAndExpand, MIRAI_IMG_MEAN, MIRAI_IMG_STD } from "./normalize.js";
import type { PreprocessResult } from "../types.js";

export const MIRAI_INPUT_ROWS = 2048;
export const MIRAI_INPUT_COLS = 1664;

export interface PreprocessOptions extends DecodeOptions {
  mean?: number;
  std?: number;
}

export function preprocessDicom(
  buffer: ArrayBufferLike | Uint8Array,
  opts: PreprocessOptions = {},
): PreprocessResult {
  const decoded = decodeDicom(buffer, opts);
  const wide = uint16ToInt32(decoded.pixels);
  const resized = resizeBilinearMode1(
    wide,
    decoded.rows,
    decoded.cols,
    MIRAI_INPUT_ROWS,
    MIRAI_INPUT_COLS,
  );
  const aligned = alignToLeft(resized, MIRAI_INPUT_ROWS, MIRAI_INPUT_COLS);
  const pixels = normalizeAndExpand(
    aligned.pixels,
    MIRAI_INPUT_ROWS,
    MIRAI_INPUT_COLS,
    opts.mean ?? MIRAI_IMG_MEAN,
    opts.std ?? MIRAI_IMG_STD,
  );
  return { pixels, view: decoded.view, side: decoded.side, flipped: aligned.flipped };
}

export { MIRAI_IMG_MEAN, MIRAI_IMG_STD } from "./normalize.js";
export { decodeDicom } from "./dicom.js";
export type { DecodedDicom, DecodeOptions, WindowMethod } from "./dicom.js";
export { resizeBilinearMode1 } from "./resize.js";
export { alignToLeft } from "./alignToLeft.js";
export { normalizeAndExpand } from "./normalize.js";
