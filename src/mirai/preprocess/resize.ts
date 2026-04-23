/**
 * Port of PIL's BILINEAR resample for mode 'I' (int32) images.
 *
 * Tracks `src/libImaging/Resample.c` in Pillow 9.0.0:
 *   - `bilinear_filter` with support = 1.0
 *   - `precompute_coeffs` widens the filter footprint to `support * max(scale, 1.0)`
 *     on downscale (our case for 3062→2048 and 2394→1664).
 *   - The horizontal and vertical 32bpc passes accumulate in fp64 and write int32
 *     using `ROUND_UP(f)` which is "round half away from zero".
 *
 * The two passes are executed separately (horizontal first, then vertical),
 * so the intermediate image is rounded to int32 between passes — matching PIL
 * byte-for-byte in most cases within ±1 LSB.
 */

export interface Coeffs {
  ksize: number;
  bounds: Int32Array; // length = 2 * outSize; [xmin, xmax] pairs
  coeffs: Float64Array; // length = outSize * ksize
}

function bilinearFilter(x: number): number {
  if (x < 0) x = -x;
  return x < 1 ? 1 - x : 0;
}

function roundUp(f: number): number {
  // PIL's ROUND_UP: round half away from zero.
  return f >= 0 ? Math.trunc(f + 0.5) : Math.trunc(f - 0.5);
}

export function precomputeCoeffs(inSize: number, outSize: number): Coeffs {
  const BILINEAR_SUPPORT = 1.0;

  const scale = inSize / outSize;
  const filterscale = scale < 1.0 ? 1.0 : scale;
  const support = BILINEAR_SUPPORT * filterscale;
  const ksize = Math.trunc(Math.ceil(support)) * 2 + 1;

  const bounds = new Int32Array(outSize * 2);
  const coeffs = new Float64Array(outSize * ksize);

  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1.0 / filterscale;

    // PIL uses C-style truncation on (float + 0.5) for these rounds.
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    const count = xmax - xmin;

    const kBase = xx * ksize;
    let ww = 0;
    for (let x = 0; x < count; x++) {
      const w = bilinearFilter((x + xmin - center + 0.5) * ss);
      coeffs[kBase + x] = w;
      ww += w;
    }
    if (ww !== 0) {
      for (let x = 0; x < count; x++) coeffs[kBase + x] /= ww;
    }
    // Remaining slots stay at 0 by virtue of Float64Array init.
    bounds[xx * 2 + 0] = xmin;
    bounds[xx * 2 + 1] = count;
  }

  return { ksize, bounds, coeffs };
}

/** Horizontal pass: (inRows, inCols) int32 → (inRows, outCols) int32. */
function resampleHorizontal(
  input: Int32Array,
  inRows: number,
  inCols: number,
  outCols: number,
  coeffs: Coeffs,
): Int32Array {
  const { ksize, bounds, coeffs: kk } = coeffs;
  const out = new Int32Array(inRows * outCols);
  for (let yy = 0; yy < inRows; yy++) {
    const rowIn = yy * inCols;
    const rowOut = yy * outCols;
    for (let xx = 0; xx < outCols; xx++) {
      const xmin = bounds[xx * 2 + 0];
      const xmax = bounds[xx * 2 + 1];
      const kBase = xx * ksize;
      let ss = 0;
      for (let x = 0; x < xmax; x++) {
        ss += input[rowIn + xmin + x] * kk[kBase + x];
      }
      out[rowOut + xx] = roundUp(ss);
    }
  }
  return out;
}

/** Vertical pass: (inRows, cols) int32 → (outRows, cols) int32. */
function resampleVertical(
  input: Int32Array,
  cols: number,
  outRows: number,
  coeffs: Coeffs,
): Int32Array {
  const { ksize, bounds, coeffs: kk } = coeffs;
  const out = new Int32Array(outRows * cols);
  for (let yy = 0; yy < outRows; yy++) {
    const ymin = bounds[yy * 2 + 0];
    const ymax = bounds[yy * 2 + 1];
    const kBase = yy * ksize;
    const rowOut = yy * cols;
    for (let xx = 0; xx < cols; xx++) {
      let ss = 0;
      for (let y = 0; y < ymax; y++) {
        ss += input[(ymin + y) * cols + xx] * kk[kBase + y];
      }
      out[rowOut + xx] = roundUp(ss);
    }
  }
  return out;
}

/** Full two-pass bilinear resize for mode-'I' (int32) images. */
export function resizeBilinearMode1(
  input: Int32Array,
  inRows: number,
  inCols: number,
  outRows: number,
  outCols: number,
): Int32Array {
  if (input.length !== inRows * inCols) {
    throw new Error(`resizeBilinearMode1: input length ${input.length} != ${inRows}*${inCols}`);
  }
  const xCoeffs = precomputeCoeffs(inCols, outCols);
  const horz = resampleHorizontal(input, inRows, inCols, outCols, xCoeffs);
  const yCoeffs = precomputeCoeffs(inRows, outRows);
  return resampleVertical(horz, outCols, outRows, yCoeffs);
}

/** Convenience: widen Uint16Array to Int32Array (matches the PIL `.astype(int32)` step). */
export function uint16ToInt32(src: Uint16Array): Int32Array {
  const out = new Int32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = src[i];
  return out;
}
