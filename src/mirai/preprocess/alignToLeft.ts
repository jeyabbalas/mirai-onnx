/**
 * Port of `Align_To_Left` (upstream `onconet/transformers/image.py:314`).
 *
 * Upstream builds two 1-bit masks (each 1 on the opposite 1/4 of the image) and
 * uses PIL `Image.paste(black, mask)` to black out that 1/4 before summing. The
 * variable naming in the upstream code is inverted — `left_sum` is actually the
 * sum of the right 3/4 and vice versa. The net rule is: **flip horizontally iff
 * the sum over the left 3/4 of the image exceeds the sum over the right 3/4**.
 * Once flipped, the breast tissue will occupy the right side (and the chest wall
 * the left). The transform runs *after* resize, on the (2048, 1664) int32 grid.
 *
 * Uses plain JS Number accumulation (fp64) for the sums, matching the upstream
 * `np.float64` accumulator.
 */

export interface AlignResult {
  pixels: Int32Array;
  flipped: boolean;
  sumLeft3q: number;
  sumRight3q: number;
}

export function alignToLeft(pixels: Int32Array, rows: number, cols: number): AlignResult {
  if (pixels.length !== rows * cols) {
    throw new Error(`alignToLeft: length ${pixels.length} != ${rows}*${cols}`);
  }
  const qx = Math.trunc((cols * 3) / 4); // upstream: size[0] * 3 // 4 — left edge of the right 1/4.

  // sum of left 3/4  = sum of cols [0, qx)
  // sum of right 3/4 = sum of cols [cols - qx, cols)
  // (Upstream uses "3/4" via the INVERSE of the 1/4-mask; cols - qx = cols//4.)
  const leftEnd = qx;
  const rightStart = cols - qx;

  let sumLeft3q = 0;
  let sumRight3q = 0;
  for (let y = 0; y < rows; y++) {
    const row = y * cols;
    let sl = 0;
    let sr = 0;
    for (let x = 0; x < leftEnd; x++) sl += pixels[row + x];
    for (let x = rightStart; x < cols; x++) sr += pixels[row + x];
    sumLeft3q += sl;
    sumRight3q += sr;
  }

  const flipped = sumLeft3q > sumRight3q;
  if (!flipped) {
    return { pixels, flipped, sumLeft3q, sumRight3q };
  }

  const out = new Int32Array(pixels.length);
  for (let y = 0; y < rows; y++) {
    const row = y * cols;
    for (let x = 0; x < cols; x++) out[row + x] = pixels[row + (cols - 1 - x)];
  }
  return { pixels: out, flipped, sumLeft3q, sumRight3q };
}
