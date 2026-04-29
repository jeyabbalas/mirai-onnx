// Lightweight DICOM thumbnail renderer for the clinician-facing demo.
// Decodes a DICOM via mirai-onnx-web's existing decodeDicom (same uint16
// pixel buffer the model sees, post-VOI-LUT for GE / post-windowing
// otherwise), histogram-windows it to 8-bit grayscale, and box-averages
// down to a small canvas. No new deps — DICOM parsing is reused from the
// pipeline; rendering is the native Canvas 2D API.

import { decodeDicom, type DecodedDicom } from "mirai-onnx-web";

export interface DicomThumbnail {
  canvas: HTMLCanvasElement;
  view: DecodedDicom["viewStr"];
  side: DecodedDicom["sideStr"];
  rows: number;
  cols: number;
}

export interface ThumbnailOptions {
  maxWidth?: number;
  maxHeight?: number;
  /** Lower percentile cutoff (0..1). Default 0.01. */
  loPct?: number;
  /** Upper percentile cutoff (0..1). Default 0.99. */
  hiPct?: number;
}

const DEFAULT_MAX_W = 200;
const DEFAULT_MAX_H = 250;

export function renderDicomThumbnail(
  buffer: ArrayBufferLike | Uint8Array,
  opts: ThumbnailOptions = {},
): DicomThumbnail {
  const decoded = decodeDicom(buffer);
  const { pixels, rows, cols, viewStr, sideStr } = decoded;

  const maxW = opts.maxWidth ?? DEFAULT_MAX_W;
  const maxH = opts.maxHeight ?? DEFAULT_MAX_H;
  const loPct = opts.loPct ?? 0.01;
  const hiPct = opts.hiPct ?? 0.99;

  const [pmin, pmax] = percentileBounds(pixels, loPct, hiPct);
  const range = Math.max(1, pmax - pmin);

  // Aspect-fit into (maxW, maxH).
  const srcAspect = cols / rows;
  let outW = maxW;
  let outH = Math.round(maxW / srcAspect);
  if (outH > maxH) {
    outH = maxH;
    outW = Math.round(maxH * srcAspect);
  }
  outW = Math.max(1, outW);
  outH = Math.max(1, outH);

  const canvas = document.createElement("canvas");
  canvas.width = outW;
  canvas.height = outH;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("renderDicomThumbnail: 2D context unavailable");
  const imageData = ctx.createImageData(outW, outH);

  // Box-averaged downsample. For each output pixel (x, y) compute the average
  // of the source rectangle [sx0, sx1) × [sy0, sy1), then map [pmin, pmax] →
  // [0, 255] with clamping. Works whether outW/outH are smaller than rows/cols
  // (averaging) or equal (1×1 box = pass-through).
  for (let y = 0; y < outH; y++) {
    const sy0 = Math.floor((y * rows) / outH);
    const sy1 = Math.max(sy0 + 1, Math.floor(((y + 1) * rows) / outH));
    for (let x = 0; x < outW; x++) {
      const sx0 = Math.floor((x * cols) / outW);
      const sx1 = Math.max(sx0 + 1, Math.floor(((x + 1) * cols) / outW));
      let sum = 0;
      let count = 0;
      for (let sy = sy0; sy < sy1; sy++) {
        const rowOffset = sy * cols;
        for (let sx = sx0; sx < sx1; sx++) {
          sum += pixels[rowOffset + sx];
          count++;
        }
      }
      const avg = sum / count;
      let t = Math.round(((avg - pmin) * 255) / range);
      if (t < 0) t = 0;
      else if (t > 255) t = 255;
      const o = (y * outW + x) * 4;
      imageData.data[o] = t;
      imageData.data[o + 1] = t;
      imageData.data[o + 2] = t;
      imageData.data[o + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
  return { canvas, view: viewStr, side: sideStr, rows, cols };
}

function percentileBounds(
  pixels: Uint16Array,
  loPct: number,
  hiPct: number,
): [number, number] {
  const histogram = new Uint32Array(65536);
  for (let i = 0; i < pixels.length; i++) histogram[pixels[i]]++;
  const total = pixels.length;
  const loCount = Math.floor(total * loPct);
  const hiCount = Math.floor(total * hiPct);
  let cum = 0;
  let pmin = 0;
  let pmax = 65535;
  let foundMin = false;
  for (let i = 0; i < 65536; i++) {
    cum += histogram[i];
    if (!foundMin && cum >= loCount) {
      pmin = i;
      foundMin = true;
    }
    if (cum >= hiCount) {
      pmax = i;
      break;
    }
  }
  if (pmax <= pmin) pmax = pmin + 1;
  return [pmin, pmax];
}
