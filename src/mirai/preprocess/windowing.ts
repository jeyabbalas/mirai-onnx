export type VoiType = "LINEAR" | "SIGMOID";

export function applyWindowing(image: Float64Array, center: number, width: number, voiType: VoiType): void {
  const yMax = 65535;
  if (voiType === "LINEAR") {
    const c = center - 0.5;
    const w = width - 1.0;
    const low = c - w / 2;
    const high = c + w / 2;
    for (let i = 0; i < image.length; i++) {
      const p = image[i];
      if (p <= low) image[i] = 0;
      else if (p > high) image[i] = yMax;
      else image[i] = ((p - c) / w + 0.5) * yMax;
    }
  } else {
    for (let i = 0; i < image.length; i++) {
      image[i] = yMax / (1 + Math.exp((-4 * (image[i] - center)) / width));
    }
  }
}

export function castFloat64ToUint16(image: Float64Array): Uint16Array {
  const out = new Uint16Array(image.length);
  for (let i = 0; i < image.length; i++) {
    out[i] = Math.trunc(image[i]);
  }
  return out;
}
