export type View = 0 | 1;
export type Side = 0 | 1;

export interface PreprocessResult {
  pixels: Float32Array;
  view: View;
  side: Side;
  flipped: boolean;
}
