/**
 * Normalization + channel replication.
 *
 * Upstream pipeline (`onconet/transformers/tensor.py`):
 *   1. `torchvision.transforms.ToTensor()` on a PIL mode-'I' int32 image produces
 *      a float32 tensor of shape (1, H, W). Note: for mode 'I' there is NO
 *      division by 255 (that only happens for 'L' and 'RGB'); ToTensor just
 *      casts int32 → float32.
 *   2. `Force_Num_Chan_Tensor_2d` → `.expand(3, H, W)` — broadcast-repeat the
 *      single channel to 3. In torch this is a zero-copy view; in our tensor
 *      layout we materialise 3 independent channels, which is what
 *      `torch.save` serializes per `preproc_tensor/*.npy`.
 *   3. `Normalize_Tensor_2d` → `(x - 7047.99) / 12005.5` per channel.
 *
 * Output: Float32Array of length 3*H*W, CHW contiguous order (channel-major).
 */

export const MIRAI_IMG_MEAN = 7047.99;
export const MIRAI_IMG_STD = 12005.5;

export function normalizeAndExpand(
  pixels: Int32Array,
  rows: number,
  cols: number,
  mean: number = MIRAI_IMG_MEAN,
  std: number = MIRAI_IMG_STD,
): Float32Array {
  const planeSize = rows * cols;
  if (pixels.length !== planeSize) {
    throw new Error(`normalizeAndExpand: length ${pixels.length} != ${rows}*${cols}`);
  }
  const out = new Float32Array(3 * planeSize);
  const invStd = 1 / std;
  // Compute normalized values once, then copy to channels 1 and 2.
  // (Three sequential passes stream through contiguous memory; one normalize
  // + two memcpys is the cache-friendliest form.)
  for (let i = 0; i < planeSize; i++) {
    out[i] = (pixels[i] - mean) * invStd;
  }
  out.copyWithin(planeSize, 0, planeSize);
  out.copyWithin(2 * planeSize, 0, planeSize);
  return out;
}
