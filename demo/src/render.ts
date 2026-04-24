// 34 × 18 = 612 cell heatmap for the post-ReLU pre-hazard embedding. Linear
// grayscale 0 → max (post-ReLU is non-negative). No dependencies.

const ROWS = 34;
const COLS = 18;

export function drawHeatmap(canvas: HTMLCanvasElement, embedding: Float32Array): void {
  if (embedding.length !== ROWS * COLS) {
    throw new Error(`drawHeatmap: expected ${ROWS * COLS} values, got ${embedding.length}`);
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("drawHeatmap: 2D context unavailable");

  const cellW = Math.floor(canvas.width / COLS);
  const cellH = Math.floor(canvas.height / ROWS);

  let max = 0;
  for (let i = 0; i < embedding.length; i++) {
    if (embedding[i] > max) max = embedding[i];
  }
  if (max === 0) max = 1; // all-zero degenerates to a uniform black canvas; keep finite.

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const v = embedding[r * COLS + c];
      const t = Math.min(1, v / max);
      const g = Math.round(255 * (1 - t));
      ctx.fillStyle = `rgb(${g}, ${g}, ${g})`;
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
    }
  }

  // Separator between image (first 512 dims = 28+ rows) and RF block (last 100).
  // 512 / 18 = 28.44, so the boundary falls mid-row at (r=28, c=8). Draw a hint
  // line across row 28 so viewers can tell roughly where the RF block starts.
  ctx.strokeStyle = "#ff7a00";
  ctx.lineWidth = 1;
  const splitRow = Math.floor(512 / COLS);
  const splitCol = 512 - splitRow * COLS;
  ctx.beginPath();
  ctx.moveTo(0, splitRow * cellH);
  ctx.lineTo(splitCol * cellW, splitRow * cellH);
  ctx.moveTo(splitCol * cellW, splitRow * cellH);
  ctx.lineTo(splitCol * cellW, (splitRow + 1) * cellH);
  ctx.moveTo(splitCol * cellW, (splitRow + 1) * cellH);
  ctx.lineTo(canvas.width, (splitRow + 1) * cellH);
  ctx.stroke();
}
