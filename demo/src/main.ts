// Phase 9 browser demo entrypoint.
//
// Wires the mirai-onnx-web barrel (preprocessDicom, vectorizeRiskFactors,
// calibrator, runMirai) to onnxruntime-web via createWebSessions, renders
// predictions, slot order, the 612-dim embedding as a heatmap, and per-stage
// timings. A "Benchmark" button runs 10 iterations and summarizes.
//
// The ORT-web WASM assets are copied into /ort/ by demo/scripts/link-models.mjs;
// we point env.wasm.wasmPaths at that path so threaded WASM works under the
// dev server's COOP/COEP headers. WebGPU ignores wasmPaths.

import * as ort from "onnxruntime-web";
import {
  runMirai,
  loadCalibrator,
  createWebSessionsFromBytes,
  MIRAI_MODEL_VERSION,
  FEATURE_NAMES,
  type Calibrator,
  type MiraiSessions,
  type MiraiStage,
  type MiraiRiskFactors,
  type DensityCode,
  type RaceCode,
  type RelativeCode,
  type Relative,
  type HrtType,
} from "mirai-onnx-web";
import { drawHeatmap, ROWS, COLS } from "./render.js";

// Image-encoder dims occupy `embedding[0..511]`; risk-factor dims occupy
// `embedding[512..611]` and align with FEATURE_NAMES (canonical training-time
// feature names from the upstream Python `get_feature_names()`). This array
// drives both the heatmap tooltip and the labeled JSON download — the demo
// never re-types these strings, so a label can never silently drift from
// what the model trained against.
const IMAGE_EMBEDDING_DIM = 512;
const RISK_FACTOR_EMBEDDING_DIM = 100;
const HEATMAP_DIM = IMAGE_EMBEDDING_DIM + RISK_FACTOR_EMBEDDING_DIM;
if (FEATURE_NAMES.length !== RISK_FACTOR_EMBEDDING_DIM) {
  throw new Error(
    `mirai-onnx-web FEATURE_NAMES length=${FEATURE_NAMES.length}, expected ${RISK_FACTOR_EMBEDDING_DIM}`,
  );
}
if (HEATMAP_DIM !== ROWS * COLS) {
  throw new Error(`heatmap dim mismatch: ${HEATMAP_DIM} != ${ROWS}*${COLS}`);
}
const LABELS: readonly string[] = Object.freeze([
  ...Array.from({ length: IMAGE_EMBEDDING_DIM }, (_, i) => `img_emb_${i + 1}`),
  ...FEATURE_NAMES,
]);

// Vite injects `import.meta.env.BASE_URL` with a trailing slash: "/" in dev,
// "/mirai-onnx/" under the Pages workflow (vite.config.ts reads DEPLOY_BASE_URL).
const BASE = import.meta.env.BASE_URL;

// Production-only: point ORT at the standalone `ort-wasm-simd-threaded*.{mjs,wasm}`
// files that link-models.mjs copies into public/ort/ (Vite propagates them to
// dist/ort/).
//
// Why this is necessary in prod: `import * as ort` resolves to
// `ort.bundle.min.mjs`, and Vite's production build inlines that into our
// main entry. ORT's Emscripten pthread spawner reads `import.meta.url` at
// runtime and uses it as the Worker URL — so when the ORT code is bundled
// into our main chunk, `import.meta.url` becomes `/…/assets/index-<hash>.js`.
// The browser then tries to run the entire app bundle as a Web Worker, which
// crashes immediately in Vite's modulepreload polyfill (no `document` in
// worker scope — Uncaught ReferenceError: document is not defined).
//
// Pointing wasmPaths at the standalone .mjs fixes this: inside those raw
// Emscripten modules `import.meta.url` is the .mjs itself, which is what the
// em-pthread Worker needs. WebGPU is unaffected (it doesn't use pthread
// workers for inference).
//
// Why we SKIP this in dev: Vite's dev server serves ORT as a real ES module
// from /node_modules/onnxruntime-web/dist/ort.bundle.min.mjs — already at the
// correct URL for `import.meta.url`. Setting wasmPaths to "/ort/..." in dev
// fails because Vite refuses module imports of files under public/.
if (import.meta.env.PROD) {
  ort.env.wasm.wasmPaths = `${BASE}ort/`;
}
const MODEL_URLS = {
  encoder: `${BASE}models/image_encoder.onnx`,
  risk: `${BASE}models/risk_model.onnx`,
} as const;
const CALIBRATOR_URL = `${BASE}models/calibrator.json`;
const SAMPLE_FILES = ["ccl1.dcm", "ccr1.dcm", "mlol2.dcm", "mlor2.dcm"] as const;

const STAGES: MiraiStage[] = ["preprocess", "encoder", "risk", "calibrate", "total"];

interface StageAccumulator {
  samples: number[];
}

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing DOM node #${id}`);
  return el as T;
};

const fileInput = $<HTMLInputElement>("file-input");
const loadDemoBtn = $<HTMLButtonElement>("load-demo");
const fileListEl = $<HTMLDivElement>("file-list");
const dropzone = $<HTMLDivElement>("dropzone");
const runBtn = $<HTMLButtonElement>("run-btn");
const benchBtn = $<HTMLButtonElement>("bench-btn");
const downloadBtn = $<HTMLButtonElement>("download-btn");
const statusEl = $<HTMLDivElement>("status");
const epSelect = $<HTMLSelectElement>("ep-select");
const epTag = $<HTMLSpanElement>("ep-tag");
const threadsTag = $<HTMLSpanElement>("threads-tag");
const predTableBody = $<HTMLTableSectionElement>(
  "pred-table",
).querySelector("tbody") as HTMLTableSectionElement;
const slotsTableBody = $<HTMLTableSectionElement>(
  "slots-table",
).querySelector("tbody") as HTMLTableSectionElement;
const timingsTableBody = $<HTMLTableSectionElement>(
  "timings-table",
).querySelector("tbody") as HTMLTableSectionElement;
const heatmap = $<HTMLCanvasElement>("heatmap");
const heatmapTooltip = $<HTMLDivElement>("heatmap-tooltip");
const rfDensity = $<HTMLSelectElement>("rf-density");
const rfFamhist = $<HTMLInputElement>("rf-famhist");
const rfBiopsyBenign = $<HTMLInputElement>("rf-biopsy-benign");
const rfBiopsyLcis = $<HTMLInputElement>("rf-biopsy-lcis");
const rfBiopsyAtypical = $<HTMLInputElement>("rf-biopsy-atypical");
const rfAge = $<HTMLInputElement>("rf-age");
const rfMenarcheAge = $<HTMLInputElement>("rf-menarche-age");
const rfMenopauseAge = $<HTMLInputElement>("rf-menopause-age");
const rfFirstPregnancyAge = $<HTMLInputElement>("rf-first-pregnancy-age");
const rfPriorHist = $<HTMLInputElement>("rf-prior-hist");
const rfRace = $<HTMLSelectElement>("rf-race");
const rfNumBirths = $<HTMLInputElement>("rf-num-births");
const rfWeight = $<HTMLInputElement>("rf-weight");
const rfHeight = $<HTMLInputElement>("rf-height");
const rfOvarianCancer = $<HTMLInputElement>("rf-ovarian-cancer");
const rfOvarianCancerAge = $<HTMLInputElement>("rf-ovarian-cancer-age");
const rfAshkenazi = $<HTMLInputElement>("rf-ashkenazi");
const rfBrca = $<HTMLSelectElement>("rf-brca");
const rfMomBc = $<HTMLInputElement>("rf-mom-bc");
const rfMAuntBc = $<HTMLInputElement>("rf-m-aunt-bc");
const rfPAuntBc = $<HTMLInputElement>("rf-p-aunt-bc");
const rfMGrandmotherBc = $<HTMLInputElement>("rf-m-grandmother-bc");
const rfPGrandmotherBc = $<HTMLInputElement>("rf-p-grandmother-bc");
const rfSisterBc = $<HTMLInputElement>("rf-sister-bc");
const rfMomOc = $<HTMLInputElement>("rf-mom-oc");
const rfMAuntOc = $<HTMLInputElement>("rf-m-aunt-oc");
const rfPAuntOc = $<HTMLInputElement>("rf-p-aunt-oc");
const rfMGrandmotherOc = $<HTMLInputElement>("rf-m-grandmother-oc");
const rfPGrandmotherOc = $<HTMLInputElement>("rf-p-grandmother-oc");
const rfSisterOc = $<HTMLInputElement>("rf-sister-oc");
const rfHrtType = $<HTMLSelectElement>("rf-hrt-type");
const rfHrtDuration = $<HTMLInputElement>("rf-hrt-duration");
const rfHrtLastAge = $<HTMLInputElement>("rf-hrt-last-age");
const footerVersion = $<HTMLSpanElement>("footer-version");
footerVersion.textContent = `modelVersion: ${MIRAI_MODEL_VERSION}`;

let selectedFiles: File[] = [];
let sessions: MiraiSessions | null = null;
let sessionsEP: "webgpu" | "wasm" | null = null;
let calibrator: Calibrator | null = null;
let lastEmbedding: Float32Array | null = null;
const accumulators: Record<MiraiStage, StageAccumulator> = {
  preprocess: { samples: [] },
  encoder: { samples: [] },
  risk: { samples: [] },
  calibrate: { samples: [] },
  total: { samples: [] },
};

function setStatus(msg: string, isError = false): void {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", isError);
}

function renderFileList(): void {
  if (selectedFiles.length === 0) {
    fileListEl.textContent = "";
  } else {
    fileListEl.textContent = selectedFiles.map((f) => `• ${f.name} (${(f.size / 1e6).toFixed(1)} MB)`).join("\n");
  }
  // Gate only on file count. Sessions + calibrator are lazily re-initialized
  // inside onRun/onBench (ensureSessions/ensureCalibrator), so requiring them
  // here would strand the user whenever the eager init below is slow or
  // errors out.
  runBtn.disabled = selectedFiles.length !== 4;
  benchBtn.disabled = runBtn.disabled;
}

async function ensureSessions(): Promise<void> {
  const preferWebGPU = epSelect.value !== "wasm";
  const needNew = !sessions || (preferWebGPU ? sessionsEP !== "webgpu" : sessionsEP !== "wasm");
  if (!needNew) return;

  setStatus("Loading ONNX models…");
  const [encoderResp, riskResp] = await Promise.all([fetch(MODEL_URLS.encoder), fetch(MODEL_URLS.risk)]);
  if (!encoderResp.ok || !riskResp.ok) {
    throw new Error(`Failed to fetch models (${encoderResp.status}/${riskResp.status})`);
  }
  const [encoderBytes, riskBytes] = await Promise.all([encoderResp.arrayBuffer(), riskResp.arrayBuffer()]);
  setStatus(`Creating ${preferWebGPU ? "WebGPU" : "WASM"} sessions…`);
  sessions = await createWebSessionsFromBytes(
    { encoder: encoderBytes, risk: riskBytes },
    { preferWebGPU },
  );
  // Detect which EP actually bound. onnxruntime-web doesn't expose active EP on
  // the session directly, so we rely on preferWebGPU + the presence of the
  // WebGPU runtime in the global env to infer it. Good enough for the demo.
  sessionsEP = preferWebGPU && isWebGPUAvailable() ? "webgpu" : "wasm";
  epTag.textContent = `EP: ${sessionsEP}`;
  threadsTag.textContent = `threads: ${ort.env.wasm.numThreads ?? "?"}`;
  setStatus(`Sessions ready (${sessionsEP}).`);
}

function isWebGPUAvailable(): boolean {
  const anyNav = navigator as Navigator & { gpu?: unknown };
  return !!anyNav.gpu;
}

async function ensureCalibrator(): Promise<void> {
  if (calibrator) return;
  const resp = await fetch(CALIBRATOR_URL);
  if (!resp.ok) throw new Error(`Failed to fetch calibrator: ${resp.status}`);
  const json = await resp.json();
  calibrator = loadCalibrator(json);
}

function readInt(el: HTMLInputElement | HTMLSelectElement): number | undefined {
  const v = Number.parseInt(el.value, 10);
  return Number.isFinite(v) ? v : undefined;
}

function buildRiskFactors(): MiraiRiskFactors {
  const rf: MiraiRiskFactors = {};

  // 1. density
  const density = readInt(rfDensity);
  if (density !== undefined) rf.density = density as DensityCode;

  // 3-5. biopsies
  if (rfBiopsyBenign.checked) rf.biopsyHyperplasia = true;
  if (rfBiopsyLcis.checked) rf.biopsyLCIS = true;
  if (rfBiopsyAtypical.checked) rf.biopsyAtypicalHyperplasia = true;

  // 6-9. ages
  const age = readInt(rfAge);
  if (age !== undefined) rf.age = age;
  const menarcheAge = readInt(rfMenarcheAge);
  if (menarcheAge !== undefined) rf.menarcheAge = menarcheAge;
  const menopauseAge = readInt(rfMenopauseAge);
  if (menopauseAge !== undefined) rf.menopauseAge = menopauseAge;
  const firstPregnancyAge = readInt(rfFirstPregnancyAge);
  if (firstPregnancyAge !== undefined) rf.firstPregnancyAge = firstPregnancyAge;

  // 10. prior hist
  if (rfPriorHist.checked) rf.priorHist = true;

  // 11. race
  const race = readInt(rfRace);
  if (race !== undefined) rf.race = race as RaceCode;

  // 12. parous (number of births drives `parous` per factors.ts:101)
  const numBirths = readInt(rfNumBirths);
  if (numBirths !== undefined) rf.numBirths = numBirths;

  // 14-15. weight, height
  const weight = readInt(rfWeight);
  if (weight !== undefined) rf.weight = weight;
  const height = readInt(rfHeight);
  if (height !== undefined) rf.height = height;

  // 16-17. ovarian cancer
  if (rfOvarianCancer.checked) rf.ovarianCancer = true;
  const ocAge = readInt(rfOvarianCancerAge);
  if (ocAge !== undefined) rf.ovarianCancerAge = ocAge;

  // 18. ashkenazi
  if (rfAshkenazi.checked) rf.ashkenazi = true;

  // 19. brca: drives the 4-class one-hot via brca1/brca2 booleans.
  // ""→don't set (idx 0, never/unknown), "negative"→brca1=false (idx 1),
  // "brca1"→brca1=true (idx 2), "brca2"→brca2=true (idx 3). See
  // src/mirai/riskFactors/transformers/brca.ts.
  switch (rfBrca.value) {
    case "negative":
      rf.brca1 = false;
      break;
    case "brca1":
      rf.brca1 = true;
      break;
    case "brca2":
      rf.brca2 = true;
      break;
  }

  // 20-31. per-relative breast/ovarian cancer history. Each relative code
  // gets one Relative object combining the bc and oc checkboxes. If any are
  // set, those drive `binary_family_history` and the per-relative keys. If
  // none are set but the row-2 shortcut is checked, fall back to the legacy
  // `{ M: [{}] }` sentinel that flips just `binary_family_history`.
  const relSpec: { code: RelativeCode; bc: HTMLInputElement; oc: HTMLInputElement }[] = [
    { code: "M", bc: rfMomBc, oc: rfMomOc },
    { code: "MA", bc: rfMAuntBc, oc: rfMAuntOc },
    { code: "PA", bc: rfPAuntBc, oc: rfPAuntOc },
    { code: "MG", bc: rfMGrandmotherBc, oc: rfMGrandmotherOc },
    { code: "PG", bc: rfPGrandmotherBc, oc: rfPGrandmotherOc },
    { code: "S", bc: rfSisterBc, oc: rfSisterOc },
  ];
  const relatives: Partial<Record<RelativeCode, Relative[]>> = {};
  let anyPerRelative = false;
  for (const { code, bc, oc } of relSpec) {
    if (bc.checked || oc.checked) {
      anyPerRelative = true;
      const r: Relative = {};
      if (bc.checked) r.breastCancer = true;
      if (oc.checked) r.ovarianCancer = true;
      relatives[code] = [r];
    }
  }
  if (anyPerRelative) {
    rf.relatives = relatives;
  } else if (rfFamhist.checked) {
    rf.relatives = { M: [{}] };
  }

  // 32-34. HRT
  const hrtType = rfHrtType.value;
  if (hrtType === "combined" || hrtType === "estrogen" || hrtType === "unknown") {
    rf.hrt = { type: hrtType as HrtType };
    const dur = readInt(rfHrtDuration);
    if (dur !== undefined) rf.hrt.duration = dur;
    const lastAge = readInt(rfHrtLastAge);
    if (lastAge !== undefined) rf.hrt.lastAge = lastAge;
  }

  return rf;
}

function clearAccumulators(): void {
  for (const stage of STAGES) accumulators[stage].samples = [];
}

function pushTiming(stage: MiraiStage, ms: number): void {
  accumulators[stage].samples.push(ms);
}

function renderTimings(lastRun: Partial<Record<MiraiStage, number>>): void {
  timingsTableBody.innerHTML = "";
  for (const stage of STAGES) {
    const s = accumulators[stage].samples;
    const sorted = [...s].sort((a, b) => a - b);
    const p = (q: number) => (sorted.length > 0 ? sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] : NaN);
    const mean = s.length > 0 ? s.reduce((a, b) => a + b, 0) / s.length : NaN;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${stage}</td><td>${lastRun[stage]?.toFixed(1) ?? "—"}</td><td>${p(0.5).toFixed(1)}</td><td>${p(0.95).toFixed(1)}</td><td>${mean.toFixed(1)}</td><td>${s.length}</td>`;
    timingsTableBody.appendChild(tr);
  }
}

function renderResult(
  result: Awaited<ReturnType<typeof runMirai>>,
  lastRun: Partial<Record<MiraiStage, number>>,
): void {
  predTableBody.innerHTML = "";
  const years = ["year1", "year2", "year3", "year4", "year5"] as const;
  for (let i = 0; i < 5; i++) {
    const rounded = result.predictions[years[i]];
    const calib = result.calibrated[i];
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i + 1}</td><td>${rounded.toFixed(4)}</td><td>${calib.toFixed(8)}</td>`;
    predTableBody.appendChild(tr);
  }

  slotsTableBody.innerHTML = "";
  for (let i = 0; i < result.slotOrder.length; i++) {
    const s = result.slotOrder[i];
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i}</td><td>${s.view === 0 ? "CC" : "MLO"}</td><td>${s.side === 0 ? "R" : "L"}</td><td>${s.flipped}</td>`;
    slotsTableBody.appendChild(tr);
  }

  drawHeatmap(heatmap, result.embedding);
  lastEmbedding = result.embedding;
  downloadBtn.disabled = false;

  renderTimings(lastRun);
}

async function runOnce(): Promise<Awaited<ReturnType<typeof runMirai>>> {
  if (!sessions || !calibrator) throw new Error("sessions/calibrator not initialized");
  if (selectedFiles.length !== 4) throw new Error("need exactly 4 DICOMs");
  const buffers = await Promise.all(selectedFiles.map((f) => f.arrayBuffer()));
  const lastRun: Partial<Record<MiraiStage, number>> = {};
  const rf = buildRiskFactors();
  const result = await runMirai(buffers, sessions, calibrator, rf, {
    onStage: (stage, ms) => {
      pushTiming(stage, ms);
      lastRun[stage] = ms;
    },
  });
  renderResult(result, lastRun);
  return result;
}

async function onRun(): Promise<void> {
  runBtn.disabled = true;
  benchBtn.disabled = true;
  try {
    await ensureSessions();
    await ensureCalibrator();
    clearAccumulators();
    setStatus("Running…");
    const result = await runOnce();
    setStatus(
      `Done. Predictions: ${JSON.stringify(result.predictions)}  ·  EP: ${sessionsEP}`,
    );
  } catch (err) {
    console.error(err);
    setStatus(`ERROR: ${(err as Error).message}`, true);
  } finally {
    renderFileList();
  }
}

async function onBench(): Promise<void> {
  const ITERS = 10;
  runBtn.disabled = true;
  benchBtn.disabled = true;
  try {
    await ensureSessions();
    await ensureCalibrator();
    clearAccumulators();
    // Warmup.
    setStatus("Warmup…");
    await runOnce();
    clearAccumulators();
    for (let i = 0; i < ITERS; i++) {
      setStatus(`Benchmark iter ${i + 1}/${ITERS}…`);
      await runOnce();
    }
    setStatus(`Benchmark complete (${ITERS} iters, EP: ${sessionsEP}).`);
  } catch (err) {
    console.error(err);
    setStatus(`ERROR: ${(err as Error).message}`, true);
  } finally {
    renderFileList();
  }
}

async function onLoadDemo(): Promise<void> {
  setStatus("Fetching bundled demo DICOMs…");
  try {
    const files: File[] = [];
    for (const name of SAMPLE_FILES) {
      const resp = await fetch(`${BASE}sample/${name}`);
      if (!resp.ok) throw new Error(`/sample/${name}: ${resp.status}`);
      const blob = await resp.blob();
      files.push(new File([blob], name, { type: "application/dicom" }));
    }
    selectedFiles = files;
    renderFileList();
    setStatus(`Loaded ${files.length} demo DICOMs. Click "Run prediction".`);
  } catch (err) {
    setStatus(`ERROR: ${(err as Error).message}`, true);
  }
}

function onFileInput(): void {
  const files = Array.from(fileInput.files ?? []);
  selectedFiles = files;
  renderFileList();
  if (files.length === 4) {
    setStatus("4 DICOMs selected. Click 'Run prediction'.");
  } else if (files.length > 0) {
    setStatus(`Selected ${files.length} of 4 DICOMs. Need exactly 4.`, true);
  }
}

function onDownload(): void {
  if (!lastEmbedding) return;
  const imageEmbedding = Array.from(lastEmbedding.subarray(0, IMAGE_EMBEDDING_DIM));
  const rfSlice = lastEmbedding.subarray(IMAGE_EMBEDDING_DIM);
  const riskFactorEmbedding: Record<string, number> = {};
  for (let i = 0; i < FEATURE_NAMES.length; i++) {
    riskFactorEmbedding[FEATURE_NAMES[i]] = rfSlice[i];
  }
  const payload = {
    modelVersion: MIRAI_MODEL_VERSION,
    dtype: "float32",
    postReLU: true,
    imageEmbedding,
    riskFactorEmbedding,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "mirai-embedding.json";
  a.click();
  URL.revokeObjectURL(url);
}

function onHeatmapMove(e: MouseEvent): void {
  if (!lastEmbedding) {
    heatmapTooltip.hidden = true;
    return;
  }
  const rect = heatmap.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  // Map CSS-pixel coords to canvas-pixel coords (in case the canvas is scaled
  // by CSS), then to (row, col) using the same `Math.floor(canvas.width/COLS)`
  // cell-size that render.ts uses to fill cells. 540/18=30 and 476/34=14 are
  // exact, so the integer floor lines up with cell boundaries on this canvas.
  const sx = heatmap.width / rect.width;
  const sy = heatmap.height / rect.height;
  const cellW = Math.floor(heatmap.width / COLS);
  const cellH = Math.floor(heatmap.height / ROWS);
  const col = Math.floor((x * sx) / cellW);
  const row = Math.floor((y * sy) / cellH);
  if (row < 0 || row >= ROWS || col < 0 || col >= COLS) {
    heatmapTooltip.hidden = true;
    return;
  }
  const idx = row * COLS + col;
  heatmapTooltip.textContent = `${LABELS[idx]}: ${lastEmbedding[idx].toFixed(4)}`;
  heatmapTooltip.style.left = `${x + 12}px`;
  heatmapTooltip.style.top = `${y + 12}px`;
  heatmapTooltip.hidden = false;
}

function onHeatmapLeave(): void {
  heatmapTooltip.hidden = true;
}

function wireDropzone(): void {
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("over");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("over"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("over");
    const files = Array.from(e.dataTransfer?.files ?? []).filter((f) => /\.dcm$/i.test(f.name) || f.type === "application/dicom");
    selectedFiles = files;
    renderFileList();
    if (files.length === 4) setStatus("4 DICOMs dropped. Click 'Run prediction'.");
    else setStatus(`Dropped ${files.length} DICOMs; need exactly 4.`, true);
  });
}

fileInput.addEventListener("change", onFileInput);
loadDemoBtn.addEventListener("click", onLoadDemo);
runBtn.addEventListener("click", onRun);
benchBtn.addEventListener("click", onBench);
downloadBtn.addEventListener("click", onDownload);
heatmap.addEventListener("mousemove", onHeatmapMove);
heatmap.addEventListener("mouseleave", onHeatmapLeave);
epSelect.addEventListener("change", () => {
  sessions = null;
  sessionsEP = null;
  epTag.textContent = "EP: —";
  setStatus("EP changed — sessions will rebuild on next run.");
});
wireDropzone();

threadsTag.textContent = `threads: ${ort.env.wasm.numThreads ?? "?"}`;
setStatus(
  isWebGPUAvailable()
    ? "Preparing ONNX sessions… you can load DICOMs while this runs."
    : "Preparing ONNX sessions (WebGPU not detected; using WASM)… you can load DICOMs while this runs.",
);
// Eagerly create sessions + calibrator so first click is fast. If this fails,
// the Run button is still enabled once 4 files are loaded — onRun will retry
// the init on click and surface any error then.
(async () => {
  try {
    await ensureSessions();
    await ensureCalibrator();
    renderFileList();
    setStatus(`Sessions + calibrator ready. EP: ${sessionsEP}. Load 4 DICOMs to start.`);
  } catch (err) {
    setStatus(
      `Session init failed: ${(err as Error).message}. Load 4 DICOMs and click Run — init will retry.`,
      true,
    );
  }
})();
