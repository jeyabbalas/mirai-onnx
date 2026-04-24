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
  type Calibrator,
  type MiraiSessions,
  type MiraiStage,
  type MiraiRiskFactors,
  type DensityCode,
} from "mirai-onnx-web";
import { drawHeatmap } from "./render.js";

// Do NOT set ort.env.wasm.wasmPaths. In dev, Vite serves the ORT assets
// straight from /node_modules/onnxruntime-web/dist/; in prod, Vite sees
// `new URL("./ort-wasm-simd-threaded.jsep.wasm", import.meta.url)` inside
// the ORT bundle and copies the .wasm into dist/assets/ with the correct
// hashed URL. Either way, ORT's default "resolve relative to the bundle's
// import.meta.url" path is what works — setting wasmPaths to anything else
// broke WebGPU init with "both async and sync fetching of the wasm failed".

const MODEL_URLS = {
  encoder: "/models/image_encoder.onnx",
  risk: "/models/risk_model.onnx",
} as const;
const CALIBRATOR_URL = "/models/calibrator.json";
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
const rfAge = $<HTMLInputElement>("rf-age");
const rfDensity = $<HTMLSelectElement>("rf-density");
const rfFamhist = $<HTMLInputElement>("rf-famhist");
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

function buildRiskFactors(): MiraiRiskFactors {
  const rf: MiraiRiskFactors = {};
  const age = Number.parseInt(rfAge.value, 10);
  if (Number.isFinite(age)) rf.age = age;
  const density = Number.parseInt(rfDensity.value, 10);
  if (Number.isFinite(density)) rf.density = density as DensityCode;
  if (rfFamhist.checked) {
    // `binary_family_history` is derived from the `relatives` dict: any
    // relative-code list with at least one entry flips the bit. Use mother as
    // the sentinel when the UI just asserts "someone in the family."
    rf.relatives = { M: [{}] };
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
      const resp = await fetch(`/sample/${name}`);
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
  const payload = {
    modelVersion: MIRAI_MODEL_VERSION,
    shape: [lastEmbedding.length],
    dtype: "float32",
    postReLU: true,
    data: Array.from(lastEmbedding),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "mirai-embedding.json";
  a.click();
  URL.revokeObjectURL(url);
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
