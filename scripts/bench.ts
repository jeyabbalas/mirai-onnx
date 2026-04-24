// Phase 9: Node benchmark. Runs the full pipeline N times on the demo DICOMs,
// measures per-stage wall-clock via runMirai's onStage callback, and reports
// p50/p95/mean/stddev per stage. Results are written to
// artifacts/phase_9/bench_node.json (gitignored).
//
// Regression gate: if artifacts/phase_9/bench_node.baseline.json exists, fail the
// script when any stage's p50 exceeds the baseline by >= REGRESSION_MULTIPLIER.
//
// Usage:
//   npm run bench                     # default 10 iters + 1 warmup
//   npm run bench -- --iters 25       # more iterations
//   npm run bench -- --update-baseline  # overwrite baseline with current run

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  runMirai,
  createNodeSessions,
  type MiraiSessions,
  type MiraiStage,
} from "../src/mirai/index.js";
import { loadCalibratorFromFile } from "../src/mirai/calibrator.node.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");

const MODELS_DIR = path.join(REPO_ROOT, "models");
const DEMO_DIR = path.join(REPO_ROOT, "mirai_demo_data");
const ARTIFACTS_DIR = path.join(REPO_ROOT, "artifacts", "phase_9");

const DEMO_FILES = ["ccl1.dcm", "ccr1.dcm", "mlol2.dcm", "mlor2.dcm"];
const STAGES: MiraiStage[] = ["preprocess", "encoder", "risk", "calibrate", "total"];
const DEFAULT_ITERS = 10;
const REGRESSION_MULTIPLIER = 2.0;

interface StageStats {
  samples: number;
  mean: number;
  stddev: number;
  p50: number;
  p95: number;
  min: number;
  max: number;
}

type StageMap = Record<MiraiStage, number[]>;
type StageStatsMap = Record<MiraiStage, StageStats>;

function parseArgs(argv: string[]): { iters: number; updateBaseline: boolean } {
  let iters = DEFAULT_ITERS;
  let updateBaseline = false;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--iters" && i + 1 < argv.length) {
      iters = Number.parseInt(argv[++i], 10);
      if (!Number.isFinite(iters) || iters < 1) {
        throw new Error("--iters must be a positive integer");
      }
    } else if (argv[i] === "--update-baseline") {
      updateBaseline = true;
    } else {
      throw new Error(`Unknown argument: ${argv[i]}`);
    }
  }
  return { iters, updateBaseline };
}

function summarize(samples: number[]): StageStats {
  const sorted = [...samples].sort((a, b) => a - b);
  const n = sorted.length;
  const mean = sorted.reduce((s, x) => s + x, 0) / n;
  const variance = sorted.reduce((s, x) => s + (x - mean) ** 2, 0) / n;
  const p = (q: number) => sorted[Math.min(n - 1, Math.floor(q * n))];
  return {
    samples: n,
    mean,
    stddev: Math.sqrt(variance),
    p50: p(0.5),
    p95: p(0.95),
    min: sorted[0],
    max: sorted[n - 1],
  };
}

function formatTable(stats: StageStatsMap): string {
  const rows = [
    "| stage      | p50 (ms) | p95 (ms) | mean (ms) | stddev (ms) | min (ms) | max (ms) |",
    "|------------|---------:|---------:|----------:|------------:|---------:|---------:|",
  ];
  for (const stage of STAGES) {
    const s = stats[stage];
    rows.push(
      `| ${stage.padEnd(10)} | ${s.p50.toFixed(1).padStart(8)} | ${s.p95.toFixed(1).padStart(8)} | ` +
        `${s.mean.toFixed(1).padStart(9)} | ${s.stddev.toFixed(1).padStart(11)} | ` +
        `${s.min.toFixed(1).padStart(8)} | ${s.max.toFixed(1).padStart(8)} |`,
    );
  }
  return rows.join("\n");
}

async function runOne(
  sessions: MiraiSessions,
  calibrator: Parameters<typeof runMirai>[2],
  files: ReadonlyArray<Buffer>,
  collect: StageMap,
): Promise<void> {
  await runMirai(files, sessions, calibrator, {}, {
    onStage: (stage, ms) => {
      collect[stage].push(ms);
    },
  });
}

function compareAgainstBaseline(
  current: StageStatsMap,
  baseline: { stages: StageStatsMap } | null,
): { regressed: Array<{ stage: MiraiStage; currentP50: number; baselineP50: number; ratio: number }> } {
  if (!baseline) return { regressed: [] };
  const regressed: Array<{ stage: MiraiStage; currentP50: number; baselineP50: number; ratio: number }> = [];
  for (const stage of STAGES) {
    const cur = current[stage].p50;
    const base = baseline.stages[stage].p50;
    const ratio = cur / base;
    if (ratio >= REGRESSION_MULTIPLIER) {
      regressed.push({ stage, currentP50: cur, baselineP50: base, ratio });
    }
  }
  return { regressed };
}

async function main(): Promise<number> {
  const { iters, updateBaseline } = parseArgs(process.argv.slice(2));

  console.log(`Mirai Node benchmark — ${iters} iters (+ 1 warmup)`);
  console.log(`Host: ${os.platform()}/${os.arch()} · Node ${process.version} · CPU ${os.cpus()[0].model} · ${os.cpus().length} cores`);

  const encoderPath = path.join(MODELS_DIR, "image_encoder.onnx");
  const riskPath = path.join(MODELS_DIR, "risk_model.onnx");
  const calibratorPath = path.join(MODELS_DIR, "calibrator.json");
  for (const p of [encoderPath, riskPath, calibratorPath]) {
    if (!fs.existsSync(p)) {
      console.error(`ERROR: missing ${p}`);
      return 2;
    }
  }
  for (const f of DEMO_FILES) {
    const p = path.join(DEMO_DIR, f);
    if (!fs.existsSync(p)) {
      console.error(`ERROR: missing ${p}`);
      return 2;
    }
  }

  const calibrator = loadCalibratorFromFile(calibratorPath);
  const sessions = await createNodeSessions({ encoder: encoderPath, risk: riskPath });
  const files = DEMO_FILES.map((f) => fs.readFileSync(path.join(DEMO_DIR, f)));

  const empty: StageMap = {
    preprocess: [],
    encoder: [],
    risk: [],
    calibrate: [],
    total: [],
  };

  // Warmup (first run pays ORT jitting + FS caching; exclude from stats).
  const warmup: StageMap = JSON.parse(JSON.stringify(empty));
  await runOne(sessions, calibrator, files, warmup);
  console.log(
    `Warmup: total=${warmup.total[0]?.toFixed(1) ?? "?"}ms encoder=${warmup.encoder[0]?.toFixed(1) ?? "?"}ms`,
  );

  const collected: StageMap = JSON.parse(JSON.stringify(empty));
  for (let i = 0; i < iters; i++) {
    await runOne(sessions, calibrator, files, collected);
    const last = STAGES.reduce(
      (acc, s) => ({ ...acc, [s]: collected[s][collected[s].length - 1] }),
      {} as Record<MiraiStage, number>,
    );
    console.log(
      `  iter ${String(i + 1).padStart(2)}/${iters}: total=${last.total.toFixed(1)}ms ` +
        `preprocess=${last.preprocess.toFixed(1)} encoder=${last.encoder.toFixed(1)} ` +
        `risk=${last.risk.toFixed(1)} calibrate=${last.calibrate.toFixed(2)}`,
    );
  }

  const stats: StageStatsMap = STAGES.reduce(
    (acc, s) => ({ ...acc, [s]: summarize(collected[s]) }),
    {} as StageStatsMap,
  );

  console.log("");
  console.log(formatTable(stats));

  fs.mkdirSync(ARTIFACTS_DIR, { recursive: true });
  const outPath = path.join(ARTIFACTS_DIR, "bench_node.json");
  const baselinePath = path.join(ARTIFACTS_DIR, "bench_node.baseline.json");
  const payload = {
    phase: 9,
    provider: "onnxruntime-node (cpu)",
    iters,
    samples: collected,
    stages: stats,
    env: {
      platform: os.platform(),
      arch: os.arch(),
      node: process.version,
      cpu: os.cpus()[0].model,
      cores: os.cpus().length,
      totalMemMB: Math.round(os.totalmem() / 1024 / 1024),
      host: os.hostname(),
      timestamp: new Date().toISOString(),
    },
  };
  fs.writeFileSync(outPath, JSON.stringify(payload, null, 2));
  console.log(`\nWrote ${path.relative(REPO_ROOT, outPath)}`);

  let baseline: { stages: StageStatsMap } | null = null;
  if (fs.existsSync(baselinePath) && !updateBaseline) {
    baseline = JSON.parse(fs.readFileSync(baselinePath, "utf8")) as { stages: StageStatsMap };
  }
  const { regressed } = compareAgainstBaseline(stats, baseline);
  if (regressed.length > 0) {
    console.error("\nREGRESSION: stages exceeded baseline p50 by ≥2×:");
    for (const r of regressed) {
      console.error(
        `  ${r.stage}: p50=${r.currentP50.toFixed(1)}ms vs baseline ${r.baselineP50.toFixed(1)}ms (${r.ratio.toFixed(2)}×)`,
      );
    }
    console.error("Re-run with --update-baseline if this is an intentional change.");
    return 1;
  }

  if (updateBaseline || !fs.existsSync(baselinePath)) {
    fs.writeFileSync(baselinePath, JSON.stringify(payload, null, 2));
    console.log(`Wrote ${path.relative(REPO_ROOT, baselinePath)} (baseline ${updateBaseline ? "updated" : "created"})`);
  } else {
    console.log("No regression vs baseline.");
  }
  return 0;
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    console.error(err);
    process.exit(2);
  });
