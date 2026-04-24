// Phase 8: Compose preprocess + vectorize + image_encoder.onnx + risk_model.onnx + calibrator.json
// end-to-end in TypeScript via onnxruntime-node. The TS analog of scripts/run_onnx_pipeline.py.
//
// Prints the 4-decimal predictions and checks them against the pinned Phase 0 pydicom baseline
// (mirai-migration-plan.md §1). Exits non-zero on any mismatch — safe to use as an automated
// manual-verification gate.
//
// Usage:
//   npm install       # pulls onnxruntime-node + tsx
//   npm run pipeline:ts

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { runMirai, createNodeSessions } from "../src/mirai/index.js";
import { loadCalibratorFromFile } from "../src/mirai/calibrator.node.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");

const MODELS_DIR = path.join(REPO_ROOT, "models");
const DEMO_DIR = path.join(REPO_ROOT, "mirai_demo_data");

// Demo input order (matches tests/reference/fixtures/batch_order.json slot order).
const DEMO_FILES = ["ccl1.dcm", "ccr1.dcm", "mlol2.dcm", "mlor2.dcm"];

const PINNED_PYDICOM: Readonly<Record<"year1" | "year2" | "year3" | "year4" | "year5", number>> = {
  year1: 0.0314,
  year2: 0.0505,
  year3: 0.0711,
  year4: 0.0935,
  year5: 0.1052,
};

async function main(): Promise<number> {
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
  const result = await runMirai(files, sessions, calibrator);

  console.log(`[pydicom] predictions: ${JSON.stringify(result.predictions)}`);
  console.log(`[pydicom] slotOrder:   ${JSON.stringify(result.slotOrder)}`);
  const calibStr = Array.from(result.calibrated).map((v) => v.toFixed(8)).join(", ");
  console.log(`[pydicom] calibrated (fp64): [${calibStr}]`);

  const mismatches: string[] = [];
  for (const year of ["year1", "year2", "year3", "year4", "year5"] as const) {
    const got = result.predictions[year];
    const want = PINNED_PYDICOM[year];
    if (got !== want) {
      mismatches.push(`  ${year}: got ${got}, expected ${want}`);
    }
  }
  if (mismatches.length > 0) {
    console.error("[pydicom] MISMATCH vs pinned baseline (mirai-migration-plan.md §1):");
    for (const m of mismatches) console.error(m);
    return 1;
  }
  console.log("[pydicom] OK: rounded predictions bit-equal to pinned baseline");
  return 0;
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    console.error(err);
    process.exit(2);
  });
