// Creates symlinks from demo/public/ into the repo-level models/ and
// mirai_demo_data/ directories so Vite can serve them without copying ~100MB
// of artifacts into a second directory. Both source dirs are gitignored.
//
// Also copies the onnxruntime-web WASM assets into demo/public/ort/ so the
// WASM execution provider can fetch them from the same origin (required for
// cross-origin-isolated threaded mode).
//
// Idempotent: overwrites existing symlinks with current targets.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DEMO_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(DEMO_ROOT, "..");
const PUBLIC_DIR = path.join(DEMO_ROOT, "public");

function linkInto(target, linkName) {
  const linkPath = path.join(PUBLIC_DIR, linkName);
  if (!fs.existsSync(target)) {
    console.warn(`[link-models] skip ${linkName}: target does not exist at ${target}`);
    return;
  }
  if (fs.existsSync(linkPath) || fs.lstatSync(linkPath, { throwIfNoEntry: false })) {
    try {
      fs.unlinkSync(linkPath);
    } catch {
      // Might be a directory symlink on some platforms.
      fs.rmSync(linkPath, { recursive: true, force: true });
    }
  }
  fs.symlinkSync(target, linkPath, "dir");
  console.log(`[link-models] ${linkName} → ${path.relative(DEMO_ROOT, target)}`);
}

function copyOrtWasm() {
  const ortDir = path.join(REPO_ROOT, "node_modules", "onnxruntime-web", "dist");
  const outDir = path.join(PUBLIC_DIR, "ort");
  if (!fs.existsSync(ortDir)) {
    console.warn(`[link-models] skip ort: ${ortDir} not found — run 'npm install' at repo root first`);
    return;
  }
  fs.mkdirSync(outDir, { recursive: true });
  // Copy only the runtime assets the ORT-web runtime actually fetches. The .mjs
  // loaders reference their sibling .wasm by name; keep both together.
  const patterns = [/^ort-wasm-simd-threaded(\.jsep|\.asyncify|\.jspi)?\.(mjs|wasm)$/];
  for (const entry of fs.readdirSync(ortDir)) {
    if (patterns.some((re) => re.test(entry))) {
      fs.copyFileSync(path.join(ortDir, entry), path.join(outDir, entry));
    }
  }
  console.log(`[link-models] copied onnxruntime-web WASM assets → public/ort/`);
}

function copyCoiServiceWorker() {
  // coi-serviceworker enables COOP/COEP via a service worker on hosts that
  // can't set response headers (GH Pages). Its npm package lives in the
  // demo's own node_modules — unlike onnxruntime-web which sits at the repo
  // root's node_modules because it's pulled in through the file:.. package.
  const pkgDir = path.join(DEMO_ROOT, "node_modules", "coi-serviceworker");
  const src = path.join(pkgDir, "coi-serviceworker.min.js");
  const dst = path.join(PUBLIC_DIR, "coi-serviceworker.min.js");
  if (!fs.existsSync(src)) {
    console.warn(`[link-models] skip coi-serviceworker: ${src} not found — run 'npm install' in demo/ first`);
    return;
  }
  fs.copyFileSync(src, dst);
  console.log(`[link-models] copied coi-serviceworker.min.js → public/`);
}

fs.mkdirSync(PUBLIC_DIR, { recursive: true });
linkInto(path.join(REPO_ROOT, "models"), "models");
linkInto(path.join(REPO_ROOT, "mirai_demo_data"), "sample");
copyOrtWasm();
copyCoiServiceWorker();
