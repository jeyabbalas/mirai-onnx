// Node-only helper: load a calibrator JSON from disk. Kept in its own file so
// the browser barrel can avoid the `node:fs` import site entirely. Bundlers
// that tree-shake this module (because browser consumers never reference
// `loadCalibratorFromFile`) will emit no warning.

import fs from "node:fs";
import { loadCalibrator, type Calibrator } from "./calibrator.js";

export function loadCalibratorFromFile(path: string): Calibrator {
  return loadCalibrator(JSON.parse(fs.readFileSync(path, "utf8")));
}
