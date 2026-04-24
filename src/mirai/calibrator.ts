export interface CalibratorYear {
  readonly index: number;
  readonly base_slope: number;
  readonly base_offset: number;
  readonly calibrator_slope: number;
  readonly calibrator_offset: number;
}

export interface Calibrator {
  readonly schema_version: number;
  readonly source_pickle_sha256: string;
  readonly years: readonly CalibratorYear[];
}

export const CALIBRATOR_SCHEMA_VERSION = 1;
export const CALIBRATOR_N_YEARS = 5;

const PARAM_KEYS = [
  "base_slope",
  "base_offset",
  "calibrator_slope",
  "calibrator_offset",
] as const;

function isFiniteNumber(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

export function loadCalibrator(json: unknown): Calibrator {
  if (typeof json !== "object" || json === null) {
    throw new Error("loadCalibrator: payload is not an object");
  }
  const obj = json as Record<string, unknown>;
  if (obj.schema_version !== CALIBRATOR_SCHEMA_VERSION) {
    throw new Error(
      `loadCalibrator: unsupported schema_version ${String(obj.schema_version)} (expected ${CALIBRATOR_SCHEMA_VERSION})`,
    );
  }
  if (typeof obj.source_pickle_sha256 !== "string") {
    throw new Error("loadCalibrator: source_pickle_sha256 is not a string");
  }
  if (!Array.isArray(obj.years)) {
    throw new Error("loadCalibrator: missing or non-array 'years'");
  }
  if (obj.years.length !== CALIBRATOR_N_YEARS) {
    throw new Error(
      `loadCalibrator: expected ${CALIBRATOR_N_YEARS} year entries, got ${obj.years.length}`,
    );
  }

  const parsed: CalibratorYear[] = new Array(CALIBRATOR_N_YEARS);
  for (const entry of obj.years) {
    if (typeof entry !== "object" || entry === null) {
      throw new Error(`loadCalibrator: year entry is not an object: ${JSON.stringify(entry)}`);
    }
    const y = entry as Record<string, unknown>;
    if (typeof y.index !== "number" || !Number.isInteger(y.index) || y.index < 0 || y.index >= CALIBRATOR_N_YEARS) {
      throw new Error(`loadCalibrator: invalid year index ${String(y.index)}`);
    }
    if (parsed[y.index] !== undefined) {
      throw new Error(`loadCalibrator: duplicate year index ${y.index}`);
    }
    const scalars: Record<string, number> = {};
    for (const k of PARAM_KEYS) {
      if (!isFiniteNumber(y[k])) {
        throw new Error(`loadCalibrator: year ${y.index} ${k} is non-finite: ${String(y[k])}`);
      }
      scalars[k] = y[k] as number;
    }
    parsed[y.index] = {
      index: y.index,
      base_slope: scalars.base_slope,
      base_offset: scalars.base_offset,
      calibrator_slope: scalars.calibrator_slope,
      calibrator_offset: scalars.calibrator_offset,
    };
  }

  // Defensive: new Array(N) leaves holes until every index is assigned; reject if any slot is still empty.
  for (let i = 0; i < CALIBRATOR_N_YEARS; i++) {
    if (parsed[i] === undefined) {
      throw new Error(`loadCalibrator: year index ${i} not present`);
    }
  }

  return {
    schema_version: CALIBRATOR_SCHEMA_VERSION,
    source_pickle_sha256: obj.source_pickle_sha256,
    years: parsed,
  };
}

// Note: `loadCalibratorFromFile` lives in `./calibrator.node.ts` — import it
// directly from there in Node contexts. The barrel keeps Node-only code out of
// the browser bundle's import graph, so Vite/Rollup builds stay clean.

// Line-for-line port of MiraiCalibrator.predict_proba(..., expand=False):
//   _y = base_slope * p + base_offset
//   _y = calibrator_slope * _y + calibrator_offset
//   return 1 / (1 + exp(_y))     // note: exp(+_y), not exp(-_y)
export function calibrateYear(p: number, year: CalibratorYear): number {
  let y = year.base_slope * p + year.base_offset;
  y = year.calibrator_slope * y + year.calibrator_offset;
  return 1.0 / (1.0 + Math.exp(y));
}

export function calibrateAll(rawSigmoid: ArrayLike<number>, calibrator: Calibrator): Float64Array {
  if (rawSigmoid.length !== calibrator.years.length) {
    throw new Error(
      `calibrateAll: rawSigmoid.length=${rawSigmoid.length} does not match calibrator.years.length=${calibrator.years.length}`,
    );
  }
  const out = new Float64Array(calibrator.years.length);
  for (let i = 0; i < calibrator.years.length; i++) {
    out[i] = calibrateYear(rawSigmoid[i], calibrator.years[i]);
  }
  return out;
}
