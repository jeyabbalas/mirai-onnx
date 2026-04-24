// Phase 9: onnxruntime-web session factories. Uses dynamic imports so library
// consumers that only want preprocess/riskFactors/calibrator don't pay for ORT-web.
//
// Both factories return a MiraiSessions compatible with the backend-agnostic shape
// in runMirai.ts. WebGPU is tried first by default, with WASM fallback; callers can
// pin to WASM via `opts.preferWebGPU = false`.

import type { MiraiSessions, OrtTensorCtor } from "../runMirai.js";

export interface CreateWebSessionsOptions {
  /** Try WebGPU first, fall back to WASM. Default: true. */
  preferWebGPU?: boolean;
  /** Override onnxruntime-web's WASM asset base URL (e.g. "/ort-wasm/"). */
  wasmPaths?: string;
  /** Hard cap on WASM threads. Default: min(navigator.hardwareConcurrency ?? 4, 4). */
  numThreads?: number;
}

type OrtWebModule = {
  InferenceSession: {
    create(
      model: string | ArrayBuffer | ArrayBufferView,
      options?: Record<string, unknown>,
    ): Promise<unknown>;
  };
  Tensor: unknown;
  env: {
    wasm: {
      numThreads?: number;
      wasmPaths?: string;
      [key: string]: unknown;
    };
    [key: string]: unknown;
  };
};

async function loadOrtWeb(): Promise<OrtWebModule> {
  try {
    // Dynamic import keeps ORT-web out of the dep graph when this file is
    // tree-shaken by consumers who don't call createWebSessions*.
    const mod = (await import(/* @vite-ignore */ "onnxruntime-web")) as unknown as OrtWebModule;
    return mod;
  } catch (err) {
    throw new Error(
      "createWebSessions: 'onnxruntime-web' is not installed. Add it as a dependency " +
        "(listed under optionalDependencies on the mirai-onnx-web package). " +
        `Underlying error: ${(err as Error).message}`,
    );
  }
}

function resolveThreads(opts: CreateWebSessionsOptions | undefined): number {
  if (typeof opts?.numThreads === "number") {
    return Math.max(1, Math.min(opts.numThreads, 4));
  }
  const nav = (globalThis as unknown as { navigator?: { hardwareConcurrency?: number } }).navigator;
  const hc = typeof nav?.hardwareConcurrency === "number" ? nav.hardwareConcurrency : 4;
  return Math.max(1, Math.min(hc, 4));
}

function buildSessionOptions(opts?: CreateWebSessionsOptions): Record<string, unknown> {
  const preferWebGPU = opts?.preferWebGPU !== false;
  return {
    executionProviders: preferWebGPU ? ["webgpu", "wasm"] : ["wasm"],
    graphOptimizationLevel: "all",
    // Ask JSEP to land outputs in CPU memory. Not sufficient on its own — the
    // runMirai read path also calls tensor.getData() to force the GPU→CPU
    // copy to complete before the data is consumed — but pairing the two
    // avoids any path where ORT keeps a lazy GPU buffer around.
    preferredOutputLocation: "cpu",
  };
}

function configureEnv(ort: OrtWebModule, opts?: CreateWebSessionsOptions): void {
  ort.env.wasm.numThreads = resolveThreads(opts);
  if (opts?.wasmPaths) {
    ort.env.wasm.wasmPaths = opts.wasmPaths;
  }
}

export async function createWebSessionsFromBytes(
  bytes: { encoder: ArrayBuffer | Uint8Array; risk: ArrayBuffer | Uint8Array },
  opts?: CreateWebSessionsOptions,
): Promise<MiraiSessions> {
  const ort = await loadOrtWeb();
  configureEnv(ort, opts);
  const sessionOpts = buildSessionOptions(opts);
  const [encoder, risk] = await Promise.all([
    ort.InferenceSession.create(bytes.encoder as ArrayBuffer, sessionOpts),
    ort.InferenceSession.create(bytes.risk as ArrayBuffer, sessionOpts),
  ]);
  return {
    encoder: encoder as MiraiSessions["encoder"],
    risk: risk as MiraiSessions["risk"],
    Tensor: ort.Tensor as OrtTensorCtor,
  };
}

export async function createWebSessions(
  urls: { encoder: string; risk: string },
  opts?: CreateWebSessionsOptions,
): Promise<MiraiSessions> {
  const [encoderBytes, riskBytes] = await Promise.all([
    fetchModel(urls.encoder),
    fetchModel(urls.risk),
  ]);
  return createWebSessionsFromBytes({ encoder: encoderBytes, risk: riskBytes }, opts);
}

async function fetchModel(url: string): Promise<ArrayBuffer> {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`createWebSessions: failed to fetch ${url}: ${resp.status} ${resp.statusText}`);
  }
  return resp.arrayBuffer();
}
