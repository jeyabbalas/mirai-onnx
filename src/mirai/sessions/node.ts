// Phase 9: onnxruntime-node session factory. Dynamic import so importing this
// module from a browser build errors out loudly instead of pulling native bindings.

import type { MiraiSessions, OrtTensorCtor } from "../runMirai.js";

type OrtNodeModule = {
  InferenceSession: {
    create(path: string, options?: Record<string, unknown>): Promise<unknown>;
  };
  Tensor: unknown;
};

async function loadOrtNode(): Promise<OrtNodeModule> {
  try {
    const mod = (await import("onnxruntime-node")) as unknown as OrtNodeModule;
    return mod;
  } catch (err) {
    throw new Error(
      "createNodeSessions: 'onnxruntime-node' is not installed. " +
        `Underlying error: ${(err as Error).message}`,
    );
  }
}

export async function createNodeSessions(
  paths: { encoder: string; risk: string },
): Promise<MiraiSessions> {
  const ort = await loadOrtNode();
  const [encoder, risk] = await Promise.all([
    ort.InferenceSession.create(paths.encoder),
    ort.InferenceSession.create(paths.risk),
  ]);
  return {
    encoder: encoder as MiraiSessions["encoder"],
    risk: risk as MiraiSessions["risk"],
    Tensor: ort.Tensor as OrtTensorCtor,
  };
}
