import { defineConfig } from "vite";

// onnxruntime-web's threaded WASM backend needs cross-origin isolation. These
// headers enable SharedArrayBuffer; without them, numThreads silently clamps
// to 1 and WASM inference is ~4× slower. WebGPU ignores these but they don't
// hurt.
const crossOriginIsolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  server: {
    headers: crossOriginIsolationHeaders,
    // The demo links mirai-onnx-web via `file:..`, which makes the source
    // files in the parent repo import `onnxruntime-web` from that repo's
    // node_modules. The emscripten .mjs loader then fetches its sibling
    // .wasm via /@fs/<parent-path>/... — Vite's default fs sandbox blocks
    // anything above the demo root (403). Allow the parent repo too.
    fs: { allow: [".."] },
  },
  preview: {
    headers: crossOriginIsolationHeaders,
  },
  // Force both the demo's static `import "onnxruntime-web"` and the
  // transitive dynamic import inside src/mirai/sessions/web.ts to resolve
  // to the SAME copy. Without this, two ORT instances load: setting
  // `ort.env.*` on one doesn't affect the other (caused the silent wasmPaths
  // failures we tried to debug earlier).
  resolve: {
    dedupe: ["onnxruntime-web"],
  },
  optimizeDeps: {
    // ORT-web does its own WASM lazy-loading; don't let Vite pre-bundle it.
    // onnxruntime-node has native .node bindings that esbuild can't load —
    // exclude it so the dep scanner doesn't try to pre-bundle it. The browser
    // barrel only reaches onnxruntime-node via a dynamic import inside
    // sessions/node.ts, which the browser never actually calls.
    exclude: ["onnxruntime-web", "onnxruntime-node"],
  },
  ssr: {
    noExternal: [],
    external: ["onnxruntime-node"],
  },
  build: {
    target: "es2022",
    rollupOptions: {
      external: ["onnxruntime-node"],
    },
  },
});
