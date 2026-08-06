/**
 * deno-mlx demo — a local, streaming clipboard summarizer + chat.
 *
 * A single Deno process: loads a chat model once (on-device, via mlx-c FFI —
 * no Python, no server sidecar) and streams tokens to a browser UI over a
 * chunked HTTP response. Works with plain `deno run`; `deno compile` turns it
 * into a self-contained native binary (the UI is embedded below).
 *
 *   deno task demo                       # default model
 *   DENO_MLX_MODEL=<repoId> deno task demo
 */

import { chat, loadModel } from "../../packages/models/mod.ts";
import { HTML } from "./ui.ts";

const MODEL = Deno.env.get("DENO_MLX_MODEL") ??
  "HuggingFaceTB/SmolLM2-360M-Instruct";
const PORT = Number(Deno.env.get("PORT") ?? 8787);

console.log(`[deno-mlx] loading ${MODEL} …`);
const t0 = performance.now();
const model = await loadModel(MODEL);
console.log(`[deno-mlx] ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

// One model instance -> serialize generations so KV caches never interleave.
let tail = Promise.resolve();
function acquire(): Promise<() => void> {
  const prev = tail;
  let release!: () => void;
  tail = new Promise((r) => (release = r));
  return prev.then(() => release);
}

function readClipboard(): string {
  try {
    const out = new Deno.Command("pbpaste", { stdout: "piped" }).outputSync();
    return new TextDecoder().decode(out.stdout);
  } catch {
    return "";
  }
}

function promptFor(mode: string, text: string): string {
  return mode === "summarize"
    ? `Summarize the following text in one or two clear sentences:\n\n${text}`
    : text;
}

function streamTokens(prompt: string, maxTokens: number): Response {
  const enc = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    async start(controller) {
      const release = await acquire();
      try {
        for await (const { text } of chat(model, prompt, { maxTokens })) {
          controller.enqueue(enc.encode(text));
        }
      } catch (err) {
        controller.enqueue(enc.encode(`\n[error: ${(err as Error).message}]`));
      } finally {
        release();
        controller.close();
      }
    },
  });
  return new Response(body, {
    headers: { "content-type": "text/plain; charset=utf-8", "cache-control": "no-cache" },
  });
}

Deno.serve({ port: PORT }, async (req) => {
  const url = new URL(req.url);
  if (req.method === "GET" && url.pathname === "/") {
    return new Response(HTML.replaceAll("__MODEL__", MODEL), {
      headers: { "content-type": "text/html; charset=utf-8" },
    });
  }
  if (req.method === "GET" && url.pathname === "/clipboard") {
    return new Response(readClipboard());
  }
  if (req.method === "POST" && url.pathname === "/generate") {
    const { mode = "chat", text = "" } = await req.json();
    if (!text.trim()) return new Response("empty input", { status: 400 });
    const maxTokens = mode === "summarize" ? 220 : 320;
    return streamTokens(promptFor(mode, text), maxTokens);
  }
  return new Response("not found", { status: 404 });
});

console.log(`[deno-mlx] http://localhost:${PORT}`);
