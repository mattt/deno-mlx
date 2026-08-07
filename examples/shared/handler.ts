/**
 * Loopback HTTP application shared by the web and desktop examples.
 *
 * Endpoints:
 *   GET  /           HTML UI
 *   GET  /health     JSON status
 *   POST /generate   { mode, text, maxTokens? } → streaming text/plain
 *   POST /embed      { a, b } → { similarity, dim }
 *   GET  /clipboard  optional;
 *                    requires --allow-run for pbpaste
 */

import {
  chat,
  cosineSimilarity,
  type Embedder,
  type LoadedModel,
} from "@deno-mlx/models";
import { HTML } from "../summarizer/ui.ts";
import {
  labelOf,
  loadChatModel,
  loadEmbedModel,
  resolveChatSource,
  resolveEmbedSource,
} from "./models.ts";

const MAX_BODY = 64 * 1024;
const MAX_TOKENS_CHAT = 320;
const MAX_TOKENS_SUMMARIZE = 220;

export interface AppState {
  chatModel: LoadedModel;
  embedder?: Embedder;
  chatLabel: string;
  embedLabel: string;
}

export async function createAppState(opts?: {
  withEmbeddings?: boolean;
}): Promise<AppState> {
  const chatSrc = resolveChatSource();
  console.log(`[deno-mlx] loading chat model ${chatSrc.value} …`);
  const t0 = performance.now();
  const chatModel = await loadChatModel();
  console.log(
    `[deno-mlx] chat ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`,
  );

  let embedder: Embedder | undefined;
  let embedLabel = "";
  if (opts?.withEmbeddings !== false) {
    const embedSrc = resolveEmbedSource();
    embedLabel = labelOf(embedSrc);
    try {
      console.log(`[deno-mlx] loading embedder ${embedSrc.value} …`);
      const t1 = performance.now();
      embedder = await loadEmbedModel();
      console.log(
        `[deno-mlx] embedder ready in ${((performance.now() - t1) / 1000).toFixed(1)}s`,
      );
    } catch (err) {
      console.warn(
        `[deno-mlx] embeddings unavailable: ${(err as Error).message}`,
      );
    }
  }

  return {
    chatModel,
    embedder,
    chatLabel: labelOf(chatSrc),
    embedLabel,
  };
}

export function makeHandler(state: AppState): (req: Request) => Promise<Response> {
  return async (req) => {
    const url = new URL(req.url);
    try {
      if (req.method === "GET" && url.pathname === "/") {
        return new Response(HTML.replaceAll("__MODEL__", state.chatLabel), {
          headers: { "content-type": "text/html; charset=utf-8" },
        });
      }
      if (req.method === "GET" && url.pathname === "/health") {
        return Response.json({
          ok: true,
          chatModel: state.chatLabel,
          embedModel: state.embedder ? state.embedLabel : null,
        });
      }
      if (req.method === "GET" && url.pathname === "/clipboard") {
        return new Response(readClipboard());
      }
      if (req.method === "POST" && url.pathname === "/generate") {
        const body = await readJson(req);
        const mode = String(body.mode ?? "chat");
        const text = String(body.text ?? "");
        if (!text.trim()) {
          return Response.json({ error: "empty input" }, { status: 400 });
        }
        const maxTokens = Math.min(
          Number(body.maxTokens) ||
            (mode === "summarize" ? MAX_TOKENS_SUMMARIZE : MAX_TOKENS_CHAT),
          512,
        );
        const prompt = mode === "summarize"
          ? `Summarize the following text in one or two clear sentences:\n\n${text}`
          : text;
        return streamTokens(state.chatModel, prompt, maxTokens, req.signal);
      }
      if (req.method === "POST" && url.pathname === "/embed") {
        if (!state.embedder) {
          return Response.json(
            { error: "embeddings model not loaded" },
            { status: 503 },
          );
        }
        const body = await readJson(req);
        const a = String(body.a ?? "");
        const b = String(body.b ?? "");
        if (!a.trim() || !b.trim()) {
          return Response.json({ error: "a and b required" }, { status: 400 });
        }
        const va = await state.embedder.embed(a, { signal: req.signal });
        const vb = await state.embedder.embed(b, { signal: req.signal });
        return Response.json({
          similarity: cosineSimilarity(va, vb),
          dim: state.embedder.dimension,
        });
      }
      return new Response("not found", { status: 404 });
    } catch (err) {
      const message = (err as Error).message ?? String(err);
      const status = message.includes("aborted") ? 499 : 500;
      return Response.json({ error: message }, { status });
    }
  };
}

function streamTokens(
  model: LoadedModel,
  prompt: string,
  maxTokens: number,
  signal: AbortSignal,
): Response {
  const enc = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (
          const { text, id } of chat(model, prompt, { maxTokens, signal })
        ) {
          if (id < 0) continue;
          controller.enqueue(enc.encode(text));
        }
      } catch (err) {
        if (!(err instanceof DOMException && err.name === "AbortError")) {
          controller.enqueue(
            enc.encode(`\n[error: ${(err as Error).message}]`),
          );
        }
      } finally {
        controller.close();
      }
    },
    cancel() {
      // Client disconnect — AbortSignal on req is also aborted by Deno.serve.
    },
  });
  return new Response(body, {
    headers: {
      "content-type": "text/plain; charset=utf-8",
      "cache-control": "no-cache",
    },
  });
}

async function readJson(req: Request): Promise<Record<string, unknown>> {
  const len = Number(req.headers.get("content-length") ?? 0);
  if (len > MAX_BODY) throw new Error("request body too large");
  const buf = await req.arrayBuffer();
  if (buf.byteLength > MAX_BODY) throw new Error("request body too large");
  return JSON.parse(new TextDecoder().decode(buf));
}

function readClipboard(): string {
  try {
    const out = new Deno.Command("pbpaste", { stdout: "piped" }).outputSync();
    return new TextDecoder().decode(out.stdout);
  } catch {
    return "";
  }
}
