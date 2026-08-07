/**
 * deno-mlx web demo — local streaming chat/summarizer + embeddings.
 *
 *   deno task web
 *   DENO_MLX_MODEL=<repoId> deno task web
 *
 * Binds to 127.0.0.1 only.
 * Clipboard paste needs --allow-run (optional).
 */

import { createAppState, makeHandler } from "../shared/handler.ts";

const PORT = Number(Deno.env.get("PORT") ?? 8787);
const HOST = "127.0.0.1";

if (Deno.build.os !== "darwin" || Deno.build.arch !== "aarch64") {
  console.error("deno-mlx requires Apple Silicon macOS.");
  Deno.exit(1);
}

const state = await createAppState({ withEmbeddings: true });
const handler = makeHandler(state);

Deno.serve({ port: PORT, hostname: HOST }, handler);
console.log(`[deno-mlx] http://${HOST}:${PORT}`);
