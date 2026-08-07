/**
 * Minimal streaming chat (compat entry for `deno task chat`).
 *
 * Prefer the full CLI: `deno task cli -- chat "…"`.
 */

import { chat } from "@deno-mlx/models";
import { loadChatModel } from "./shared/models.ts";

if (Deno.build.os !== "darwin" || Deno.build.arch !== "aarch64") {
  console.error("deno-mlx requires Apple Silicon macOS.");
  Deno.exit(1);
}

const prompt = Deno.args[0]?.startsWith("--")
  ? "In one sentence, what is Deno?"
  : (Deno.args[0] ?? "In one sentence, what is Deno?");

using m = await loadChatModel();
const enc = new TextEncoder();
const t0 = performance.now();
let n = 0;
for await (const t of chat(m, prompt, { maxTokens: 100 })) {
  if (t.id < 0) continue;
  await Deno.stdout.write(enc.encode(t.text));
  n++;
}
const dt = (performance.now() - t0) / 1000;
console.log(`\n\n[${n} tokens, ${(n / Math.max(dt, 1e-6)).toFixed(1)} tok/s]`);
