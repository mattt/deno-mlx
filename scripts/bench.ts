#!/usr/bin/env -S deno run -P=mlx --allow-write --allow-net
/**
 * Lightweight reference benchmarks (advisory for 0.1.0).
 *
 *   deno task bench
 */

import { chat, generateResult, loadModel } from "@deno-mlx/models";
import { loadEmbedder } from "@deno-mlx/models";

const chatRepo = Deno.env.get("DENO_MLX_MODEL") ??
  "HuggingFaceTB/SmolLM2-135M-Instruct";
const embedRepo = Deno.env.get("DENO_MLX_EMBED_MODEL") ??
  "sentence-transformers/all-MiniLM-L6-v2";

if (Deno.build.os !== "darwin" || Deno.build.arch !== "aarch64") {
  console.error("bench requires Apple Silicon macOS");
  Deno.exit(1);
}

using model = await loadModel(chatRepo);
const prompt = "Name three primes.";
const ids = model.tokenizer.applyChatTemplate([{ role: "user", content: prompt }]);

let beats = 0;
const timer = setInterval(() => beats++, 5);
const t0 = performance.now();
const result = await generateResult(model.model, model.tokenizer, ids, {
  maxTokens: 64,
  lock: model.lock,
});
const dt = (performance.now() - t0) / 1000;
clearInterval(timer);

console.log("chat");
console.log(`  model: ${chatRepo}`);
console.log(`  tokens: ${result.tokens.length}`);
console.log(`  decode tok/s: ${(result.tokens.length / dt).toFixed(2)}`);
console.log(`  wall_s: ${dt.toFixed(3)}`);
console.log(`  event_loop_beats: ${beats}`);
console.log(`  finish: ${result.finishReason}`);

try {
  using emb = await loadEmbedder(embedRepo);
  const t1 = performance.now();
  await emb.embed("The cat sits on the mat.");
  const embedMs = performance.now() - t1;
  console.log("embed");
  console.log(`  model: ${embedRepo}`);
  console.log(`  latency_ms: ${embedMs.toFixed(1)}`);
  console.log(`  dim: ${emb.dimension}`);
} catch (err) {
  console.log(`embed skipped: ${(err as Error).message}`);
}

// abort latency
const ac = new AbortController();
const tAbort = performance.now();
let n = 0;
for await (
  const t of chat(model, prompt, { maxTokens: 128, signal: ac.signal })
) {
  if (t.id >= 0) n++;
  if (n === 2) ac.abort();
  if (t.finishReason === "cancelled") break;
}
console.log("abort");
console.log(`  tokens_before_cancel: ${n}`);
console.log(`  abort_wall_ms: ${(performance.now() - tAbort).toFixed(1)}`);
