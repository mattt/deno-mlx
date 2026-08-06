/**
 * Minimal streaming chat over a local MLX model.
 *
 *   deno task chat                       # default model + prompt
 *   deno task chat <repoId> "<prompt>"
 *
 * The model is read from the shared Hugging Face cache
 * (`hf download HuggingFaceTB/SmolLM2-135M-Instruct` first).
 */

import { chat, loadModel } from "../packages/models/mod.ts";

const repo = Deno.args[0] ?? "HuggingFaceTB/SmolLM2-135M-Instruct";
const prompt = Deno.args[1] ?? "In one sentence, what is Deno?";

using m = await loadModel(repo);
const enc = new TextEncoder();

const t0 = performance.now();
let n = 0;
for await (const { text } of chat(m, prompt, { maxTokens: 100 })) {
  await Deno.stdout.write(enc.encode(text));
  n++;
}
const dt = (performance.now() - t0) / 1000;
console.log(`\n\n[${n} tokens, ${(n / dt).toFixed(1)} tok/s]`);
