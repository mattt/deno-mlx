/**
 * @deno-mlx/models — Layer 3: model runners.
 *
 * v0 ships chat-model generation (Llama family: SmolLM2, Llama 3.x, …). Whisper
 * and embeddings follow. Weights load from the shared HF Hub cache; tokenization
 * uses transformers.js.
 *
 * ```ts
 * import { loadModel, chat } from "@deno-mlx/models";
 *
 * const m = await loadModel("HuggingFaceTB/SmolLM2-135M-Instruct");
 * for await (const { text } of chat(m, "Name three primes.")) {
 *   await Deno.stdout.write(new TextEncoder().encode(text));
 * }
 * ```
 */

import { type LlamaConfig, loadConfig } from "./config.ts";
import { LlamaModel } from "./llama.ts";
import { loadTokenizer, type Tokenizer } from "./tokenizer.ts";
import {
  generate,
  type GeneratedToken,
  type GenerateOptions,
} from "./generate.ts";

export { KVCache, LlamaModel } from "./llama.ts";
export { type LlamaConfig, loadConfig } from "./config.ts";
export { hubFile, resolveSnapshot } from "./hub.ts";
export { Weights } from "./weights.ts";
export { type ChatMessage, loadTokenizer, type Tokenizer } from "./tokenizer.ts";
export {
  generate,
  type GeneratedToken,
  type GenerateOptions,
  generateText,
} from "./generate.ts";

export interface LoadedModel {
  model: LlamaModel;
  tokenizer: Tokenizer;
  cfg: LlamaConfig;
  [Symbol.dispose](): void;
}

/** Load config + weights + tokenizer for a Hub repo id. */
export async function loadModel(repoId: string): Promise<LoadedModel> {
  const cfg = loadConfig(repoId);
  const model = LlamaModel.load(repoId, cfg);
  const tokenizer = await loadTokenizer(repoId);
  return {
    model,
    tokenizer,
    cfg,
    [Symbol.dispose]() {
      model[Symbol.dispose]();
    },
  };
}

/** Stream a chat reply to a single user message (applies the chat template). */
export function chat(
  loaded: LoadedModel,
  message: string,
  opts?: GenerateOptions,
): AsyncGenerator<GeneratedToken> {
  const ids = loaded.tokenizer.applyChatTemplate([
    { role: "user", content: message },
  ]);
  return generate(loaded.model, loaded.tokenizer, ids, opts);
}
