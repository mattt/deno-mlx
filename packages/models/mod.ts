/**
 * @deno-mlx/models — Layer 3: model runners.
 *
 * Ships chat-model generation (Llama family: SmolLM2, Llama 3.x, …) and BERT embeddings.
 * Weights load from the shared HF Hub cache or an explicit local directory;
 * tokenization uses transformers.js.
 *
 * ```ts
 * import { loadModel, chat } from "@deno-mlx/models";
 *
 * using m = await loadModel("HuggingFaceTB/SmolLM2-135M-Instruct");
 * for await (const { text } of chat(m, "Name three primes.")) {
 *   await Deno.stdout.write(new TextEncoder().encode(text));
 * }
 * ```
 *
 * Concurrent calls on the same LoadedModel / Embedder are serialized.
 * The mlx-c dylib is process-wide (`openMlxc`);
 * unloading mid-process is unsupported.
 */

import { type LlamaConfig, loadConfigFromDir } from "./config.ts";
import { LlamaModel } from "./llama.ts";
import {
  type ChatMessage,
  loadTokenizer,
  loadTokenizerFromDir,
  type Tokenizer,
} from "./tokenizer.ts";
import { generate, type GeneratedToken, type GenerateOptions } from "./generate.ts";
import { InferenceLock } from "./inference_lock.ts";
import { assertModelReady, resolveModelDir, resolveSnapshot } from "./hub.ts";

export { KVCache, LlamaModel } from "./llama.ts";
export {
  type BertConfig,
  BertModel,
  loadBertConfig,
  loadBertConfigFromDir,
} from "./bert.ts";
export {
  cosineSimilarity,
  type Embedder,
  type EmbedOptions,
  loadEmbedder,
  loadEmbedderFromDir,
} from "./embeddings.ts";
export { type LlamaConfig, loadConfig, loadConfigFromDir } from "./config.ts";
export {
  assertModelReady,
  hubCacheDir,
  hubFile,
  resolveModelDir,
  resolveSnapshot,
} from "./hub.ts";
export { resolveWeightFiles } from "./safetensors.ts";
export { Weights } from "./weights.ts";
export {
  type ChatMessage,
  type ChatTemplateOptions,
  loadTokenizer,
  loadTokenizerFromDir,
  type Tokenizer,
} from "./tokenizer.ts";
export {
  type FinishReason,
  generate,
  type GeneratedToken,
  type GenerateOptions,
  type GenerateResult,
  generateResult,
  generateText,
} from "./generate.ts";
export { InferenceLock } from "./inference_lock.ts";

export interface LoadedModel {
  model: LlamaModel;
  tokenizer: Tokenizer;
  cfg: LlamaConfig;
  /** Serializes generate/chat on this instance. */
  readonly lock: InferenceLock;
  [Symbol.dispose](): void;
}

/** Load config + weights + tokenizer for a Hub repo id. */
export function loadModel(repoId: string): Promise<LoadedModel> {
  return loadModelFromDir(resolveSnapshot(repoId), repoId);
}

/** Load from an explicit model directory (config.json + safetensors + tokenizer). */
export async function loadModelFromDir(
  modelDir: string,
  tokenizerSource?: string,
): Promise<LoadedModel> {
  const dir = resolveModelDir(modelDir);
  assertModelReady(dir, "chat");
  const cfg = loadConfigFromDir(dir);
  const model = LlamaModel.loadDir(dir, cfg);
  const tokenizer = tokenizerSource
    ? await loadTokenizer(tokenizerSource)
    : await loadTokenizerFromDir(dir);
  const lock = new InferenceLock();
  return {
    model,
    tokenizer,
    cfg,
    lock,
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
  return chatMessages(loaded, [{ role: "user", content: message }], opts);
}

/** Stream a chat reply for a full message history. */
export function chatMessages(
  loaded: LoadedModel,
  messages: ChatMessage[],
  opts?: GenerateOptions,
): AsyncGenerator<GeneratedToken> {
  const ids = loaded.tokenizer.applyChatTemplate(messages);
  return generate(loaded.model, loaded.tokenizer, ids, {
    ...opts,
    lock: opts?.lock ?? loaded.lock,
  });
}
