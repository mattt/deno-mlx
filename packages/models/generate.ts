/**
 * Streaming token generation: prompt -> tokens, with KV-cache decode and
 * greedy/temperature sampling. Yields incrementally-detokenized text so the
 * caller can stream it straight to a UI.
 */

import { LlamaModel } from "./llama.ts";
import { KVCache } from "./llama.ts";
import type { Tokenizer } from "./tokenizer.ts";
import { argmax, categorical, itemU32, mulScalar, randomKey } from "./ops.ts";

export interface GenerateOptions {
  maxTokens?: number;
  /** 0 (default) = greedy; >0 = temperature sampling. */
  temperature?: number;
  /** Seed for reproducible sampling (temperature > 0 only). */
  seed?: number;
}

export interface GeneratedToken {
  id: number;
  /** The newly-decoded text delta for this token (streaming-safe). */
  text: string;
}

/** Async generator of tokens. Frees the KV cache when iteration finishes. */
export async function* generate(
  model: LlamaModel,
  tokenizer: Tokenizer,
  promptIds: number[],
  opts: GenerateOptions = {},
): AsyncGenerator<GeneratedToken> {
  const maxTokens = opts.maxTokens ?? 256;
  const temperature = opts.temperature ?? 0;
  const eos = eosSet(tokenizer.eosTokenId);

  using cache = new KVCache(model.cfg.numLayers);
  using key = opts.seed !== undefined ? randomKey(opts.seed) : undefined;

  const produced: number[] = [];
  let emitted = ""; // detokenize incrementally: decode all, emit the delta
  let input = promptIds;

  for (let n = 0; n < maxTokens; n++) {
    const id = nextToken(model, input, cache, temperature, key);
    if (eos.has(id)) break;

    produced.push(id);
    const full = tokenizer.decode(produced);
    const text = full.slice(emitted.length);
    emitted = full;

    yield { id, text };
    input = [id];
    await Promise.resolve(); // give the event loop a turn between tokens
  }
}

/** Convenience: run generation to completion and return the full string. */
export async function generateText(
  model: LlamaModel,
  tokenizer: Tokenizer,
  promptIds: number[],
  opts: GenerateOptions = {},
): Promise<string> {
  let out = "";
  for await (const t of generate(model, tokenizer, promptIds, opts)) out += t.text;
  return out;
}

function nextToken(
  model: LlamaModel,
  input: number[],
  cache: KVCache,
  temperature: number,
  key: import("@deno-mlx/tensor").Tensor | undefined,
): number {
  using logits = model.forward(input, cache);
  if (temperature <= 0) {
    using idx = argmax(logits);
    return itemU32(idx);
  }
  using scaled = mulScalar(logits, 1 / temperature);
  using idx = categorical(scaled, key);
  return itemU32(idx);
}

function eosSet(eos: number | number[] | undefined): Set<number> {
  if (eos === undefined) return new Set();
  return new Set(Array.isArray(eos) ? eos : [eos]);
}
