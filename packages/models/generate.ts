/**
 * Streaming token generation: prompt -> tokens, with KV-cache decode and greedy/temperature sampling.
 * Yields incrementally-detokenized text so the caller can stream it straight to a UI.
 *
 * GPU evaluation runs on Deno's nonblocking FFI path so the event loop stays responsive between tokens.
 * Each call owns a request-local KV cache.
 */

import { Tensor } from "@deno-mlx/tensor";
import { LlamaModel } from "./llama.ts";
import { KVCache } from "./llama.ts";
import type { Tokenizer } from "./tokenizer.ts";
import {
  argmax,
  categorical,
  itemU32Async,
  mulScalar,
  randomKey,
  randomSplit,
} from "./ops.ts";
import { InferenceLock } from "./inference_lock.ts";

export type FinishReason = "stop" | "length" | "cancelled";

export interface GenerateOptions {
  maxTokens?: number;
  /** 0 (default) = greedy; >0 = temperature sampling. */
  temperature?: number;
  /** Seed for reproducible sampling (temperature > 0 only). */
  seed?: number;
  /** Abort generation (including while waiting on the inference lock). */
  signal?: AbortSignal;
  /** Extra stop token ids in addition to the tokenizer EOS. */
  stopTokenIds?: number[];
  /** Optional per-model lock; when omitted, generation is not serialized. */
  lock?: InferenceLock;
}

export interface GeneratedToken {
  id: number;
  /** The newly-decoded text delta for this token (streaming-safe). */
  text: string;
  /** Zero-based index among generated tokens. */
  index: number;
  /** Set when this event ends the generation. */
  finishReason?: FinishReason;
}

export interface GenerateResult {
  text: string;
  tokens: number[];
  finishReason: FinishReason;
}

/**
 * Async generator of tokens.
 * Frees the KV cache when iteration finishes.
 */
export async function* generate(
  model: LlamaModel,
  tokenizer: Tokenizer,
  promptIds: number[],
  opts: GenerateOptions = {},
): AsyncGenerator<GeneratedToken> {
  const release = opts.lock ? await opts.lock.acquire(opts.signal) : undefined;
  try {
    yield* generateUnlocked(model, tokenizer, promptIds, opts);
  } finally {
    release?.();
  }
}

async function* generateUnlocked(
  model: LlamaModel,
  tokenizer: Tokenizer,
  promptIds: number[],
  opts: GenerateOptions,
): AsyncGenerator<GeneratedToken> {
  const maxTokens = opts.maxTokens ?? 256;
  const temperature = opts.temperature ?? 0;
  const signal = opts.signal;
  const stop = eosSet(tokenizer.eosTokenId, opts.stopTokenIds);
  const limit = model.cfg.maxPositionEmbeddings;

  if (promptIds.length >= limit) {
    throw new Error(
      `Prompt length ${promptIds.length} exceeds model context limit ${limit}`,
    );
  }

  const cache = new KVCache(model.cfg.numLayers);
  let key: Tensor | undefined = opts.seed !== undefined
    ? randomKey(opts.seed)
    : undefined;

  const produced: number[] = [];
  let emitted = "";
  let input = promptIds;
  let finishReason: FinishReason = "length";

  try {
    for (let n = 0; n < maxTokens; n++) {
      if (signal?.aborted) {
        finishReason = "cancelled";
        break;
      }

      const step = await nextToken(model, input, cache, temperature, key);
      key = step.nextKey;

      if (stop.has(step.id)) {
        finishReason = "stop";
        break;
      }

      produced.push(step.id);
      const full = tokenizer.decode(produced);
      const text = full.slice(emitted.length);
      emitted = full;

      const last = n === maxTokens - 1;
      yield {
        id: step.id,
        text,
        index: n,
        finishReason: last ? "length" : undefined,
      };
      if (last) {
        finishReason = "length";
        return;
      }
      input = [step.id];
    }

    // Terminal marker when we stop without a final length token (EOS / abort).
    yield {
      id: -1,
      text: "",
      index: produced.length,
      finishReason,
    };
  } finally {
    key?.[Symbol.dispose]();
    cache[Symbol.dispose]();
  }
}

/** Convenience: run generation to completion and return the full string. */
export async function generateText(
  model: LlamaModel,
  tokenizer: Tokenizer,
  promptIds: number[],
  opts: GenerateOptions = {},
): Promise<string> {
  const result = await generateResult(model, tokenizer, promptIds, opts);
  return result.text;
}

/** Run generation to completion and return text, token ids, and finish reason. */
export async function generateResult(
  model: LlamaModel,
  tokenizer: Tokenizer,
  promptIds: number[],
  opts: GenerateOptions = {},
): Promise<GenerateResult> {
  let text = "";
  const tokens: number[] = [];
  let finishReason: FinishReason = "length";
  for await (const t of generate(model, tokenizer, promptIds, opts)) {
    if (t.finishReason) finishReason = t.finishReason;
    if (t.id < 0) continue;
    text += t.text;
    tokens.push(t.id);
  }
  return { text, tokens, finishReason };
}

async function nextToken(
  model: LlamaModel,
  input: number[],
  cache: KVCache,
  temperature: number,
  key: Tensor | undefined,
): Promise<{ id: number; nextKey: Tensor | undefined }> {
  using logits = model.forward(input, cache);
  if (temperature <= 0) {
    using idx = argmax(logits);
    return { id: await itemU32Async(idx), nextKey: key };
  }

  let useKey: Tensor | undefined;
  let nextKey: Tensor | undefined = key;
  if (key) {
    const split = randomSplit(key);
    nextKey = split.keep;
    useKey = split.use;
  }
  try {
    using scaled = mulScalar(logits, 1 / temperature);
    using idx = categorical(scaled, useKey);
    return { id: await itemU32Async(idx), nextKey };
  } finally {
    useKey?.[Symbol.dispose]();
  }
}

function eosSet(
  eos: number | number[] | undefined,
  extra?: number[],
): Set<number> {
  const set = new Set<number>();
  if (eos !== undefined) {
    for (const id of Array.isArray(eos) ? eos : [eos]) set.add(id);
  }
  if (extra) { for (const id of extra) set.add(id); }
  return set;
}
