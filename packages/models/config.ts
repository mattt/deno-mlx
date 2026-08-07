/**
 * Model config parsing (config.json).
 * v0 targets the Llama family (SmolLM2, Llama 3.x, and close relatives):
 * RMSNorm, RoPE, SwiGLU MLP, GQA.
 */

import { hubFile, resolveModelDir } from "./hub.ts";

export interface LlamaConfig {
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  intermediateSize: number;
  vocabSize: number;
  rmsNormEps: number;
  ropeTheta: number;
  tieWordEmbeddings: boolean;
  maxPositionEmbeddings: number;
  modelType: string;
}

/** Architectures with a Llama-family forward path in this package. */
const SUPPORTED_TYPES = new Set(["llama", "mistral", "qwen2"]);

// deno-lint-ignore no-explicit-any
function coerce(raw: any, source: string): LlamaConfig {
  const modelType = String(raw.model_type ?? "llama");
  if (!SUPPORTED_TYPES.has(modelType)) {
    throw new Error(
      `Unsupported model_type "${modelType}" in ${source}. ` +
        `Supported chat architectures: ${[...SUPPORTED_TYPES].join(", ")}.`,
    );
  }

  const heads = Number(raw.num_attention_heads);
  const hiddenSize = Number(raw.hidden_size);
  const numLayers = Number(raw.num_hidden_layers);
  if (
    !Number.isFinite(heads) || !Number.isFinite(hiddenSize) || !Number.isFinite(numLayers)
  ) {
    throw new Error(`Invalid Llama config in ${source}: missing required fields`);
  }

  return {
    hiddenSize,
    numLayers,
    numHeads: heads,
    numKVHeads: Number(raw.num_key_value_heads ?? heads),
    headDim: Number(raw.head_dim ?? Math.floor(hiddenSize / heads)),
    intermediateSize: Number(raw.intermediate_size),
    vocabSize: Number(raw.vocab_size),
    rmsNormEps: Number(raw.rms_norm_eps ?? 1e-5),
    ropeTheta: Number(raw.rope_theta ?? 10000),
    tieWordEmbeddings: Boolean(raw.tie_word_embeddings ?? false),
    maxPositionEmbeddings: Number(raw.max_position_embeddings ?? 2048),
    modelType,
  };
}

/** Load Llama config from a Hub repo id (uses the shared HF cache). */
export function loadConfig(repoId: string): LlamaConfig {
  const path = hubFile(repoId, "config.json");
  return coerce(JSON.parse(Deno.readTextFileSync(path)), path);
}

/** Load Llama config from an explicit model directory. */
export function loadConfigFromDir(modelDir: string): LlamaConfig {
  const path = `${resolveModelDir(modelDir)}/config.json`;
  return coerce(JSON.parse(Deno.readTextFileSync(path)), path);
}
