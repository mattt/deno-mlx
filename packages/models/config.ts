/**
 * Model config parsing (config.json). v0 targets the Llama family (SmolLM2,
 * Llama 3.x, and close relatives): RMSNorm, RoPE, SwiGLU MLP, GQA.
 */

import { hubFile } from "./hub.ts";

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
}

// deno-lint-ignore no-explicit-any
function coerce(raw: any): LlamaConfig {
  const heads = raw.num_attention_heads;
  const modelType = raw.model_type;
  if (modelType && modelType !== "llama") {
    // Not a hard failure — many models are Llama-shaped — but worth surfacing.
    console.warn(
      `[deno-mlx] model_type "${modelType}" is not "llama"; assuming a ` +
        `Llama-compatible architecture.`,
    );
  }
  return {
    hiddenSize: raw.hidden_size,
    numLayers: raw.num_hidden_layers,
    numHeads: heads,
    numKVHeads: raw.num_key_value_heads ?? heads,
    headDim: raw.head_dim ?? Math.floor(raw.hidden_size / heads),
    intermediateSize: raw.intermediate_size,
    vocabSize: raw.vocab_size,
    rmsNormEps: raw.rms_norm_eps ?? 1e-5,
    ropeTheta: raw.rope_theta ?? 10000,
    tieWordEmbeddings: raw.tie_word_embeddings ?? false,
  };
}

export function loadConfig(repoId: string): LlamaConfig {
  return coerce(JSON.parse(Deno.readTextFileSync(hubFile(repoId, "config.json"))));
}
