/**
 * BERT-family encoder for embeddings (all-MiniLM-L6-v2, bge, e5, …).
 *
 * Differs from the Llama decoder:
 * learned position + token-type embeddings,
 * LayerNorm (weight+bias) instead of RMSNorm,
 * bidirectional attention (no causal mask, no RoPE),
 * GELU FFN,
 * biased linears,
 * and mean pooling + L2 normalize to a sentence vector.
 * Batch size 1, no padding (single sequence per call).
 */

import { Tensor, tidy } from "@deno-mlx/tensor";
import { Dtype } from "@deno-mlx/core";
import { resolveModelDir, resolveSnapshot } from "./hub.ts";
import { resolveWeightFiles } from "./safetensors.ts";
import { Weights } from "./weights.ts";
import {
  embedding,
  gelu,
  int32,
  l2Normalize,
  layerNorm,
  linearBias,
  meanAxis,
  sdpa,
  transpose,
} from "./ops.ts";

export interface BertConfig {
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  headDim: number;
  intermediateSize: number;
  vocabSize: number;
  layerNormEps: number;
  maxPositionEmbeddings: number;
  modelType: string;
}

export function loadBertConfig(repoId: string): BertConfig {
  return loadBertConfigFromDir(resolveSnapshot(repoId));
}

export function loadBertConfigFromDir(modelDir: string): BertConfig {
  const path = `${resolveModelDir(modelDir)}/config.json`;
  // deno-lint-ignore no-explicit-any
  const c: any = JSON.parse(Deno.readTextFileSync(path));
  const modelType = String(c.model_type ?? "bert");
  if (modelType !== "bert" && modelType !== "roberta") {
    throw new Error(
      `Unsupported embedding model_type "${modelType}" in ${path}. ` +
        `Supported: bert, roberta.`,
    );
  }
  return {
    hiddenSize: c.hidden_size,
    numLayers: c.num_hidden_layers,
    numHeads: c.num_attention_heads,
    headDim: Math.floor(c.hidden_size / c.num_attention_heads),
    intermediateSize: c.intermediate_size,
    vocabSize: c.vocab_size,
    layerNormEps: c.layer_norm_eps ?? 1e-12,
    maxPositionEmbeddings: Number(c.max_position_embeddings ?? 512),
    modelType,
  };
}

interface BertLayer {
  qW: Tensor;
  qB: Tensor;
  kW: Tensor;
  kB: Tensor;
  vW: Tensor;
  vB: Tensor;
  oW: Tensor;
  oB: Tensor;
  attnNormW: Tensor;
  attnNormB: Tensor;
  interW: Tensor;
  interB: Tensor;
  outW: Tensor;
  outB: Tensor;
  outNormW: Tensor;
  outNormB: Tensor;
}

export class BertModel {
  #wordEmb: Tensor;
  #posEmb: Tensor;
  #typeEmb: Tensor;
  #embNormW: Tensor;
  #embNormB: Tensor;
  #layers: BertLayer[];
  readonly cfg: BertConfig;

  private constructor(cfg: BertConfig, e: {
    wordEmb: Tensor;
    posEmb: Tensor;
    typeEmb: Tensor;
    embNormW: Tensor;
    embNormB: Tensor;
    layers: BertLayer[];
  }) {
    this.cfg = cfg;
    this.#wordEmb = e.wordEmb;
    this.#posEmb = e.posEmb;
    this.#typeEmb = e.typeEmb;
    this.#embNormW = e.embNormW;
    this.#embNormB = e.embNormB;
    this.#layers = e.layers;
  }

  static load(repoId: string, cfg: BertConfig): BertModel {
    return BertModel.loadDir(resolveSnapshot(repoId), cfg);
  }

  static loadDir(modelDir: string, cfg: BertConfig): BertModel {
    const { paths } = resolveWeightFiles(modelDir);
    using w = Weights.loadPaths(paths);
    const g = (n: string) => w.get(n);
    const layers: BertLayer[] = [];
    for (let i = 0; i < cfg.numLayers; i++) {
      const p = `encoder.layer.${i}`;
      layers.push({
        qW: g(`${p}.attention.self.query.weight`),
        qB: g(`${p}.attention.self.query.bias`),
        kW: g(`${p}.attention.self.key.weight`),
        kB: g(`${p}.attention.self.key.bias`),
        vW: g(`${p}.attention.self.value.weight`),
        vB: g(`${p}.attention.self.value.bias`),
        oW: g(`${p}.attention.output.dense.weight`),
        oB: g(`${p}.attention.output.dense.bias`),
        attnNormW: g(`${p}.attention.output.LayerNorm.weight`),
        attnNormB: g(`${p}.attention.output.LayerNorm.bias`),
        interW: g(`${p}.intermediate.dense.weight`),
        interB: g(`${p}.intermediate.dense.bias`),
        outW: g(`${p}.output.dense.weight`),
        outB: g(`${p}.output.dense.bias`),
        outNormW: g(`${p}.output.LayerNorm.weight`),
        outNormB: g(`${p}.output.LayerNorm.bias`),
      });
    }
    return new BertModel(cfg, {
      wordEmb: g("embeddings.word_embeddings.weight"),
      posEmb: g("embeddings.position_embeddings.weight"),
      typeEmb: g("embeddings.token_type_embeddings.weight"),
      embNormW: g("embeddings.LayerNorm.weight"),
      embNormB: g("embeddings.LayerNorm.bias"),
      layers,
    });
  }

  /** Encode token ids to a single normalized [hidden] sentence embedding. */
  embed(tokenIds: number[]): Tensor {
    const { hiddenSize, numHeads, headDim, layerNormEps, maxPositionEmbeddings } =
      this.cfg;
    if (tokenIds.length > maxPositionEmbeddings) {
      throw new Error(
        `Embedding input length ${tokenIds.length} exceeds max_position_embeddings ${maxPositionEmbeddings}`,
      );
    }
    const seq = tokenIds.length;
    const scale = 1 / Math.sqrt(headDim);

    using ids = int32(tokenIds, [seq]);
    using positions = int32(Array.from({ length: seq }, (_, i) => i), [seq]);
    using types = int32(new Array(seq).fill(0), [seq]);

    let h = tidy(() => {
      const word = embedding(this.#wordEmb, ids);
      const pos = embedding(this.#posEmb, positions);
      const typ = embedding(this.#typeEmb, types);
      const sum = word.add(pos).add(typ).reshape([1, seq, hiddenSize]);
      return layerNorm(sum, this.#embNormW, this.#embNormB, layerNormEps);
    });

    for (const lw of this.#layers) {
      const next = tidy(() => {
        const toHeads = (t: Tensor) =>
          transpose(t.reshape([1, seq, numHeads, headDim]), [0, 2, 1, 3]);
        const q = toHeads(linearBias(h, lw.qW, lw.qB));
        const k = toHeads(linearBias(h, lw.kW, lw.kB));
        const v = toHeads(linearBias(h, lw.vW, lw.vB));
        const attn = sdpa(q, k, v, scale, ""); // bidirectional (no mask)
        const merged = transpose(attn, [0, 2, 1, 3]).reshape([1, seq, hiddenSize]);
        const ao = linearBias(merged, lw.oW, lw.oB);
        const h1 = layerNorm(h.add(ao), lw.attnNormW, lw.attnNormB, layerNormEps);

        const inter = gelu(linearBias(h1, lw.interW, lw.interB));
        const out = linearBias(inter, lw.outW, lw.outB);
        return layerNorm(h1.add(out), lw.outNormW, lw.outNormB, layerNormEps);
      });
      h[Symbol.dispose]();
      h = next;
    }

    const pooled = tidy(() => {
      using mean = meanAxis(h, 1).reshape([1, hiddenSize]); // mean over tokens
      return l2Normalize(mean).reshape([hiddenSize]).astype(Dtype.MLX_FLOAT32);
    });
    h[Symbol.dispose]();
    return pooled;
  }

  [Symbol.dispose](): void {
    for (
      const t of [
        this.#wordEmb,
        this.#posEmb,
        this.#typeEmb,
        this.#embNormW,
        this.#embNormB,
      ]
    ) t[Symbol.dispose]();
    for (const l of this.#layers) {
      for (const t of Object.values(l)) t[Symbol.dispose]();
    }
  }
}
