/**
 * Llama-family forward pass (SmolLM2, Llama 3.x, and close relatives) built from
 * mlx-c fast ops. Batch size is fixed at 1.
 *
 * Memory: each layer runs inside a `tidy` that keeps only [hidden, k, v]; the
 * ~15 intermediates it creates are freed immediately. The KV cache tensors live
 * across calls and are disposed explicitly when replaced.
 */

import { Tensor, tidy } from "@deno-mlx/tensor";
import { Dtype } from "@deno-mlx/core";
import type { LlamaConfig } from "./config.ts";
import { hubFile } from "./hub.ts";
import { Weights } from "./weights.ts";
import {
  argmax,
  concat,
  embedding,
  int32,
  itemU32,
  linear,
  rmsNorm,
  rope,
  sdpa,
  silu,
  transpose,
} from "./ops.ts";

interface LayerWeights {
  inputNorm: Tensor;
  q: Tensor;
  k: Tensor;
  v: Tensor;
  o: Tensor;
  postNorm: Tensor;
  gate: Tensor;
  up: Tensor;
  down: Tensor;
}

interface KV {
  k: Tensor;
  v: Tensor;
}

/** Per-request key/value cache; one entry per layer, grown along the seq axis. */
export class KVCache {
  readonly layers: (KV | undefined)[];
  constructor(numLayers: number) {
    this.layers = new Array(numLayers).fill(undefined);
  }
  /** Number of tokens already cached (0 before the first forward). */
  get length(): number {
    const first = this.layers[0];
    return first ? first.k.shape[2] : 0;
  }
  [Symbol.dispose](): void {
    for (const l of this.layers) {
      l?.k[Symbol.dispose]();
      l?.v[Symbol.dispose]();
    }
  }
}

export class LlamaModel {
  #embed: Tensor;
  #norm: Tensor;
  #layers: LayerWeights[];
  readonly cfg: LlamaConfig;

  private constructor(
    cfg: LlamaConfig,
    embed: Tensor,
    norm: Tensor,
    layers: LayerWeights[],
  ) {
    this.cfg = cfg;
    this.#embed = embed;
    this.#norm = norm;
    this.#layers = layers;
  }

  static load(repoId: string, cfg: LlamaConfig): LlamaModel {
    using w = Weights.load(hubFile(repoId, "model.safetensors"));
    const embed = w.get("model.embed_tokens.weight");
    const norm = w.get("model.norm.weight");
    const layers: LayerWeights[] = [];
    for (let i = 0; i < cfg.numLayers; i++) {
      const p = `model.layers.${i}`;
      layers.push({
        inputNorm: w.get(`${p}.input_layernorm.weight`),
        q: w.get(`${p}.self_attn.q_proj.weight`),
        k: w.get(`${p}.self_attn.k_proj.weight`),
        v: w.get(`${p}.self_attn.v_proj.weight`),
        o: w.get(`${p}.self_attn.o_proj.weight`),
        postNorm: w.get(`${p}.post_attention_layernorm.weight`),
        gate: w.get(`${p}.mlp.gate_proj.weight`),
        up: w.get(`${p}.mlp.up_proj.weight`),
        down: w.get(`${p}.mlp.down_proj.weight`),
      });
    }
    return new LlamaModel(cfg, embed, norm, layers);
  }

  /**
   * Run the transformer over `tokenIds`, updating `cache`, and return the
   * logits for the final position as a [vocab] float32 tensor (caller owns it).
   */
  forward(tokenIds: number[], cache: KVCache): Tensor {
    const { numHeads, numKVHeads, headDim, hiddenSize, ropeTheta, rmsNormEps } = this.cfg;
    const seq = tokenIds.length;
    const offset = cache.length;
    const scale = 1 / Math.sqrt(headDim);

    using ids = int32(tokenIds, [seq]);
    // [seq, hidden] -> [1, seq, hidden]
    let h = embedding(this.#embed, ids).reshape([1, seq, hiddenSize]);

    for (let i = 0; i < this.#layers.length; i++) {
      const lw = this.#layers[i];
      const past = cache.layers[i];

      const [hNext, kFull, vFull] = tidy(() => {
        const hn = rmsNorm(h, lw.inputNorm, rmsNormEps);

        // project -> [1, seq, heads, headDim] -> [1, heads, seq, headDim]
        const toHeads = (t: Tensor, heads: number) =>
          transpose(t.reshape([1, seq, heads, headDim]), [0, 2, 1, 3]);
        let q = toHeads(linear(hn, lw.q), numHeads);
        let k = toHeads(linear(hn, lw.k), numKVHeads);
        const v = toHeads(linear(hn, lw.v), numKVHeads);

        q = rope(q, headDim, ropeTheta, offset);
        k = rope(k, headDim, ropeTheta, offset);
        const kCat = past ? concat([past.k, k], 2) : k;
        const vCat = past ? concat([past.v, v], 2) : v;

        const attn = sdpa(q, kCat, vCat, scale, "causal");
        // [1, heads, seq, headDim] -> [1, seq, hidden]
        const merged = transpose(attn, [0, 2, 1, 3]).reshape([1, seq, hiddenSize]);
        const h1 = h.add(linear(merged, lw.o));

        const hn2 = rmsNorm(h1, lw.postNorm, rmsNormEps);
        const act = silu(linear(hn2, lw.gate)).multiply(linear(hn2, lw.up));
        const h2 = h1.add(linear(act, lw.down));

        return [h2, kCat, vCat] as [Tensor, Tensor, Tensor];
      });

      past?.k[Symbol.dispose]();
      past?.v[Symbol.dispose]();
      cache.layers[i] = { k: kFull, v: vFull };
      h[Symbol.dispose]();
      h = hNext;
    }

    // final norm + tied lm_head on the LAST position only
    const logits = tidy(() => {
      using hn = rmsNorm(h, this.#norm, rmsNormEps);
      using last = lastPosition(hn, seq, hiddenSize); // [1, hidden]
      // tied embeddings: lm_head weight == embed_tokens weight [vocab, hidden]
      return linear(last, this.#embed).reshape([this.cfg.vocabSize]).astype(
        Dtype.MLX_FLOAT32,
      );
    });
    h[Symbol.dispose]();
    return logits;
  }

  /** Greedy next-token id from a logits tensor. */
  argmaxId(logits: Tensor): number {
    using idx = argmax(logits);
    return itemU32(idx);
  }

  [Symbol.dispose](): void {
    this.#embed[Symbol.dispose]();
    this.#norm[Symbol.dispose]();
    for (const l of this.#layers) {
      for (const t of Object.values(l)) t[Symbol.dispose]();
    }
  }
}

/** Slice the last sequence position: [1, seq, hidden] -> [1, hidden]. */
function lastPosition(h: Tensor, seq: number, hidden: number): Tensor {
  using flat = h.reshape([seq, hidden]);
  using idx = int32([seq - 1], [1]);
  return embedding(flat, idx).reshape([1, hidden]); // gather row seq-1
}
