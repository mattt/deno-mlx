/**
 * High-level text embeddings: load a BERT-family encoder + its tokenizer, embed
 * strings to normalized vectors, and compare them. Mirrors the sentence-
 * transformers recipe (mean pooling + L2 normalize).
 */

import { BertConfig, BertModel, loadBertConfig } from "./bert.ts";
import { loadTokenizer, type Tokenizer } from "./tokenizer.ts";

export interface Embedder {
  /** Embed one string to a normalized Float32Array of length hiddenSize. */
  embed(text: string): Float32Array;
  /** Embed many strings. */
  embedAll(texts: string[]): Float32Array[];
  readonly cfg: BertConfig;
  [Symbol.dispose](): void;
}

export async function loadEmbedder(repoId: string): Promise<Embedder> {
  const cfg = loadBertConfig(repoId);
  const model = BertModel.load(repoId, cfg);
  const tokenizer: Tokenizer = await loadTokenizer(repoId);

  const embed = (text: string): Float32Array => {
    const ids = tokenizer.encode(text); // adds [CLS] … [SEP]
    using vec = model.embed(ids);
    return vec.toFloat32Array();
  };

  return {
    cfg,
    embed,
    embedAll: (texts) => texts.map(embed),
    [Symbol.dispose]() {
      model[Symbol.dispose]();
    },
  };
}

/** Cosine similarity of two equal-length vectors (unit vectors -> dot product). */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}
