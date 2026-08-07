/**
 * High-level text embeddings: load a BERT-family encoder + its tokenizer,
 * embed strings to normalized vectors,
 * and compare them.
 * Mirrors the sentence-transformers recipe (mean pooling + L2 normalize).
 */

import { type BertConfig, BertModel, loadBertConfigFromDir } from "./bert.ts";
import { InferenceLock } from "./inference_lock.ts";
import { loadTokenizer, loadTokenizerFromDir, type Tokenizer } from "./tokenizer.ts";
import { resolveModelDir, resolveSnapshot } from "./hub.ts";

export interface EmbedOptions {
  /** Truncate to this many tokens (default: model max_position_embeddings). */
  maxLength?: number;
  /** When true (default), error if input exceeds maxLength without truncation. */
  truncate?: boolean;
  signal?: AbortSignal;
}

export interface Embedder {
  /** Embedding dimensionality. */
  readonly dimension: number;
  readonly cfg: BertConfig;
  /** Embed one string to a normalized Float32Array. */
  embed(text: string, opts?: EmbedOptions): Promise<Float32Array>;
  /** Embed many strings sequentially (serialized on this embedder). */
  embedAll(texts: string[], opts?: EmbedOptions): Promise<Float32Array[]>;
  [Symbol.dispose](): void;
}

export function loadEmbedder(repoId: string): Promise<Embedder> {
  return loadEmbedderFromDir(resolveSnapshot(repoId), repoId);
}

export async function loadEmbedderFromDir(
  modelDir: string,
  tokenizerSource?: string,
): Promise<Embedder> {
  const dir = resolveModelDir(modelDir);
  const cfg = loadBertConfigFromDir(dir);
  const model = BertModel.loadDir(dir, cfg);
  const tokenizer: Tokenizer = tokenizerSource
    ? await loadTokenizer(tokenizerSource)
    : await loadTokenizerFromDir(dir);
  return createEmbedder(model, tokenizer, cfg);
}

function createEmbedder(
  model: BertModel,
  tokenizer: Tokenizer,
  cfg: BertConfig,
): Embedder {
  const lock = new InferenceLock();
  let disposed = false;

  const embedOne = (
    text: string,
    opts: EmbedOptions = {},
  ): Promise<Float32Array> => {
    if (disposed) throw new Error("Embedder used after dispose");
    return lock.run(async () => {
      let ids = tokenizer.encode(text);
      const maxLen = opts.maxLength ?? cfg.maxPositionEmbeddings;
      if (ids.length > maxLen) {
        if (opts.truncate) ids = ids.slice(0, maxLen);
        else {
          throw new Error(
            `Embedding input length ${ids.length} exceeds maxLength ${maxLen}. ` +
              `Pass { truncate: true } to truncate.`,
          );
        }
      }
      using vec = model.embed(ids);
      return await vec.toFloat32ArrayAsync();
    }, opts.signal);
  };

  return {
    cfg,
    dimension: cfg.hiddenSize,
    embed: embedOne,
    embedAll: async (texts, opts) => {
      const out: Float32Array[] = [];
      for (const t of texts) out.push(await embedOne(t, opts));
      return out;
    },
    [Symbol.dispose]() {
      if (disposed) return;
      disposed = true;
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
