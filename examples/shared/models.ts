/**
 * Shared model loading helpers for examples.
 */

import {
  type Embedder,
  type LoadedModel,
  loadEmbedder,
  loadEmbedderFromDir,
  loadModel,
  loadModelFromDir,
} from "@deno-mlx/models";

export const DEFAULT_CHAT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct";
export const DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2";

/** App-support directory for downloaded models (desktop / CLI). */
export function appModelsDir(): string {
  const home = Deno.env.get("HOME") ?? ".";
  return `${home}/Library/Application Support/deno-mlx/models`;
}

export function resolveChatSource(): { kind: "repo" | "dir"; value: string } {
  const dir = Deno.env.get("DENO_MLX_MODEL_DIR");
  if (dir) return { kind: "dir", value: dir };
  return {
    kind: "repo",
    value: Deno.env.get("DENO_MLX_MODEL") ?? DEFAULT_CHAT_MODEL,
  };
}

export function resolveEmbedSource(): { kind: "repo" | "dir"; value: string } {
  const dir = Deno.env.get("DENO_MLX_EMBED_DIR");
  if (dir) return { kind: "dir", value: dir };
  return {
    kind: "repo",
    value: Deno.env.get("DENO_MLX_EMBED_MODEL") ?? DEFAULT_EMBED_MODEL,
  };
}

export async function loadChatModel(): Promise<LoadedModel> {
  const src = resolveChatSource();
  return src.kind === "dir"
    ? await loadModelFromDir(src.value)
    : await loadModel(src.value);
}

export async function loadEmbedModel(): Promise<Embedder> {
  const src = resolveEmbedSource();
  return src.kind === "dir"
    ? await loadEmbedderFromDir(src.value)
    : await loadEmbedder(src.value);
}

export function labelOf(src: { kind: string; value: string }): string {
  return src.value;
}
