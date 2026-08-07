/**
 * Tokenization via transformers.js (pure JS) —
 * no second native dependency.
 * Loads the tokenizer for a Hub repo id or a local model directory.
 */

import { AutoTokenizer } from "@huggingface/transformers";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatTemplateOptions {
  /** Default true — append the generation prompt suffix. */
  addGenerationPrompt?: boolean;
}

export interface Tokenizer {
  encode(text: string): number[];
  decode(ids: number[], skipSpecial?: boolean): string;
  /** Render chat messages to token ids. */
  applyChatTemplate(
    messages: ChatMessage[],
    opts?: ChatTemplateOptions,
  ): number[];
  readonly eosTokenId: number | undefined;
}

export async function loadTokenizer(repoId: string): Promise<Tokenizer> {
  const tk = await AutoTokenizer.from_pretrained(repoId);
  return wrap(tk);
}

/** Load a tokenizer from a local model directory (tokenizer.json present). */
export async function loadTokenizerFromDir(modelDir: string): Promise<Tokenizer> {
  const tk = await AutoTokenizer.from_pretrained(modelDir, {
    local_files_only: true,
  });
  return wrap(tk);
}

// deno-lint-ignore no-explicit-any
function wrap(tk: any): Tokenizer {
  return {
    encode: (text) => toIds(tk.encode(text)),
    decode: (ids, skipSpecial = true) =>
      tk.decode(ids, { skip_special_tokens: skipSpecial }),
    applyChatTemplate: (messages, opts) =>
      toIds(
        tk.apply_chat_template(messages, {
          tokenize: true,
          add_generation_prompt: opts?.addGenerationPrompt ?? true,
        }),
      ),
    eosTokenId: tk.eos_token_id ?? tk.config?.eos_token_id,
  };
}

/**
 * Normalize transformers.js token output to number[].
 * `encode` returns a plain array,
 * but `apply_chat_template({tokenize:true})` returns a Tensor whose `.data` is a BigInt64Array —
 * flatten and coerce either shape.
 */
// deno-lint-ignore no-explicit-any
function toIds(x: any): number[] {
  if (Array.isArray(x)) return x.map(Number);
  if (x?.data) return Array.from(x.data as ArrayLike<bigint | number>, Number);
  return Array.from(x as Iterable<bigint | number>, Number);
}
