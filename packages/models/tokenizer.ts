/**
 * Tokenization via transformers.js (pure JS) — no second native dependency.
 * Loads the tokenizer for a repo id (transformers.js manages its own
 * download/cache of tokenizer.json).
 */

import { AutoTokenizer } from "@huggingface/transformers";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface Tokenizer {
  encode(text: string): number[];
  decode(ids: number[], skipSpecial?: boolean): string;
  /** Render chat messages to token ids (adds the generation prompt). */
  applyChatTemplate(messages: ChatMessage[]): number[];
  readonly eosTokenId: number | undefined;
}

export async function loadTokenizer(repoId: string): Promise<Tokenizer> {
  const tk = await AutoTokenizer.from_pretrained(repoId);
  return {
    encode: (text) => toIds(tk.encode(text)),
    decode: (ids, skipSpecial = true) =>
      tk.decode(ids, { skip_special_tokens: skipSpecial }),
    applyChatTemplate: (messages) =>
      toIds(
        tk.apply_chat_template(messages, {
          tokenize: true,
          add_generation_prompt: true,
        }),
      ),
    // deno-lint-ignore no-explicit-any
    eosTokenId: (tk as any).eos_token_id ?? (tk as any).config?.eos_token_id,
  };
}

/**
 * Normalize transformers.js token output to number[]. `encode` returns a plain
 * array, but `apply_chat_template({tokenize:true})` returns a Tensor whose
 * `.data` is a BigInt64Array — flatten and coerce either shape.
 */
// deno-lint-ignore no-explicit-any
function toIds(x: any): number[] {
  if (Array.isArray(x)) return x.map(Number);
  if (x?.data) return Array.from(x.data as ArrayLike<bigint | number>, Number);
  return Array.from(x as Iterable<bigint | number>, Number);
}
