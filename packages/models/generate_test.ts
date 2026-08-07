/**
 * Generation runtime tests — async eval, abort, and finish reasons.
 * Gated on SmolLM2 in the HF cache (same as models_test.ts).
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import { loadConfig } from "./config.ts";
import { generate, generateResult } from "./generate.ts";
import { resolveSnapshot } from "./hub.ts";
import { LlamaModel } from "./llama.ts";

const REPO = "HuggingFaceTB/SmolLM2-135M-Instruct";
let present = true;
try {
  resolveSnapshot(REPO);
} catch {
  present = false;
  console.warn(`[skip] ${REPO} not in HF cache — run \`hf download ${REPO}\``);
}
const opts = { sanitizeResources: false, sanitizeOps: false, ignore: !present };

const PROMPT_IDS = [504, 3575, 282, 4649, 314];
const idTok = {
  encode: () => [],
  decode: () => "",
  applyChatTemplate: () => [],
  eosTokenId: undefined as number | undefined,
};

Deno.test("async generate keeps event loop responsive", opts, async () => {
  const cfg = loadConfig(REPO);
  using model = LlamaModel.load(REPO, cfg);
  let beats = 0;
  const timer = setInterval(() => {
    beats++;
  }, 5);
  try {
    const result = await generateResult(model, idTok, PROMPT_IDS, {
      maxTokens: 8,
    });
    assertEquals(result.tokens.length, 8);
    assertEquals(result.finishReason, "length");
    assert(beats > 0, `expected timer ticks during generate, got ${beats}`);
  } finally {
    clearInterval(timer);
  }
});

Deno.test("AbortSignal cancels generation", opts, async () => {
  const cfg = loadConfig(REPO);
  using model = LlamaModel.load(REPO, cfg);
  const ac = new AbortController();
  const tokens: number[] = [];
  for await (
    const t of generate(model, idTok, PROMPT_IDS, {
      maxTokens: 64,
      signal: ac.signal,
    })
  ) {
    if (t.id >= 0) {
      tokens.push(t.id);
      if (tokens.length === 2) ac.abort();
    }
    if (t.finishReason === "cancelled") {
      assert(tokens.length >= 2);
      return;
    }
  }
  assert(false, "expected cancelled finishReason");
});
