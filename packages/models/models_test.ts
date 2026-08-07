/**
 * @deno-mlx/models tests.
 * Deterministic and network-free:
 * they use fixed token ids and assert against values captured from Python `mlx_lm`.
 * Requires the model in the HF cache
 * (`hf download HuggingFaceTB/SmolLM2-135M-Instruct`) and libmlxc.dylib;
 * the suite is skipped if the model is absent.
 */

import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1";
import { loadConfig } from "./config.ts";
import { KVCache, LlamaModel } from "./llama.ts";
import { generate } from "./generate.ts";
import { resolveSnapshot } from "./hub.ts";
import { Weights } from "./weights.ts";
import { hubFile } from "./hub.ts";

const REPO = "HuggingFaceTB/SmolLM2-135M-Instruct";
let present = true;
try {
  resolveSnapshot(REPO);
} catch {
  present = false;
  console.warn(`[skip] ${REPO} not in HF cache — run \`hf download ${REPO}\``);
}
const opts = { sanitizeResources: false, sanitizeOps: false, ignore: !present };

// "The capital of France is" per the model's tokenizer.
const PROMPT_IDS = [504, 3575, 282, 4649, 314];

Deno.test("config parses SmolLM2 (Llama) fields", opts, () => {
  const cfg = loadConfig(REPO);
  assertEquals(cfg.numLayers, 30);
  assertEquals(cfg.numHeads, 9);
  assertEquals(cfg.numKVHeads, 3);
  assertEquals(cfg.headDim, 64);
  assert(cfg.tieWordEmbeddings);
  assert(cfg.maxPositionEmbeddings > 0);
  assertEquals(cfg.modelType, "llama");
});

Deno.test("weights load with expected shapes / tied embeddings", opts, () => {
  using w = Weights.load(hubFile(REPO, "model.safetensors"));
  assert(w.has("model.embed_tokens.weight"));
  assert(!w.has("lm_head.weight"), "SmolLM2 ties embeddings");
  assertEquals(w.get("model.embed_tokens.weight").shape, [49152, 576]);
  assertEquals(w.get("model.layers.0.self_attn.q_proj.weight").shape, [576, 576]);
});

Deno.test("forward pass matches mlx_lm logits", opts, () => {
  const cfg = loadConfig(REPO);
  using model = LlamaModel.load(REPO, cfg);
  using cache = new KVCache(cfg.numLayers);
  using logits = model.forward(PROMPT_IDS, cache);
  assertEquals(model.argmaxId(logits), 7042); // " Paris"
  assertAlmostEquals(logits.toFloat32Array()[7042], 16.125, 0.2);
  assertEquals(cache.length, 5);
});

Deno.test("greedy KV-cache decode matches mlx_lm", opts, async () => {
  const cfg = loadConfig(REPO);
  using model = LlamaModel.load(REPO, cfg);
  // id-only tokenizer stub: exercises the real generate() loop without network.
  const idTok = {
    encode: () => [],
    decode: () => "",
    applyChatTemplate: () => [],
    eosTokenId: undefined,
  };
  const ids: number[] = [];
  for await (const t of generate(model, idTok, PROMPT_IDS, { maxTokens: 12 })) {
    if (t.id >= 0) ids.push(t.id);
  }
  assertEquals(ids, [7042, 30, 7042, 314, 260, 3995, 2240, 281, 4649, 284, 260, 3575]);
});
