/**
 * Embeddings tests — deterministic and network-free (fixed token ids). Values
 * are captured from a run validated at cosine 1.00000 against transformers.js
 * feature-extraction (mean pool + normalize). Requires the model in the HF
 * cache and libmlxc.dylib; skipped if the model is absent.
 */

import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1";
import { BertModel, loadBertConfig } from "./bert.ts";
import { cosineSimilarity } from "./embeddings.ts";
import { resolveSnapshot } from "./hub.ts";

const REPO = "sentence-transformers/all-MiniLM-L6-v2";
let present = true;
try {
  resolveSnapshot(REPO);
} catch {
  present = false;
  console.warn(`[skip] ${REPO} not in HF cache — run \`hf download ${REPO}\``);
}
const opts = { sanitizeResources: false, sanitizeOps: false, ignore: !present };

// "The cat sits on the mat." per the model's tokenizer (with [CLS]/[SEP]).
const CAT_MAT = [101, 1996, 4937, 7719, 2006, 1996, 13523, 1012, 102];
// "A feline rests on the rug." and "Quantum computing uses qubits."
const FELINE_RUG = [101, 1037, 10768, 4179, 16626, 2006, 1996, 20452, 1012, 102];
const QUANTUM = [101, 8559, 9798, 3594, 24209, 16313, 2015, 1012, 102];

Deno.test("bert config parses all-MiniLM", opts, () => {
  const cfg = loadBertConfig(REPO);
  assertEquals(cfg.numLayers, 6);
  assertEquals(cfg.hiddenSize, 384);
  assertEquals(cfg.numHeads, 12);
});

Deno.test("embedding matches transformers.js reference & is unit norm", opts, () => {
  const cfg = loadBertConfig(REPO);
  using model = BertModel.load(REPO, cfg);
  using v = model.embed(CAT_MAT);
  const arr = v.toFloat32Array();
  assertEquals(arr.length, 384);

  const expectFirst8 = [
    0.134177,
    -0.0329,
    -0.024804,
    0.041439,
    -0.036348,
    0.041814,
    0.031217,
    0.037202,
  ];
  expectFirst8.forEach((x, i) => assertAlmostEquals(arr[i], x, 1e-3));

  let ss = 0;
  for (const x of arr) ss += x * x;
  assertAlmostEquals(Math.sqrt(ss), 1, 1e-4); // L2 normalized
});

Deno.test("semantics: paraphrase closer than unrelated", opts, () => {
  const cfg = loadBertConfig(REPO);
  using model = BertModel.load(REPO, cfg);
  const embed = (ids: number[]) => {
    using v = model.embed(ids);
    return v.toFloat32Array();
  };
  const cat = embed(CAT_MAT);
  const paraphrase = cosineSimilarity(cat, embed(FELINE_RUG));
  const unrelated = cosineSimilarity(cat, embed(QUANTUM));
  assert(
    paraphrase > 0.4 && paraphrase > unrelated + 0.3,
    `paraphrase=${paraphrase.toFixed(3)} unrelated=${unrelated.toFixed(3)}`,
  );
});
