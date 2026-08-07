/**
 * Minimal embeddings example.
 *
 *   deno run -P=mlx --allow-net examples/embeddings.ts
 */

import { cosineSimilarity } from "@deno-mlx/models";
import { loadEmbedModel } from "./shared/models.ts";

using e = await loadEmbedModel();
const a = await e.embed("The cat sits on the mat.");
const b = await e.embed("A feline rests on the rug.");
console.log(`dim=${e.dimension} similarity=${cosineSimilarity(a, b).toFixed(4)}`);
