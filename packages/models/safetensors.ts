/**
 * Resolve safetensors weight paths for a model directory.
 *
 * Supports a single `model.safetensors` file or a Hugging Face shard index
 * (`model.safetensors.index.json` + `model-XXXXX-of-YYYYY.safetensors`).
 */

export interface WeightFiles {
  /** Absolute paths to load, in order. */
  paths: string[];
  /** True when an index.json shard layout was used. */
  sharded: boolean;
}

/**
 * Locate weight files under `modelDir`.
 * Throws with an actionable message when neither a single file nor a complete shard set is present.
 */
export function resolveWeightFiles(modelDir: string): WeightFiles {
  const single = `${modelDir}/model.safetensors`;
  if (exists(single)) return { paths: [single], sharded: false };

  const indexPath = `${modelDir}/model.safetensors.index.json`;
  if (!exists(indexPath)) {
    throw new Error(
      `No model.safetensors or model.safetensors.index.json in ${modelDir}`,
    );
  }

  const index = JSON.parse(Deno.readTextFileSync(indexPath)) as {
    weight_map?: Record<string, string>;
  };
  const map = index.weight_map;
  if (!map || typeof map !== "object") {
    throw new Error(`Invalid safetensors index (missing weight_map): ${indexPath}`);
  }

  const files = [...new Set(Object.values(map))];
  files.sort();
  const paths: string[] = [];
  const missing: string[] = [];
  for (const file of files) {
    const path = `${modelDir}/${file}`;
    if (exists(path)) paths.push(path);
    else missing.push(file);
  }
  if (missing.length > 0) {
    throw new Error(
      `Incomplete sharded safetensors in ${modelDir}. Missing: ${
        missing.slice(0, 5).join(", ")
      }${missing.length > 5 ? ` (+${missing.length - 5} more)` : ""}`,
    );
  }
  return { paths, sharded: true };
}

function exists(path: string): boolean {
  try {
    return Deno.statSync(path).isFile;
  } catch {
    return false;
  }
}
