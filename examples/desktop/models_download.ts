/**
 * First-launch demo model download for the desktop example.
 * Uses `@huggingface/hub` with Xet-accelerated transfers when available.
 * Writes into a local directory and verifies config.json + weight files are present.
 * Resume-friendly: skips existing files.
 */

import { downloadFile } from "@huggingface/hub";
import { resolveWeightFiles } from "@deno-mlx/models";

interface DemoModel {
  repo: string;
  dest: string;
  /** Optional revision SHA; default "main". */
  revision?: string;
}

/** Ensure each demo model directory has config + weights. */
export async function ensureDemoModels(models: DemoModel[]): Promise<void> {
  for (const m of models) {
    await ensureOne(m);
  }
}

async function ensureOne(m: DemoModel): Promise<void> {
  Deno.mkdirSync(m.dest, { recursive: true });
  const revision = m.revision ?? "main";
  const accessToken = hfToken();

  const configPath = `${m.dest}/config.json`;
  if (!exists(configPath)) {
    const ok = await downloadRepoFile({
      repo: m.repo,
      path: "config.json",
      dest: configPath,
      revision,
      accessToken,
    });
    if (!ok) throw new Error(`missing config.json in ${m.repo}@${revision}`);
  }

  // Prefer single-file weights; fall back to index + shards.
  const single = `${m.dest}/model.safetensors`;
  const index = `${m.dest}/model.safetensors.index.json`;
  if (!exists(single) && !exists(index)) {
    const gotSingle = await downloadRepoFile({
      repo: m.repo,
      path: "model.safetensors",
      dest: single,
      revision,
      accessToken,
    });
    if (!gotSingle) {
      const gotIndex = await downloadRepoFile({
        repo: m.repo,
        path: "model.safetensors.index.json",
        dest: index,
        revision,
        accessToken,
      });
      if (!gotIndex) {
        throw new Error(
          `missing model.safetensors (or index) in ${m.repo}@${revision}`,
        );
      }
      const idx = JSON.parse(Deno.readTextFileSync(index)) as {
        weight_map: Record<string, string>;
      };
      const files = [...new Set(Object.values(idx.weight_map))];
      for (const file of files) {
        const path = `${m.dest}/${file}`;
        if (!exists(path)) {
          const ok = await downloadRepoFile({
            repo: m.repo,
            path: file,
            dest: path,
            revision,
            accessToken,
          });
          if (!ok) {
            throw new Error(`missing shard ${file} in ${m.repo}@${revision}`);
          }
        }
      }
    }
  }

  // tokenizer files used by transformers.js local load
  for (const file of ["tokenizer.json", "tokenizer_config.json"]) {
    const path = `${m.dest}/${file}`;
    if (!exists(path)) {
      await downloadRepoFile({
        repo: m.repo,
        path: file,
        dest: path,
        revision,
        accessToken,
      });
    }
  }

  resolveWeightFiles(m.dest);
  console.log(`[deno-mlx] model ready: ${m.repo} -> ${m.dest}`);
}

/**
 * Download one Hub file via huggingface.js (Xet when the file is Xet-backed).
 * Returns false when the path does not exist in the repo.
 */
async function downloadRepoFile(opts: {
  repo: string;
  path: string;
  dest: string;
  revision: string;
  accessToken?: string;
}): Promise<boolean> {
  console.log(
    `[deno-mlx] downloading ${opts.repo}@${opts.revision}:${opts.path}`,
  );
  const blob = await downloadFile({
    repo: opts.repo,
    path: opts.path,
    revision: opts.revision,
    accessToken: opts.accessToken,
    xet: true,
  });
  if (!blob) return false;

  const tmp = `${opts.dest}.partial`;
  await Deno.mkdir(dirname(opts.dest), { recursive: true });
  using file = await Deno.open(tmp, {
    write: true,
    create: true,
    truncate: true,
  });
  for await (const chunk of blob.stream()) {
    await file.write(chunk);
  }
  Deno.renameSync(tmp, opts.dest);
  return true;
}

function hfToken(): string | undefined {
  return Deno.env.get("HF_TOKEN") ?? Deno.env.get("HUGGING_FACE_HUB_TOKEN") ??
    undefined;
}

function dirname(path: string): string {
  const i = path.lastIndexOf("/");
  return i <= 0 ? "." : path.slice(0, i);
}

function exists(path: string): boolean {
  try {
    return Deno.statSync(path).isFile;
  } catch {
    return false;
  }
}
