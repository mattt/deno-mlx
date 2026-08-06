/**
 * Hugging Face Hub cache resolution.
 *
 * Reads the same on-disk layout `huggingface-cli`/`hf download` writes
 * (`$HF_HOME/hub/models--org--repo/…`), so this library and the HF tools share
 * one copy of every model — no second download, no separate store.
 */

/** Root of the HF hub cache (honours HF_HOME, then HF_HUB_CACHE, else ~/.cache). */
export function hubCacheDir(): string {
  const explicit = Deno.env.get("HF_HUB_CACHE");
  if (explicit) return explicit;
  const home = Deno.env.get("HF_HOME");
  if (home) return `${home}/hub`;
  const userHome = Deno.env.get("HOME") ?? ".";
  return `${userHome}/.cache/huggingface/hub`;
}

/** Absolute path to a repo's active snapshot dir, or throw if not cached. */
export function resolveSnapshot(repoId: string): string {
  const repoDir = `${hubCacheDir()}/models--${repoId.replaceAll("/", "--")}`;
  const rev = readMainRef(repoDir) ?? newestSnapshot(repoDir);
  if (!rev) {
    throw new Error(
      `Model not found in HF cache: ${repoId}. Download it with ` +
        `\`hf download ${repoId}\`.`,
    );
  }
  return `${repoDir}/snapshots/${rev}`;
}

/** Absolute path to a single file within a repo's snapshot. */
export function hubFile(repoId: string, file: string): string {
  return `${resolveSnapshot(repoId)}/${file}`;
}

function readMainRef(repoDir: string): string | null {
  try {
    return Deno.readTextFileSync(`${repoDir}/refs/main`).trim();
  } catch {
    return null;
  }
}

function newestSnapshot(repoDir: string): string | null {
  try {
    const dirs = [...Deno.readDirSync(`${repoDir}/snapshots`)]
      .filter((e) => e.isDirectory)
      .map((e) => e.name);
    return dirs[0] ?? null;
  } catch {
    return null;
  }
}
