/**
 * dylib resolver for mlx-c.
 *
 * Resolution order
 * (a compiled binary carries no native code, so the dylib must be found at runtime):
 *
 *   1. DENO_MLX_DYLIB          explicit override (also used by CI / tests)
 *   2. Homebrew / system paths  `brew install mlx-c`
 *   3. Vendored layouts relative to the executable:
 *        {execDir}/libmlxc.dylib
 *        {execDir}/vendor/libmlxc.dylib
 *        {execDir}/../lib/libmlxc.dylib          (CLI archive layout)
 *        {execDir}/../Frameworks/libmlxc.dylib   (macOS .app layout)
 */

const HOMEBREW_CANDIDATES = [
  "/opt/homebrew/opt/mlx-c/lib/libmlxc.dylib", // Apple Silicon
  "/usr/local/opt/mlx-c/lib/libmlxc.dylib", // Intel Homebrew
];

/** Locate libmlxc.dylib, or throw with an actionable message. */
export function resolveMlxcPath(): string {
  const override = Deno.env.get("DENO_MLX_DYLIB");
  if (override) {
    if (!canStat(override)) {
      throw new Error(`DENO_MLX_DYLIB is set but not found: ${override}`);
    }
    return override;
  }

  for (const candidate of HOMEBREW_CANDIDATES) {
    if (canStat(candidate)) return candidate;
  }

  for (const candidate of vendoredCandidates()) {
    if (canStat(candidate)) return candidate;
  }

  throw new Error(
    "mlx-c is not installed (libmlxc.dylib not found). Install with " +
      "`brew install mlx-c`, set DENO_MLX_DYLIB=/path/to/libmlxc.dylib, or " +
      "place libmlxc.dylib beside the executable, in ./vendor/, ../lib/, or " +
      "../Frameworks/.",
  );
}

/**
 * Locations checked for a vendored dylib, relative to the running executable.
 */
function vendoredCandidates(): string[] {
  let dir: string;
  try {
    dir = dirname(Deno.execPath());
  } catch {
    return [];
  }
  const parent = dirname(dir);
  return [
    `${dir}/libmlxc.dylib`,
    `${dir}/vendor/libmlxc.dylib`,
    `${parent}/lib/libmlxc.dylib`,
    `${parent}/Frameworks/libmlxc.dylib`,
  ];
}

function dirname(path: string): string {
  const i = path.lastIndexOf("/");
  return i <= 0 ? "." : path.slice(0, i);
}

function canStat(path: string): boolean {
  try {
    Deno.statSync(path);
    return true;
  } catch (err) {
    if (err instanceof Deno.errors.NotFound) return false;
    if (err instanceof Deno.errors.PermissionDenied) {
      throw new Error(
        `Cannot read ${path}: missing --allow-read for the mlx-c dylib. ` +
          `Grant read access (e.g. --allow-read=/opt/homebrew) or set DENO_MLX_DYLIB.`,
        { cause: err },
      );
    }
    return false;
  }
}
