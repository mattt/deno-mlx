/**
 * dylib resolver for mlx-c.
 *
 * Resolution order (the crux of the single-binary story — a `deno compile`d
 * binary carries no native code, so the dylib must be found at runtime):
 *
 *   1. DENO_MLX_DYLIB          explicit override (also used by CI / tests)
 *   2. Homebrew / system paths  `brew install mlx-c`
 *   3. vendored                 a copy beside the executable
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

  // 3. vendored beside the executable — this is what lets a `deno compile`d app
  //    ship the dylib next to its binary and run on a machine without Homebrew.
  for (const candidate of vendoredCandidates()) {
    if (canStat(candidate)) return candidate;
  }

  throw new Error(
    "mlx-c is not installed (libmlxc.dylib not found). Install with " +
      "`brew install mlx-c`, set DENO_MLX_DYLIB=/path/to/libmlxc.dylib, or " +
      "place libmlxc.dylib beside the executable (or in ./vendor/).",
  );
}

/**
 * Locations checked for a vendored dylib, relative to the running executable.
 * For `deno compile` apps, drop libmlxc.dylib next to the binary (or in a
 * `vendor/` subdir).
 */
function vendoredCandidates(): string[] {
  let dir: string;
  try {
    dir = dirname(Deno.execPath());
  } catch {
    return [];
  }
  return [
    `${dir}/libmlxc.dylib`,
    `${dir}/vendor/libmlxc.dylib`,
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
    // A permission error is NOT absence — surface it, don't mask it as
    // "dylib missing".
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
