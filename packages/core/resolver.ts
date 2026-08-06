/**
 * dylib resolver for mlx-c.
 *
 * Resolution order (the crux of the single-binary story — a `deno compile`d
 * binary carries no native code, so the dylib must be found at runtime):
 *
 *   1. DENO_MLX_DYLIB          explicit override (also used by CI / tests)
 *   2. Homebrew / system paths  `brew install mlx-c`
 *   3. vendored / downloaded    pinned prebuilt beside the binary  (M1)
 *
 * M0 implements 1 + 2; the vendored/download path lands in M1.
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

  throw new Error(
    "Could not locate libmlxc.dylib. Install it with `brew install mlx-c`, " +
      "or set DENO_MLX_DYLIB to its path.",
  );
}

function canStat(path: string): boolean {
  try {
    Deno.statSync(path);
    return true;
  } catch (err) {
    if (err instanceof Deno.errors.NotFound) return false;
    // A permission error is NOT absence — surface it, don't mask it as
    // "dylib missing" (that misdirected the M0 compile probe).
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
