/**
 * @deno-mlx/core loader — open libmlxc with the generated symbol table, install
 * an error handler so failures carry the real MLX message, and expose a
 * `checked` view that turns mlx-c status codes into thrown `MlxError`s.
 */

import { resolveMlxcPath } from "./resolver.ts";
import { symbols } from "./generated/symbols.ts";
import { statusReturning } from "./generated/meta.ts";

/** Error raised when an mlx-c call returns a nonzero status code. */
export class MlxError extends Error {
  constructor(readonly fn: string, readonly code: number, detail?: string) {
    super(
      detail ? `${fn} failed: ${detail}` : `${fn} failed with status ${code}`,
    );
    this.name = "MlxError";
  }
}

// mlx-c reports failure detail through a global error handler; capture the last
// message here so MlxError can include it. (This is the one callback we bind by
// hand — the generated surface intentionally skips callback-taking functions.)
let lastError: string | undefined;

function takeLastError(): string | undefined {
  const e = lastError;
  lastError = undefined;
  return e;
}

const errorCallback = new Deno.UnsafeCallback(
  { parameters: ["pointer", "pointer"], result: "void" },
  (msg: Deno.PointerValue) => {
    lastError = msg ? new Deno.UnsafePointerView(msg).getCString() : undefined;
  },
);

/** The generated symbol table plus the hand-bound error-handler installer. */
const extendedSymbols = {
  ...symbols,
  // void mlx_set_error_handler(mlx_error_handler_func, void* data, void (*dtor)(void*));
  mlx_set_error_handler: {
    parameters: ["function", "pointer", "function"],
    result: "void",
  },
} as const satisfies Deno.ForeignLibraryInterface;

type Symbols = Deno.StaticForeignLibraryInterface<typeof extendedSymbols>;

export interface Mlxc {
  /** The mlx-c dylib path that was loaded. */
  readonly path: string;
  /** Raw FFI symbols — no status checking (use for hot paths you check yourself). */
  readonly raw: Symbols;
  /** Status-returning functions throw `MlxError` on failure; others pass through. */
  readonly checked: Symbols;
  /** Unload the library and dispose the error callback. */
  close(): void;
}

let singleton: Mlxc | undefined;

/** Open libmlxc (idempotent — repeated calls return the same handle). */
export function openMlxc(): Mlxc {
  if (singleton) return singleton;

  const path = resolveMlxcPath();
  const lib = Deno.dlopen(path, extendedSymbols);
  const raw = lib.symbols;

  // Route mlx-c errors through our callback for rich MlxError messages.
  raw.mlx_set_error_handler(errorCallback.pointer, null, null);

  const checked = new Proxy(raw, {
    get(target, prop: string) {
      const value = Reflect.get(target, prop);
      if (typeof value !== "function" || !statusReturning.has(prop)) return value;
      return (...args: unknown[]) => {
        // deno-lint-ignore no-explicit-any
        const rc = (value as any)(...args) as number;
        if (rc !== 0) throw new MlxError(prop, rc, takeLastError());
        return rc;
      };
    },
  }) as Symbols;

  singleton = {
    path,
    raw,
    checked,
    close() {
      lib.close();
      errorCallback.close();
      singleton = undefined;
    },
  };
  return singleton;
}
