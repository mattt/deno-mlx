/**
 * @deno-mlx/core — Layer 1: raw FFI bindings over mlx-c.
 *
 * Status: scaffold. M0 (FFI de-risk spike) is proven in ../../spike.
 * M1 replaces the hand-written spike bindings with a generated symbol table
 * (see ./codegen) and a curated exported surface.
 */
export { resolveMlxcPath } from "./resolver.ts";

/** A single-pointer opaque mlx-c handle, as seen by Deno FFI. */
export const HANDLE = { struct: ["pointer"] } as const;
