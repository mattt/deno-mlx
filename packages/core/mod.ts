/**
 * @deno-mlx/core — Layer 1: raw FFI bindings over mlx-c.
 *
 * The full symbol surface (596 functions) is generated from the mlx-c headers
 * (see ./codegen) and pinned to the mlx-c version in ./generated/meta.ts. Open
 * the library with {@link openMlxc}; `.raw` gives direct FFI symbols and
 * `.checked` turns mlx-c status codes into thrown {@link MlxError}s.
 *
 * Layer 2 (`@deno-mlx/tensor`) builds an idiomatic `Tensor` on top of this.
 */

export { MlxError, openMlxc } from "./ffi.ts";
export type { Mlxc } from "./ffi.ts";
export { resolveMlxcPath } from "./resolver.ts";

// Generated enums (Dtype, DeviceType, CompileMode, FftNorm) and the handle shape.
export * from "./generated/types.ts";
// Metadata: the pinned mlx-c version, status-returning set, and skipped surface.
export { mlxcVersion, skipped, statusReturning } from "./generated/meta.ts";
