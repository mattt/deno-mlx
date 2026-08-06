/**
 * @deno-mlx/tensor — Layer 2: safe, idiomatic tensors with lifetime management.
 *
 * ```ts
 * import { Tensor, tidy } from "@deno-mlx/tensor";
 *
 * using a = Tensor.fromFloat32([1, 2, 3, 4], [2, 2]);
 * using b = Tensor.fromFloat32([1, 0, 0, 1], [2, 2]);
 * const c = tidy(() => a.matmul(b).add(a));   // intermediates freed
 * console.log((await c.eval()).toFloat32Array());
 * c[Symbol.dispose]();
 * ```
 */

export { Tensor, tidy } from "./tensor.ts";
// Re-export dtype enum so callers don't need to reach into core for it.
export { Dtype } from "@deno-mlx/core";
