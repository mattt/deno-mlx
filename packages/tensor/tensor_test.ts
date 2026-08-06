/**
 * @deno-mlx/tensor tests — introspection, ops, async eval, and the memory model
 * (tidy + using). Requires libmlxc.dylib (`brew install mlx-c`).
 */

import {
  assert,
  assertAlmostEquals,
  assertEquals,
  assertThrows,
} from "jsr:@std/assert@^1";
import { Dtype, Tensor, tidy } from "./mod.ts";

const opts = { sanitizeResources: false, sanitizeOps: false };
const arr = (t: Tensor) => Array.from(t.toFloat32Array());

Deno.test("introspection: shape, ndim, size, dtype", opts, () => {
  using t = Tensor.fromFloat32([1, 2, 3, 4, 5, 6], [2, 3]);
  assertEquals(t.shape, [2, 3]);
  assertEquals(t.ndim, 2);
  assertEquals(t.size, 6);
  assertEquals(t.dtype, Dtype.MLX_FLOAT32);
});

Deno.test("fromFloat32 rejects shape/length mismatch", opts, () => {
  assertThrows(() => Tensor.fromFloat32([1, 2, 3], [2, 2]));
});

Deno.test("chained ops + async eval", opts, async () => {
  using a = Tensor.fromFloat32([1, 2, 3, 4], [2, 2]);
  using id = Tensor.fromFloat32([1, 0, 0, 1], [2, 2]);
  using c = a.matmul(id).add(a); // = a + a
  await c.eval();
  assertEquals(c.shape, [2, 2]);
  arr(c).forEach((v, i) => assertAlmostEquals(v, [2, 4, 6, 8][i]));
});

Deno.test("reshape and astype", opts, () => {
  using t = Tensor.fromFloat32([1.7, 2.9, 3.2, 4.8, 5.0, 6.1], [2, 3]);
  using r = t.reshape([3, 2]);
  assertEquals(r.shape, [3, 2]);

  using asInt = t.astype(Dtype.MLX_INT32);
  assertEquals(asInt.dtype, Dtype.MLX_INT32);
  assertThrows(() => asInt.toFloat32Array(), Error, "float32"); // guarded
  using back = asInt.astype(Dtype.MLX_FLOAT32);
  assertEquals(arr(back), [1, 2, 3, 4, 5, 6]); // truncated toward zero
});

Deno.test("tidy frees intermediates but keeps the result", opts, () => {
  using a = Tensor.fromFloat32([1, 2, 3, 4], [4]);
  let intermediate: Tensor | undefined;
  const result = tidy(() => {
    const step1 = a.add(a); // intermediate -> should be freed
    intermediate = step1;
    return step1.multiply(a); // returned -> kept
  });
  assert(intermediate!.disposed, "tidy should free the intermediate");
  assert(!result.disposed, "tidy should keep the returned tensor");
  assert(!a.disposed, "tidy must not free tensors created outside it");
  assertEquals(arr(result), [2, 8, 18, 32]); // (a+a)*a
  result[Symbol.dispose]();
});

Deno.test("tidy frees everything when fn throws", opts, () => {
  using a = Tensor.fromFloat32([1, 2], [2]);
  let leaked: Tensor | undefined;
  assertThrows(() =>
    tidy(() => {
      leaked = a.add(a);
      throw new Error("boom");
    })
  );
  assert(leaked!.disposed, "tidy should free tensors created before a throw");
});

Deno.test("using disposes deterministically; use-after-dispose throws", opts, () => {
  let escaped: Tensor;
  {
    using t = Tensor.fromFloat32([1], [1]);
    escaped = t;
    assert(!t.disposed);
  }
  assert(escaped.disposed, "using should dispose at block exit");
  assertThrows(() => escaped.handle, Error, "after dispose");
});

Deno.test("dispose is idempotent", opts, () => {
  const t = Tensor.fromFloat32([1], [1]);
  t[Symbol.dispose]();
  t[Symbol.dispose](); // no throw / no double free
  assert(t.disposed);
});
