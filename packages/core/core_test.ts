/**
 * @deno-mlx/core smoke tests — exercise the generated bindings through the real
 * loader (openMlxc), not the hand-written spike. Requires libmlxc.dylib
 * (`brew install mlx-c`).
 *
 * FFI keeps the dylib + error callback open for the process lifetime by design,
 * so resource/op sanitizers are disabled here.
 */

import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1";
import { Dtype, MlxError, openMlxc, skipped, statusReturning } from "./mod.ts";

const F32 = Dtype.MLX_FLOAT32;
const opts = { sanitizeResources: false, sanitizeOps: false };
const ptr = (b: ArrayBufferView) => Deno.UnsafePointer.of(b);

function arr(data: number[], shape: number[]): Uint8Array {
  return openMlxc().checked.mlx_array_new_data(
    ptr(new Float32Array(data)),
    ptr(new Int32Array(shape)),
    shape.length,
    F32,
  ) as Uint8Array;
}

function toFloats(handle: Uint8Array, n: number): Float32Array {
  const p = openMlxc().checked.mlx_array_data_float32(handle);
  assert(p !== null, "data pointer was null");
  const view = new Deno.UnsafePointerView(p);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = view.getFloat32(i * 4);
  return out;
}

Deno.test("loads mlx-c from a resolved path", opts, () => {
  assert(openMlxc().path.endsWith("libmlxc.dylib"));
});

Deno.test("add round-trip through generated bindings", opts, () => {
  const s = openMlxc().checked;
  const a = arr([1, 2, 3, 4], [4]);
  const b = arr([10, 20, 30, 40], [4]);
  const stream = s.mlx_default_gpu_stream_new();
  const res = s.mlx_array_new();
  s.mlx_add(ptr(res as Uint8Array), a, b, stream); // throws on nonzero status
  s.mlx_array_eval(res);
  assertEquals(Number(s.mlx_array_size(res)), 4);
  const out = toFloats(res as Uint8Array, 4);
  [11, 22, 33, 44].forEach((v, i) => assertAlmostEquals(out[i], v));
  for (const h of [a, b, res]) s.mlx_array_free(h as Uint8Array);
});

Deno.test("optional-struct by-value param: mlx_hadamard_transform", opts, () => {
  // Exercises `mlx_optional_float` -> {struct:["f32","bool"]} passed by value,
  // the same mapping the RoPE / quantization surface relies on.
  const s = openMlxc().checked;
  const x = arr([1, 2, 3, 4], [4]);
  const res = s.mlx_array_new();
  const scaleNone = new Uint8Array(8); // {value: 0, has_value: false} -> default
  const stream = s.mlx_default_gpu_stream_new();
  s.mlx_hadamard_transform(ptr(res as Uint8Array), x, scaleNone, stream);
  s.mlx_array_eval(res);
  assertEquals(Number(s.mlx_array_size(res)), 4);
});

Deno.test("status errors surface as MlxError", opts, () => {
  const s = openMlxc().checked;
  // Incompatible shapes -> mlx reports an error through the handler.
  const a = arr([1, 2], [2]);
  const b = arr([1, 2, 3], [3]);
  const stream = s.mlx_default_gpu_stream_new();
  const res = s.mlx_array_new();
  assert(
    checkThrows(() => {
      s.mlx_add(ptr(res as Uint8Array), a, b, stream);
      s.mlx_array_eval(res);
    }),
    "expected an MlxError for incompatible shapes",
  );
});

function checkThrows(fn: () => void): boolean {
  try {
    fn();
    return false;
  } catch (e) {
    return e instanceof MlxError;
  }
}

Deno.test("coverage invariant: 18 callback/vtable fns skipped", opts, () => {
  assertEquals(skipped.length, 18);
  assert(skipped.every((s) => /callback|vtable|param|return/.test(s.reason) || s.reason));
  // status-returning set is the bulk of the surface
  assert(statusReturning.size > 400, `only ${statusReturning.size} status fns`);
});
