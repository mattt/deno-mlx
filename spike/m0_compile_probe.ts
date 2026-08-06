/**
 * M0 — `deno compile` + dylib-resolution probe.
 *
 * A compiled Deno binary embeds no native code, so this confirms the resulting
 * single binary can (a) run the resolver and (b) dlopen libmlxc.dylib and do a
 * real op at runtime. This is the mechanic behind the single-binary pitch.
 *
 * Build: deno compile --allow-ffi --allow-env --unstable-ffi \
 *          -o /tmp/mlx-probe spike/m0_compile_probe.ts
 * Run:   /tmp/mlx-probe
 */
import { HANDLE, resolveMlxcPath } from "../packages/core/mod.ts";

const path = resolveMlxcPath();
const lib = Deno.dlopen(path, {
  mlx_array_new_data: {
    parameters: ["pointer", "pointer", "i32", "i32"],
    result: HANDLE,
  },
  mlx_default_gpu_stream_new: { parameters: [], result: HANDLE },
  mlx_add: { parameters: ["pointer", HANDLE, HANDLE, HANDLE], result: "i32" },
  mlx_array_new: { parameters: [], result: HANDLE },
  mlx_array_eval: { parameters: [HANDLE], result: "i32" },
  mlx_array_data_float32: { parameters: [HANDLE], result: "pointer" },
  mlx_array_free: { parameters: [HANDLE], result: "i32" },
});
const s = lib.symbols;

const data = new Float32Array([2, 4]);
const shape = new Int32Array([2]);
const mk = () =>
  s.mlx_array_new_data(
    Deno.UnsafePointer.of(data),
    Deno.UnsafePointer.of(shape),
    1,
    10,
  ) as Uint8Array;

const a = mk(), b = mk(), r = s.mlx_array_new() as Uint8Array;
s.mlx_add(Deno.UnsafePointer.of(r), a, b, s.mlx_default_gpu_stream_new());
s.mlx_array_eval(r);
const view = new Deno.UnsafePointerView(s.mlx_array_data_float32(r)!);
console.log(
  `compiled binary loaded ${path}\nadd -> [${view.getFloat32(0)}, ${
    view.getFloat32(4)
  }] (expected [4, 8])`,
);
lib.close();
