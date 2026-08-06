/**
 * M0 — FFI de-risk spike (gates the whole project).
 *
 * Proves the single load-bearing assumption of deno-mlx: that Deno's stable FFI
 * can drive mlx-c's ABI directly, specifically that it can
 *
 *   1. RETURN a single-pointer struct by value  (`mlx_array mlx_array_new_data(...)`)
 *   2. PASS   a single-pointer struct by value  (`mlx_add(..., mlx_array a, ...)`)
 *   3. round-trip real data through an on-device op (add) and read it back
 *   4. do all of the above at a per-call overhead low enough for a TS token loop
 *
 * All mlx-c handles are `typedef struct { void* ctx; } mlx_*;` — represented to
 * Deno FFI as `{ struct: ["pointer"] }`. A returned struct comes back as a
 * Uint8Array of its bytes; the same Uint8Array is accepted as a by-value arg.
 *
 * Run: deno task spike
 */

// --- dylib resolution (M0: hard-coded Homebrew path; real resolver lands in M1) ---
const MLXC = Deno.env.get("DENO_MLX_DYLIB") ??
  "/opt/homebrew/opt/mlx-c/lib/libmlxc.dylib";

// A single-pointer opaque handle, by value.
const HANDLE = { struct: ["pointer"] } as const;

const lib = Deno.dlopen(MLXC, {
  // mlx_array mlx_array_new(void);
  mlx_array_new: { parameters: [], result: HANDLE },
  // mlx_array mlx_array_new_data(const void* data, const int* shape, int dim, mlx_dtype dtype);
  mlx_array_new_data: {
    parameters: ["pointer", "pointer", "i32", "i32"],
    result: HANDLE,
  },
  // mlx_stream mlx_default_gpu_stream_new(void);
  mlx_default_gpu_stream_new: { parameters: [], result: HANDLE },
  // int mlx_add(mlx_array* res, const mlx_array a, const mlx_array b, const mlx_stream s);
  mlx_add: {
    parameters: ["pointer", HANDLE, HANDLE, HANDLE],
    result: "i32",
  },
  // int mlx_array_eval(mlx_array arr);
  mlx_array_eval: { parameters: [HANDLE], result: "i32" },
  // size_t mlx_array_size(const mlx_array arr);
  mlx_array_size: { parameters: [HANDLE], result: "u64" },
  // const float* mlx_array_data_float32(const mlx_array arr);
  mlx_array_data_float32: { parameters: [HANDLE], result: "pointer" },
  // int mlx_array_free(mlx_array arr);
  mlx_array_free: { parameters: [HANDLE], result: "i32" },
});

const s = lib.symbols;
const MLX_FLOAT32 = 10; // enum index in mlx_dtype (BOOL=0 .. FLOAT32=10)

function ptrOf(buf: ArrayBufferView): Deno.PointerValue {
  return Deno.UnsafePointer.of(buf);
}

/** Build an mlx_array from a JS Float32Array. Returns the handle bytes (Uint8Array). */
function arrayFromFloats(data: Float32Array, shape: number[]): Uint8Array {
  const shapeBuf = new Int32Array(shape);
  return s.mlx_array_new_data(
    ptrOf(data),
    ptrOf(shapeBuf),
    shape.length,
    MLX_FLOAT32,
  ) as Uint8Array;
}

function readFloats(handle: Uint8Array, n: number): Float32Array {
  const dataPtr = s.mlx_array_data_float32(handle);
  if (dataPtr === null) throw new Error("mlx_array_data_float32 returned null");
  const view = new Deno.UnsafePointerView(dataPtr);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = view.getFloat32(i * 4);
  return out;
}

// ---------------------------------------------------------------------------
// 1) Correctness: a + b on device, read back.
// ---------------------------------------------------------------------------
const a = arrayFromFloats(new Float32Array([1, 2, 3, 4]), [4]);
const b = arrayFromFloats(new Float32Array([10, 20, 30, 40]), [4]);
const stream = s.mlx_default_gpu_stream_new() as Uint8Array;

// mlx_add writes into *res; res must point to an existing (empty) mlx_array.
const res = s.mlx_array_new() as Uint8Array;
const rc = s.mlx_add(ptrOf(res), a, b, stream);
if (rc !== 0) throw new Error(`mlx_add failed rc=${rc}`);

s.mlx_array_eval(res);
const size = Number(s.mlx_array_size(res));
const out = readFloats(res, size);

const expected = [11, 22, 33, 44];
const ok = size === 4 && expected.every((v, i) => Math.abs(out[i] - v) < 1e-6);

console.log("mlx-c dylib:", MLXC);
console.log("add result :", Array.from(out), "size:", size);
console.log("expected   :", expected);
console.log(ok ? "✅ struct-by-value round-trip PASSED" : "❌ MISMATCH");

s.mlx_array_free(a);
s.mlx_array_free(b);
s.mlx_array_free(res);

// ---------------------------------------------------------------------------
// 2) Overhead: how expensive is one FFI call across the struct boundary?
//    - pure FFI (mlx_array_size): no compute, isolates call cost
//    - full op+eval: what a naive token-loop step pays
// ---------------------------------------------------------------------------
const probe = arrayFromFloats(new Float32Array([1]), [1]);

const N = 1_000_000;
let t0 = performance.now();
for (let i = 0; i < N; i++) s.mlx_array_size(probe);
let dt = performance.now() - t0;
console.log(
  `\npure FFI call (mlx_array_size): ${(dt / N * 1000).toFixed(1)} ns/call ` +
    `(${(N / dt / 1000).toFixed(1)}M calls/s)`,
);

const M = 10_000;
const one = arrayFromFloats(new Float32Array([1]), [1]);
t0 = performance.now();
for (let i = 0; i < M; i++) {
  const r = s.mlx_array_new() as Uint8Array;
  s.mlx_add(ptrOf(r), probe, one, stream);
  s.mlx_array_eval(r);
  s.mlx_array_free(r);
}
dt = performance.now() - t0;
console.log(
  `add + eval + free round-trip:  ${(dt / M * 1000).toFixed(1)} µs/op ` +
    `(${(M / dt * 1000).toFixed(0)} ops/s)`,
);

s.mlx_array_free(probe);
s.mlx_array_free(one);
lib.close();
