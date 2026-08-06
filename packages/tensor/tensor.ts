/**
 * @deno-mlx/tensor — Layer 2: an idiomatic, lifetime-managed tensor over
 * mlx-c's raw handles.
 *
 * Memory model (the make-or-break, per the node-mlx lessons in the plan):
 *   - MLX arrays are native and are NOT freed by GC. `Tensor` implements
 *     `Symbol.dispose`, so `using t = ...` frees deterministically.
 *   - `tidy(fn)` frees every tensor created inside `fn` except the one(s)
 *     returned — the pattern the token loop leans on to avoid per-step leaks.
 *   - `eval()` is async by default (runs the blocking MLX eval on a Deno FFI
 *     thread) so it never stalls the event loop; `evalSync()` is available too.
 */

import { Dtype, HANDLE, openMlxc, resolveMlxcPath } from "@deno-mlx/core";

const mlx = openMlxc();
const s = mlx.checked;
const ptr = (b: ArrayBufferView) => Deno.UnsafePointer.of(b);

let stream: Uint8Array | undefined;
/** Lazily-created default GPU stream, reused across ops. */
function defaultStream(): Uint8Array {
  return stream ??= s.mlx_default_gpu_stream_new() as Uint8Array;
}

// A second, non-blocking binding of the single-array eval so `Tensor.eval()`
// can return a Promise without blocking the event loop. dlopen refcounts, so
// this shares the already-loaded image with core.
let asyncEval: ((h: Uint8Array) => Promise<number>) | undefined;
function asyncEvalFn(): (h: Uint8Array) => Promise<number> {
  if (!asyncEval) {
    const lib = Deno.dlopen(resolveMlxcPath(), {
      mlx_array_eval: { parameters: [HANDLE], result: "i32", nonblocking: true },
    });
    asyncEval = lib.symbols.mlx_array_eval as (h: Uint8Array) => Promise<number>;
  }
  return asyncEval;
}

// --- tidy scope tracking ---------------------------------------------------
const scopes: Tensor[][] = [];
function track(t: Tensor): void {
  scopes[scopes.length - 1]?.push(t);
}

/**
 * Run `fn`, then dispose every `Tensor` created during it except the one(s) it
 * returns (a `Tensor`, or an array/record whose `Tensor` values are kept). If
 * `fn` throws, everything created is freed. Synchronous by design — keep the
 * MLX graph construction inside `tidy` and `await` the eval outside it.
 */
export function tidy<T>(fn: () => T): T {
  scopes.push([]);
  let out: T | undefined;
  try {
    out = fn();
    return out;
  } finally {
    const created = scopes.pop()!;
    const keep = collectTensors(out);
    for (const t of created) if (!keep.has(t)) t[Symbol.dispose]();
  }
}

function collectTensors(v: unknown): Set<Tensor> {
  const set = new Set<Tensor>();
  const add = (x: unknown) => x instanceof Tensor && set.add(x);
  if (v instanceof Tensor) add(v);
  else if (Array.isArray(v)) v.forEach(add);
  else if (v && typeof v === "object") Object.values(v).forEach(add);
  return set;
}

/** A lifetime-managed MLX array. */
export class Tensor {
  #handle: Uint8Array | null;
  #shape?: number[];
  #dtype?: number;

  private constructor(handle: Uint8Array) {
    this.#handle = handle;
    track(this);
  }

  /** Wrap a raw mlx_array handle (takes ownership of freeing it). */
  static fromHandle(handle: Uint8Array): Tensor {
    return new Tensor(handle);
  }

  /** Build a float32 tensor from JS numbers. Data is copied into MLX. */
  static fromFloat32(data: Float32Array | number[], shape: number[]): Tensor {
    const d = data instanceof Float32Array ? data : new Float32Array(data);
    if (d.length !== shape.reduce((a, b) => a * b, 1)) {
      throw new Error(`data length ${d.length} != product of shape ${shape}`);
    }
    const h = s.mlx_array_new_data(
      ptr(d),
      ptr(new Int32Array(shape)),
      shape.length,
      Dtype.MLX_FLOAT32,
    ) as Uint8Array;
    return new Tensor(h);
  }

  /** The raw handle, or throw if this tensor was disposed. */
  get handle(): Uint8Array {
    if (!this.#handle) throw new Error("Tensor used after dispose");
    return this.#handle;
  }

  get disposed(): boolean {
    return this.#handle === null;
  }

  get ndim(): number {
    return Number(s.mlx_array_ndim(this.handle));
  }

  get size(): number {
    return Number(s.mlx_array_size(this.handle));
  }

  get dtype(): number {
    return this.#dtype ??= s.mlx_array_dtype(this.handle) as number;
  }

  get shape(): number[] {
    if (this.#shape) return this.#shape;
    const nd = this.ndim;
    const p = s.mlx_array_shape(this.handle);
    const out: number[] = [];
    if (p) {
      const view = new Deno.UnsafePointerView(p);
      for (let i = 0; i < nd; i++) out.push(view.getInt32(i * 4));
    }
    return this.#shape = out;
  }

  /** Force evaluation on an FFI thread (does not block the event loop). */
  async eval(): Promise<this> {
    await asyncEvalFn()(this.handle);
    return this;
  }

  /** Force evaluation synchronously (blocks). */
  evalSync(): this {
    s.mlx_array_eval(this.handle);
    return this;
  }

  /** Evaluate and copy out as a JS Float32Array (float32 tensors only). */
  toFloat32Array(): Float32Array {
    if (this.dtype !== Dtype.MLX_FLOAT32) {
      throw new Error(
        `toFloat32Array requires float32; got dtype ${this.dtype}. Use .astype first.`,
      );
    }
    this.evalSync();
    const p = s.mlx_array_data_float32(this.handle);
    if (!p) throw new Error("null float32 data pointer");
    const n = this.size;
    const view = new Deno.UnsafePointerView(p);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = view.getFloat32(i * 4);
    return out;
  }

  /** Scalar value of a size-1 float32 tensor. */
  item(): number {
    if (this.size !== 1) throw new Error(`item() needs size 1, got ${this.size}`);
    return this.toFloat32Array()[0];
  }

  add(other: Tensor): Tensor {
    return binary("mlx_add", this, other);
  }
  subtract(other: Tensor): Tensor {
    return binary("mlx_subtract", this, other);
  }
  multiply(other: Tensor): Tensor {
    return binary("mlx_multiply", this, other);
  }
  divide(other: Tensor): Tensor {
    return binary("mlx_divide", this, other);
  }
  matmul(other: Tensor): Tensor {
    return binary("mlx_matmul", this, other);
  }

  reshape(shape: number[]): Tensor {
    const res = s.mlx_array_new() as Uint8Array;
    s.mlx_reshape(
      ptr(res),
      this.handle,
      ptr(new Int32Array(shape)),
      BigInt(shape.length), // size_t param
      defaultStream(),
    );
    return new Tensor(res);
  }

  astype(dtype: number): Tensor {
    const res = s.mlx_array_new() as Uint8Array;
    s.mlx_astype(ptr(res), this.handle, dtype, defaultStream());
    return new Tensor(res);
  }

  [Symbol.dispose](): void {
    if (this.#handle) {
      s.mlx_array_free(this.#handle);
      this.#handle = null;
    }
  }
}

/** Shared implementation for `int mlx_OP(res*, a, b, stream)` binary ops. */
function binary(op: string, a: Tensor, b: Tensor): Tensor {
  const res = s.mlx_array_new() as Uint8Array;
  // deno-lint-ignore no-explicit-any
  (s as any)[op](ptr(res), a.handle, b.handle, defaultStream());
  return Tensor.fromHandle(res);
}
