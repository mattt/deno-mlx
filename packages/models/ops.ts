/**
 * Model-level ops: thin Tensor->Tensor wrappers over the mlx-c functions a Llama
 * forward pass needs (embedding, RMSNorm, RoPE, SDPA, transpose, concat, SwiGLU
 * activation, argmax).
 * These live in Layer 3 rather than the tensor package to keep Layer 2 small and general.
 */

import { openMlxc } from "@deno-mlx/core";
import { Tensor } from "@deno-mlx/tensor";

const mlx = openMlxc();
const c = mlx.checked;
const raw = mlx.raw;
const ptr = (b: ArrayBufferView) => Deno.UnsafePointer.of(b);

let streamHandle: Uint8Array | undefined;
const stream = () => streamHandle ??= raw.mlx_default_gpu_stream_new() as Uint8Array;

/** A null mlx_array handle ({ ctx: NULL }) for "may be null" parameters. */
const NULL_ARR = new Uint8Array(8);

const enc = new TextEncoder();
const cstr = (s: string) => enc.encode(s + "\0");

function result(fill: (res: Uint8Array) => void): Tensor {
  const r = raw.mlx_array_new() as Uint8Array;
  fill(r);
  return Tensor.fromHandle(r);
}

/** Build an mlx_optional_float ({ float value; bool has_value; }). */
function optFloat(v: number): Uint8Array {
  const b = new Uint8Array(8);
  new DataView(b.buffer).setFloat32(0, v, true);
  b[4] = 1; // has_value
  return b;
}

/** Gather rows: `weight[ids]` along `axis` (token embedding). */
export function embedding(weight: Tensor, ids: Tensor, axis = 0): Tensor {
  return result((r) =>
    c.mlx_take_axis(ptr(r), weight.handle, ids.handle, axis, stream())
  );
}

export function rmsNorm(x: Tensor, weight: Tensor, eps: number): Tensor {
  return result((r) =>
    c.mlx_fast_rms_norm(ptr(r), x.handle, weight.handle, eps, stream())
  );
}

/** RoPE over the last dim; `offset` is the absolute position of x[...,0,:]. */
export function rope(x: Tensor, dims: number, base: number, offset: number): Tensor {
  const b = optFloat(base);
  return result((r) =>
    c.mlx_fast_rope(
      ptr(r),
      x.handle,
      dims,
      false, // traditional
      b,
      1.0, // scale
      offset,
      NULL_ARR, // freqs
      stream(),
    )
  );
}

/**
 * Scaled dot-product attention.
 * `maskMode` is e.g. "causal" or "".
 */
export function sdpa(
  q: Tensor,
  k: Tensor,
  v: Tensor,
  scale: number,
  maskMode: string,
): Tensor {
  const mode = cstr(maskMode);
  return result((r) =>
    c.mlx_fast_scaled_dot_product_attention(
      ptr(r),
      q.handle,
      k.handle,
      v.handle,
      scale,
      ptr(mode),
      NULL_ARR, // mask_arr
      NULL_ARR, // sinks
      stream(),
    )
  );
}

/** Permute axes, e.g. transpose([0,2,1,3]) for [B,S,H,D] -> [B,H,S,D]. */
export function transpose(x: Tensor, axes: number[]): Tensor {
  const a = new Int32Array(axes);
  return result((r) =>
    c.mlx_transpose_axes(ptr(r), x.handle, ptr(a), BigInt(axes.length), stream())
  );
}

/** Full 2D transpose (reverse axes) — used to form W^T for linear layers. */
export function transpose2d(x: Tensor): Tensor {
  return result((r) => c.mlx_transpose(ptr(r), x.handle, stream()));
}

/** y = x @ W^T for an HF-layout weight W of shape [out, in] (no bias). */
export function linear(x: Tensor, weight: Tensor): Tensor {
  using wt = transpose2d(weight);
  return x.matmul(wt);
}

/** Concatenate along `axis` (KV cache growth uses axis 2). */
export function concat(parts: Tensor[], axis: number): Tensor {
  const vec = raw.mlx_vector_array_new() as Uint8Array;
  for (const t of parts) c.mlx_vector_array_append_value(vec, t.handle);
  const out = result((r) => c.mlx_concatenate_axis(ptr(r), vec, axis, stream()));
  raw.mlx_vector_array_free(vec);
  return out;
}

export function sigmoid(x: Tensor): Tensor {
  return result((r) => c.mlx_sigmoid(ptr(r), x.handle, stream()));
}

/** LayerNorm with weight + bias (BERT-style). */
export function layerNorm(
  x: Tensor,
  weight: Tensor,
  bias: Tensor,
  eps: number,
): Tensor {
  return result((r) =>
    c.mlx_fast_layer_norm(ptr(r), x.handle, weight.handle, bias.handle, eps, stream())
  );
}

export function erf(x: Tensor): Tensor {
  return result((r) => c.mlx_erf(ptr(r), x.handle, stream()));
}

/** Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))). */
export function gelu(x: Tensor): Tensor {
  using scaled = mulScalar(x, 1 / Math.SQRT2);
  using e = erf(scaled);
  using shifted = addScalar(e, 1);
  using half = mulScalar(shifted, 0.5);
  return x.multiply(half);
}

export function meanAxis(x: Tensor, axis: number, keepdims = false): Tensor {
  return result((r) => c.mlx_mean_axis(ptr(r), x.handle, axis, keepdims, stream()));
}

export function sumAxis(x: Tensor, axis: number, keepdims = false): Tensor {
  return result((r) => c.mlx_sum_axis(ptr(r), x.handle, axis, keepdims, stream()));
}

export function sqrt(x: Tensor): Tensor {
  return result((r) => c.mlx_sqrt(ptr(r), x.handle, stream()));
}

/** x + scalar (broadcast). */
export function addScalar(x: Tensor, v: number): Tensor {
  using s = scalarF32(v);
  return x.add(s);
}

/** y = x @ W^T + b (HF-layout weight [out, in], bias [out]). */
export function linearBias(x: Tensor, weight: Tensor, bias: Tensor): Tensor {
  using y = linear(x, weight);
  return y.add(bias);
}

export function log10(x: Tensor): Tensor {
  return result((r) => c.mlx_log10(ptr(r), x.handle, stream()));
}

/** Elementwise max of two (broadcastable) tensors. */
export function maximum(a: Tensor, b: Tensor): Tensor {
  return result((r) => c.mlx_maximum(ptr(r), a.handle, b.handle, stream()));
}

/** Clip below `minVal` (no upper bound). */
export function clipMin(x: Tensor, minVal: number): Tensor {
  using lo = scalarF32(minVal);
  return result((r) => c.mlx_clip(ptr(r), x.handle, lo.handle, NULL_ARR, stream()));
}

/** Reduce-max over all elements -> scalar tensor. */
export function maxAll(x: Tensor): Tensor {
  return result((r) => c.mlx_max(ptr(r), x.handle, false, stream()));
}

/** L2-normalize along the last axis: x / ||x||_2. */
export function l2Normalize(x: Tensor): Tensor {
  using sq = x.multiply(x);
  using ss = sumAxis(sq, x.ndim - 1, true);
  using norm = sqrt(ss);
  return x.divide(norm);
}

/** SiLU / swish: x * sigmoid(x). */
export function silu(x: Tensor): Tensor {
  using sig = sigmoid(x);
  return x.multiply(sig);
}

/** argmax over `axis` (defaults to the last), returning a uint32 index tensor. */
export function argmax(x: Tensor, axis = x.ndim - 1): Tensor {
  return result((r) => c.mlx_argmax_axis(ptr(r), x.handle, axis, false, stream()));
}

/**
 * Read a size-1 uint32 tensor (blocking).
 * Prefer {@link itemU32Async}.
 */
export function itemU32(t: Tensor): number {
  t.evalSync();
  return readU32(t);
}

/** Read a size-1 uint32 tensor after nonblocking eval. */
export async function itemU32Async(t: Tensor): Promise<number> {
  await t.eval();
  return readU32(t);
}

function readU32(t: Tensor): number {
  const p = raw.mlx_array_data_uint32(t.handle);
  if (!p) throw new Error("null uint32 data pointer");
  return new Deno.UnsafePointerView(p).getUint32(0);
}

/** A float32 scalar tensor (shape [1]) for broadcasting. */
export function scalarF32(v: number): Tensor {
  const data = new Float32Array([v]);
  const h = raw.mlx_array_new_data(
    ptr(data),
    ptr(new Int32Array([1])),
    1,
    10, // MLX_FLOAT32
  ) as Uint8Array;
  return Tensor.fromHandle(h);
}

/** x * scalar (broadcast). */
export function mulScalar(x: Tensor, factor: number): Tensor {
  using f = scalarF32(factor);
  return x.multiply(f);
}

/** A PRNG key tensor from a seed (for reproducible sampling). */
export function randomKey(seed: number): Tensor {
  return result((r) => c.mlx_random_key(ptr(r), BigInt(seed)));
}

/**
 * Split a PRNG key into two independent keys (keep, use).
 * Disposes the input key and returns ownership of both outputs.
 */
export function randomSplit(key: Tensor): { keep: Tensor; use: Tensor } {
  const keep = raw.mlx_array_new() as Uint8Array;
  const use = raw.mlx_array_new() as Uint8Array;
  c.mlx_random_split(ptr(keep), ptr(use), key.handle, stream());
  key[Symbol.dispose]();
  return { keep: Tensor.fromHandle(keep), use: Tensor.fromHandle(use) };
}

/** Sample one index from unnormalized `logits` along `axis` (does softmax). */
export function categorical(logits: Tensor, key: Tensor | undefined, axis = 0): Tensor {
  return result((r) =>
    c.mlx_random_categorical(
      ptr(r),
      logits.handle,
      axis,
      key ? key.handle : NULL_ARR,
      stream(),
    )
  );
}

/** Make an int32 tensor of token ids with the given shape. */
export function int32(ids: number[], shape: number[]): Tensor {
  const data = new Int32Array(ids);
  const h = raw.mlx_array_new_data(
    ptr(data),
    ptr(new Int32Array(shape)),
    shape.length,
    7, // MLX_INT32
  ) as Uint8Array;
  return Tensor.fromHandle(h);
}
