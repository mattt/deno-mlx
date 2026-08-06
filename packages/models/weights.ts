/**
 * Weight loading via mlx-c's native safetensors reader.
 *
 * `mlx_load_safetensors` parses the file and returns a name->array map straight
 * into MLX (no JS-side parsing, zero-copy into unified memory). We keep the map
 * handle and hand out `Tensor`s lazily by name — the model builder knows the
 * keys it needs (`model.layers.0.self_attn.q_proj.weight`, …).
 */

import { openMlxc } from "@deno-mlx/core";
import { Tensor } from "@deno-mlx/tensor";

const mlx = openMlxc();
const raw = mlx.raw;
const enc = new TextEncoder();
const ptr = (b: ArrayBufferView) => Deno.UnsafePointer.of(b);
/** NUL-terminated C string buffer — keep the returned value alive across the call. */
const cstr = (s: string) => enc.encode(s + "\0");

export class Weights {
  #map: Uint8Array; // mlx_map_string_to_array handle (owned)
  #meta: Uint8Array; // mlx_map_string_to_string handle (owned)
  #stream: Uint8Array;

  private constructor(map: Uint8Array, meta: Uint8Array, stream: Uint8Array) {
    this.#map = map;
    this.#meta = meta;
    this.#stream = stream;
  }

  /** Load a .safetensors file into an MLX-backed weight map. */
  static load(path: string): Weights {
    const map = raw.mlx_map_string_to_array_new() as Uint8Array;
    const meta = raw.mlx_map_string_to_string_new() as Uint8Array;
    // Reading from disk is a CPU op — the lazy Load nodes must be CPU-scheduled,
    // or a later GPU eval fails with "Load::eval_gpu Not implemented".
    const stream = raw.mlx_default_cpu_stream_new() as Uint8Array;
    const file = cstr(path); // keep alive across the call
    const rc = raw.mlx_load_safetensors(
      ptr(map),
      ptr(meta),
      ptr(file),
      stream,
    );
    if (rc !== 0) throw new Error(`failed to load safetensors: ${path}`);
    return new Weights(map, meta, stream);
  }

  /** Fetch a weight by exact name (throws if absent). Caller owns the Tensor. */
  get(name: string): Tensor {
    const out = raw.mlx_array_new() as Uint8Array;
    const key = cstr(name);
    const rc = raw.mlx_map_string_to_array_get(ptr(out), this.#map, ptr(key));
    if (rc !== 0) throw new Error(`weight not found: ${name}`);
    return Tensor.fromHandle(out);
  }

  /** Whether a weight exists (used to detect tied vs. explicit lm_head, etc.). */
  has(name: string): boolean {
    const out = raw.mlx_array_new() as Uint8Array;
    const key = cstr(name);
    const rc = raw.mlx_map_string_to_array_get(ptr(out), this.#map, ptr(key));
    if (rc === 0) raw.mlx_array_free(out);
    return rc === 0;
  }

  /** All weight names in the file. */
  keys(): string[] {
    const it = raw.mlx_map_string_to_array_iterator_new(this.#map) as Uint8Array;
    const keys: string[] = [];
    const keyOut = new Uint8Array(8); // receives a `const char*` by value
    const keyView = new DataView(keyOut.buffer);
    const valOut = raw.mlx_array_new() as Uint8Array; // mlx_array* out
    // iterator_next returns 0 while an item was produced, nonzero at end.
    while (
      raw.mlx_map_string_to_array_iterator_next(ptr(keyOut), ptr(valOut), it) === 0
    ) {
      const strPtr = Deno.UnsafePointer.create(keyView.getBigUint64(0, true));
      if (strPtr) keys.push(new Deno.UnsafePointerView(strPtr).getCString());
    }
    raw.mlx_map_string_to_array_iterator_free(it);
    raw.mlx_array_free(valOut);
    return keys;
  }

  [Symbol.dispose](): void {
    raw.mlx_map_string_to_array_free(this.#map);
    raw.mlx_map_string_to_string_free(this.#meta);
  }
}
