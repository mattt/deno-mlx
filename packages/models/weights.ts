/**
 * Weight loading via mlx-c's native safetensors reader.
 *
 * `mlx_load_safetensors` parses the file and returns a name->array map straight into MLX
 * (no JS-side parsing, zero-copy into unified memory).
 * We keep the map handle and hand out `Tensor`s lazily by name —
 * the model builder knows the keys it needs (`model.layers.0.self_attn.q_proj.weight`, …).
 *
 * Multi-file loads (sharded safetensors) search maps in order.
 */

import { MlxError, openMlxc } from "@deno-mlx/core";
import { Tensor } from "@deno-mlx/tensor";

const mlx = openMlxc();
const raw = mlx.raw;
const checked = mlx.checked;
const enc = new TextEncoder();
const ptr = (b: ArrayBufferView) => Deno.UnsafePointer.of(b);
/** NUL-terminated C string buffer — keep the returned value alive across the call. */
const cstr = (s: string) => enc.encode(s + "\0");

interface Shard {
  map: Uint8Array;
  meta: Uint8Array;
  stream: Uint8Array;
  path: string;
}

export class Weights {
  #shards: Shard[];

  private constructor(shards: Shard[]) {
    this.#shards = shards;
  }

  /** Load a .safetensors file into an MLX-backed weight map. */
  static load(path: string): Weights {
    return Weights.loadPaths([path]);
  }

  /** Load one or more shard files; later shards override earlier keys on get. */
  static loadPaths(paths: string[]): Weights {
    if (paths.length === 0) throw new Error("Weights.loadPaths: empty path list");
    const shards: Shard[] = [];
    try {
      for (const path of paths) shards.push(loadShard(path));
    } catch (err) {
      for (const s of shards) freeShard(s);
      throw err;
    }
    return new Weights(shards);
  }

  /**
   * Fetch a weight by exact name (throws if absent).
   * Caller owns the Tensor.
   */
  get(name: string): Tensor {
    // Search last-to-first so later shards win (HF shards are usually disjoint).
    for (let i = this.#shards.length - 1; i >= 0; i--) {
      const shard = this.#shards[i];
      const out = raw.mlx_array_new() as Uint8Array;
      const key = cstr(name);
      const rc = raw.mlx_map_string_to_array_get(ptr(out), shard.map, ptr(key));
      if (rc === 0) return Tensor.fromHandle(out);
      raw.mlx_array_free(out);
    }
    throw new Error(`weight not found: ${name}`);
  }

  /** Whether a weight exists (used to detect tied vs. explicit lm_head, etc.). */
  has(name: string): boolean {
    for (const shard of this.#shards) {
      const out = raw.mlx_array_new() as Uint8Array;
      const key = cstr(name);
      const rc = raw.mlx_map_string_to_array_get(ptr(out), shard.map, ptr(key));
      if (rc === 0) {
        raw.mlx_array_free(out);
        return true;
      }
      raw.mlx_array_free(out);
    }
    return false;
  }

  /** All weight names across shards (deduplicated, later shards win). */
  keys(): string[] {
    const seen = new Set<string>();
    const keys: string[] = [];
    for (const shard of this.#shards) {
      for (const k of shardKeys(shard)) {
        if (!seen.has(k)) {
          seen.add(k);
          keys.push(k);
        }
      }
    }
    return keys;
  }

  [Symbol.dispose](): void {
    for (const s of this.#shards) freeShard(s);
    this.#shards = [];
  }
}

function loadShard(path: string): Shard {
  const map = raw.mlx_map_string_to_array_new() as Uint8Array;
  const meta = raw.mlx_map_string_to_string_new() as Uint8Array;
  // Reading from disk is a CPU op — the lazy Load nodes must be CPU-scheduled,
  // or a later GPU eval fails with "Load::eval_gpu Not implemented".
  const stream = raw.mlx_default_cpu_stream_new() as Uint8Array;
  const file = cstr(path);
  try {
    checked.mlx_load_safetensors(ptr(map), ptr(meta), ptr(file), stream);
  } catch (err) {
    raw.mlx_map_string_to_array_free(map);
    raw.mlx_map_string_to_string_free(meta);
    raw.mlx_stream_free(stream);
    if (err instanceof MlxError) {
      throw new Error(`failed to load safetensors: ${path}: ${err.message}`, {
        cause: err,
      });
    }
    throw new Error(`failed to load safetensors: ${path}`, { cause: err });
  }
  return { map, meta, stream, path };
}

function freeShard(s: Shard): void {
  raw.mlx_map_string_to_array_free(s.map);
  raw.mlx_map_string_to_string_free(s.meta);
  raw.mlx_stream_free(s.stream);
}

function shardKeys(shard: Shard): string[] {
  const it = raw.mlx_map_string_to_array_iterator_new(shard.map) as Uint8Array;
  const keys: string[] = [];
  const keyOut = new Uint8Array(8);
  const keyView = new DataView(keyOut.buffer);
  const valOut = raw.mlx_array_new() as Uint8Array;
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
