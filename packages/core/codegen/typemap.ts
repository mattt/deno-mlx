/**
 * C type -> Deno FFI type mapping for mlx-c.
 *
 * The mlx-c headers are machine-generated and use a small, closed type
 * vocabulary, so this table is nearly the whole story (see spike/README.md and
 * the M1 header survey). The rules, in order:
 *
 *   1. function pointers `(*)` and the two vtable/handler typedefs -> UNSUPPORTED
 *   2. any pointer (`T *`) -> "pointer"
 *   3. scalars -> their FFI scalar
 *   4. enums -> "i32"
 *   5. mlx_optional_* -> their real (multi-field) struct layout
 *   6. every other `mlx_*` value type is a single-pointer opaque handle -> HANDLE
 *
 * A function is bound only if its return type and every parameter map cleanly;
 * anything hitting UNSUPPORTED is recorded (with a reason) and skipped.
 */

// Deno's own FFI type unions (include the `{ struct: [...] }` form).
type NativeType = Deno.NativeType;
type NativeResultType = Deno.NativeResultType;

/** Sentinel for a type we deliberately don't bind in v0. */
export const UNSUPPORTED = Symbol("unsupported-type");

/** A single-pointer opaque mlx-c handle (`struct { void* ctx; }`). */
export const HANDLE = { struct: ["pointer"] } as const;

/** mlx-c enums are plain C enums -> 32-bit ints. */
const ENUMS = new Set([
  "mlx_dtype",
  "mlx_device_type",
  "mlx_compile_mode",
  "mlx_fft_norm",
]);

/** Multi-field value structs from optional.h (on the critical path: RoPE, quant). */
const OPTIONAL_STRUCTS: Record<string, NativeType> = {
  // struct { int value; bool has_value; }
  mlx_optional_int: { struct: ["i32", "bool"] },
  // struct { float value; bool has_value; }
  mlx_optional_float: { struct: ["f32", "bool"] },
  // struct { mlx_dtype value; bool has_value; }  (mlx_dtype is an i32 enum)
  mlx_optional_dtype: { struct: ["i32", "bool"] },
};

/** Types that carry function pointers; only referenced by UNSUPPORTED fns. */
const UNSUPPORTED_TYPES = new Set([
  "mlx_io_vtable",
  "mlx_error_handler_func",
]);

const SCALARS: Record<string, NativeType> = {
  bool: "bool",
  char: "i8",
  double: "f64",
  float: "f32",
  int: "i32",
  size_t: "usize",
  uintptr_t: "usize",
  int8_t: "i8",
  int16_t: "i16",
  int32_t: "i32",
  int64_t: "i64",
  uint8_t: "u8",
  uint16_t: "u16",
  uint32_t: "u32",
  uint64_t: "u64",
};

function normalize(c: string): string {
  return c.replace(/\bconst\b/g, "").replace(/\s+/g, " ").trim();
}

/** Map a C parameter type to a Deno FFI type, or UNSUPPORTED. */
export function mapParam(cType: string): NativeType | typeof UNSUPPORTED {
  const t = normalize(cType);
  if (t.includes("(*")) return UNSUPPORTED; // function pointer
  if (UNSUPPORTED_TYPES.has(t)) return UNSUPPORTED;
  if (t.endsWith("*")) return "pointer";
  if (t in SCALARS) return SCALARS[t];
  if (ENUMS.has(t)) return "i32";
  if (t in OPTIONAL_STRUCTS) return OPTIONAL_STRUCTS[t];
  if (t.startsWith("mlx_")) return HANDLE; // single-pointer opaque handle
  return UNSUPPORTED;
}

/** Map a C return type to a Deno FFI result type, or UNSUPPORTED. */
export function mapResult(cType: string): NativeResultType | typeof UNSUPPORTED {
  const t = normalize(cType);
  if (t === "void") return "void";
  return mapParam(cType);
}

/** True if the C return type is exactly `int` — mlx-c's status-code convention. */
export function isStatusReturn(cRet: string): boolean {
  return normalize(cRet) === "int";
}
