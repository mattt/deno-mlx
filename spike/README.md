# M0 — FFI de-risk spike (results)

The gating question for the whole project — _can Deno's stable FFI drive mlx-c's ABI
directly?_ — is **answered yes**. Everything downstream (codegen, tensor wrapper, model
runners) rests on these four results.

Reproduce:

```bash
brew install mlx-c            # ships libmlxc.dylib 0.6.0 as a prebuilt bottle
deno task spike               # correctness + overhead
deno compile --allow-ffi --allow-env --allow-read --unstable-ffi \
  -o /tmp/mlx-probe spike/m0_compile_probe.ts && env -i /tmp/mlx-probe
```

## Findings

| Question                                                                    | Result                                                                                                                                                                      |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Return single-pointer struct by value (`mlx_array mlx_array_new_data(...)`) | ✅ comes back as an 8-byte `Uint8Array`                                                                                                                                     |
| Pass single-pointer struct by value (`mlx_add(..., mlx_array a, ...)`)      | ✅ same `Uint8Array` accepted as a by-value arg                                                                                                                             |
| Real data round-trip through an on-device op                                | ✅ `[1,2,3,4] + [10,20,30,40] = [11,22,33,44]`                                                                                                                              |
| Per-call FFI overhead low enough for a TS token loop                        | ✅ pure FFI call is negligible; the 100µs/op figure is GPU dispatch+sync on a **1-element** op, not FFI — a real forward pass is one graph eval, not thousands of tiny ones |
| `deno compile` single binary resolves + loads the dylib at runtime          | ✅ 65MB binary runs under `env -i` (no PATH, no `deno`), computes `[4,8]`                                                                                                   |

## Decisions this locks in

- **Type mapping:** every mlx-c handle (`mlx_array`, `mlx_stream`, `mlx_device`, …) is
  `typedef struct { void* ctx; } ...;` → `{ struct: ["pointer"] }` in Deno FFI. The M1
  codegen type table is essentially this one rule plus scalars.
- **Distribution:** `mlx-c` is a **Homebrew bottle** (0.6.0) — no from-source build needed
  for dev. The resolver's Homebrew path is real; the vendored/download path (M1) covers
  machines without Homebrew.
- **Ops convention:** results are out-params (`mlx_array* res`, pre-created via
  `mlx_array_new()`); the `int` return is an error code → maps to thrown exceptions.
- **Pin:** codegen targets mlx-c **v0.6.0** (matches the installed bottle).

## Bug found & fixed

The resolver's `canStat` swallowed a `PermissionDenied` error as "file absent", which
mis-reported a missing `--allow-read` as "dylib not installed". Fixed to surface
permission errors with an actionable message (`packages/core/resolver.ts`).
