# deno-mlx

**MLX for Deno** provides local LLM chat and text embeddings on Apple Silicon with TypeScript,
without requiring Python or a native addon build step.

deno-mlx binds Apple's [MLX](https://github.com/ml-explore/mlx) through Deno FFI
and the [`mlx-c`](https://github.com/ml-explore/mlx-c) C API.
It provides streaming chat and sentence embeddings, deterministic tensor
lifetimes (`using` / `Symbol.dispose`), permission profiles, and support for
compiled CLI and desktop apps that include prebuilt MLX libraries.

## Status

This project currently supports chat with Llama-family models and BERT embeddings.
A Whisper frontend exists in the repository,
but is not part of the supported API.

## Requirements

- Apple Silicon Mac (arm64)
- macOS 14+
- [Deno](https://deno.land) 2.x
- [`mlx-c`](https://formulae.brew.sh/formula/mlx-c): `brew install mlx-c`
  (bindings use the version in `@deno-mlx/core`'s generated metadata)
- Models in the Hugging Face cache (`hf download ...`)
  or a local model directory

## Quick start

```bash
brew install mlx-c
hf download HuggingFaceTB/SmolLM2-360M-Instruct
hf download sentence-transformers/all-MiniLM-L6-v2

deno task chat "Name three primes."
deno task cli -- doctor
deno task web   # http://127.0.0.1:8787
```

```ts
import { chat, loadModel } from "@deno-mlx/models";

using m = await loadModel("HuggingFaceTB/SmolLM2-360M-Instruct");
for await (const { text } of chat(m, "Explain FFI in one sentence.")) {
  await Deno.stdout.write(new TextEncoder().encode(text));
}
```

## Packages

| Package            | Role                                                    |
| ------------------ | ------------------------------------------------------- |
| `@deno-mlx/core`   | Generated `mlx-c` FFI, dylib resolver, `MlxError`       |
| `@deno-mlx/tensor` | Lifetime-managed `Tensor`, `tidy()`, async `eval()`     |
| `@deno-mlx/models` | Streaming chat, BERT embeddings, safetensors / HF cache |

Workspace development uses bare specifiers (`@deno-mlx/models`).

## Examples

| Task                                   | Description                                              |
| -------------------------------------- | -------------------------------------------------------- |
| `deno task chat`                       | Minimal streaming chat                                   |
| `deno task cli -- chat\|embed\|doctor` | Full CLI                                                 |
| `deno task web`                        | Loopback `Deno.serve` UI (chat + summarize + similarity) |
| `deno task desktop`                    | Same app for experimental `deno desktop` WebView         |

On first launch, the desktop example downloads pinned demo models through
[`@huggingface/hub`](https://www.npmjs.com/package/@huggingface/hub).
It uses Xet acceleration when the Hub file is Xet-backed.
For gated repositories, set the optional `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`.

Use `-P=mlx` (ffi, env, read) for inference permissions.
Network access is required only for tokenizer and model downloads.
Clipboard paste in the web UI requires `--allow-run` (`pbpaste`).

## Distribution (compiled apps)

Ship these files with a compiled release:

1. the compiled executable (or `.app`), and
2. vendored `libmlxc.dylib` + `libmlx.dylib` with relocatable `@loader_path`
   install names.

The resolver supports these layouts:

- `{execDir}/libmlxc.dylib` or `{execDir}/vendor/libmlxc.dylib`
- `{execDir}/../lib/libmlxc.dylib` (CLI archive)
- `{execDir}/../Frameworks/libmlxc.dylib` (macOS app bundle)

```bash
deno task compile:cli
deno task bundle:macos
# optional, with Developer ID secrets configured:
SIGN=1 NOTARIZE=1 deno run -A scripts/release_macos.ts
```

See `scripts/release_macos.ts`, `scripts/sign_macos.ts`, and
`scripts/notarize_macos.ts`.
The GitHub Actions release workflow builds artifacts on `macos-15`.
It signs and notarizes them when secrets are present.
The workflow does not publish packages to JSR.

## Model loading

- Hub repository IDs resolve through the shared HF cache (`HF_HOME` /
  `HF_HUB_CACHE`).
- Use `loadModelFromDir`, `loadEmbedderFromDir`, `DENO_MLX_MODEL_DIR`, or
  `DENO_MLX_EMBED_DIR` for local directories.
- Models can use one `model.safetensors` file or a sharded
  `model.safetensors.index.json` file.

## Runtime notes

- Generation and embeddings evaluate on Deno's nonblocking FFI path.
- Concurrent calls on the same loaded model/embedder are serialized.
- Pass `AbortSignal` to cancel generation (including while queued).
- `openMlxc()` has process-wide scope.
  The current version does not support unloading mlx-c while the process runs.

## Comparison with node-mlx and mlx-node

node-mlx and mlx-node are Node projects that use N-API addons.
deno-mlx uses Deno FFI and `mlx-c` for permissioned, embeddable local inference.
It is not a training platform or a full Python MLX clone.

## License

deno-mlx is available under the MIT license.
See the [LICENSE](/LICENSE) file for more info.

Bundled MLX and mlx-c dynamic libraries use their upstream licenses.
When you package them, see `dist/licenses/THIRD_PARTY_NATIVE.txt`.
