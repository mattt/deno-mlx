# deno-mlx

**On-device inference in TypeScript, on Apple Silicon — no Python, no server, no native
build step.**

deno-mlx binds [Apple MLX](https://github.com/ml-explore/mlx) to Deno through its stable
FFI and the [`mlx-c`](https://github.com/ml-explore/mlx-c) C API. Because it's pure
`Deno.dlopen` (no compiled N-API addon) and `deno compile` produces a single binary, you
get local LLM chat, embeddings, and audio features in plain TypeScript — and a path to
shipping native Mac apps with embedded inference.

> **Status:** working v0. Chat generation, text embeddings, and the Whisper audio frontend
> are implemented and validated bit-exact against the reference Python stacks. See
> [Validation](#validation) and [Roadmap](#roadmap).

## Requirements

- Apple Silicon Mac
- [Deno](https://deno.com) 2.x
- The MLX C library: `brew install mlx-c` (ships a prebuilt `libmlxc.dylib`)

The dylib is resolved at runtime in this order: `DENO_MLX_DYLIB` → Homebrew → a copy
beside the executable (so `deno compile`d apps can vendor it).

## Packages

Three layers, published separately so you can stop at whichever suits you:

| Package            | Layer         | What it gives you                                                                                             |
| ------------------ | ------------- | ------------------------------------------------------------------------------------------------------------- |
| `@deno-mlx/core`   | raw bindings  | 596 `mlx-c` functions generated from the headers, a dylib resolver, and typed `MlxError`s.                    |
| `@deno-mlx/tensor` | safe wrapper  | A `Tensor` with `using`/`Symbol.dispose` lifetimes, async `eval()`, a `tidy()` scope, and method-chained ops. |
| `@deno-mlx/models` | model runners | `generate()` chat with streaming + KV cache, BERT text embeddings, and the Whisper log-mel frontend.          |

## Quick start

Grab a small model into the shared Hugging Face cache, then chat:

```bash
brew install mlx-c
hf download HuggingFaceTB/SmolLM2-360M-Instruct
deno task chat HuggingFaceTB/SmolLM2-360M-Instruct "Name three prime numbers."
```

### Streaming chat

```ts
import { chat, loadModel } from "@deno-mlx/models";

using m = await loadModel("HuggingFaceTB/SmolLM2-360M-Instruct");
for await (const { text } of chat(m, "Explain FFI in one sentence.")) {
  await Deno.stdout.write(new TextEncoder().encode(text));
}
```

### Embeddings

```ts
import { cosineSimilarity, loadEmbedder } from "@deno-mlx/models";

using e = await loadEmbedder("sentence-transformers/all-MiniLM-L6-v2");
const [a, b] = e.embedAll(["a cat on a mat", "a feline on a rug"]);
console.log(cosineSimilarity(a, b)); // ~0.56
```

### Tensors (Layer 2)

```ts
import { Tensor, tidy } from "@deno-mlx/tensor";

using a = Tensor.fromFloat32([1, 2, 3, 4], [2, 2]);
using b = Tensor.fromFloat32([1, 0, 0, 1], [2, 2]);
const c = tidy(() => a.matmul(b).add(a)); // intermediates freed
console.log((await c.eval()).toFloat32Array());
c[Symbol.dispose]();
```

## Demo — local streaming summarizer

A single Deno process that loads a chat model and streams tokens to a browser UI. Reads
your clipboard, summarizes on-device, no network round-trip:

```bash
deno task demo   # then open http://localhost:8787
```

`examples/summarizer/` embeds its UI in the binary, so
`deno compile --allow-ffi --allow-env --allow-read --allow-net --allow-run --unstable-ffi examples/summarizer/server.ts`
yields a self-contained native app.

## Tasks

| Task                             | Purpose                                                        |
| -------------------------------- | -------------------------------------------------------------- |
| `deno task chat [repo] [prompt]` | Stream a chat reply from the terminal.                         |
| `deno task demo`                 | Run the clipboard-summarizer web app.                          |
| `deno task codegen`              | Regenerate `@deno-mlx/core` bindings from the `mlx-c` headers. |
| `deno task test`                 | Run the suite (needs `mlx-c`; model tests skip if uncached).   |

## How it works

- **Layer 1 is generated.** `packages/core/codegen` runs `clang -ast-dump=json` over
  `mlx/c/*.h`, maps each C type to a Deno FFI type, and emits the symbol table — validated
  against the dylib's actual `nm` exports (596 bound + 18 callback/vtable functions
  skipped = every export accounted for). Pinned to `mlx-c` 0.6.0; a version bump is a
  re-run + diff.
- **MLX handles are single-pointer structs** (`struct { void* ctx; }`), which Deno FFI
  passes/returns by value — the feasibility spike in `spike/` proves this end-to-end,
  including a `deno compile`d binary loading the dylib in a clean environment.
- **Memory is explicit.** MLX arrays aren't GC-freed, so `Tensor` uses `Symbol.dispose`;
  `tidy()` frees a scope's intermediates (the token loop leans on it). `eval()` runs on an
  FFI thread so it never stalls the event loop.
- **Weights are shared with the HF tools.** Models load via `mlx_load_safetensors`
  straight from `~/.cache/huggingface/hub`, so `hf download` and deno-mlx use one copy.
  Tokenization is pure-JS via transformers.js — no second native dep.

## Validation

Every model path is checked bit-exact against the reference implementation:

| Path                  | Reference       | Result                       |
| --------------------- | --------------- | ---------------------------- |
| Llama prefill logits  | `mlx_lm`        | argmax + logit match exactly |
| Llama KV-cache decode | `mlx_lm`        | 12 greedy tokens identical   |
| BERT embeddings       | transformers.js | cosine `1.00000`             |
| Whisper log-mel       | `mlx_whisper`   | max &#124;Δ&#124; ≈ 4e-6     |

Tests are network-free (fixed token ids) and live beside each package.

## Roadmap

- ✅ M0–M3: FFI spike → core bindings → tensor wrapper → streaming chat
- ✅ M4a: text embeddings (BERT)
- ◐ M4b: Whisper — log-mel frontend done and validated; encoder/decoder + transcription
  loop pending
- ☐ M5: benchmarks vs the Node + Ollama-server pattern; JSR publish

**Scope discipline (v0):** chat generation, embeddings, and Whisper. Not chasing
vision-language models or training — the point is durable infrastructure, not a
reimplementation of `mlx-lm`.

## Layout

```
packages/core     @deno-mlx/core    (+ codegen/, generated/)
packages/tensor   @deno-mlx/tensor
packages/models   @deno-mlx/models  (llama, bert, whisper frontend, tokenizer)
examples/         chat CLI + summarizer web app
spike/            M0 FFI de-risk spike + results
```

## License

MIT
