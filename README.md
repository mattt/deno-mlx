# deno-mlx

**Local LLM inference in TypeScript on Apple Silicon — no Python, no native build step,
one binary.**

deno-mlx binds Apple's [MLX](https://github.com/ml-explore/mlx) to Deno through its stable
FFI and the [`mlx-c`](https://github.com/ml-explore/mlx-c) API. You get streaming chat and
text embeddings in plain TypeScript, running on the GPU via Metal.

## Why this exists (isn't node-mlx enough?)

[node-mlx](https://github.com/frost-beta/node-mlx) is excellent — and it's for **Node**,
shipped as a **compiled C++ N-API addon** (with per-platform prebuilt binaries). deno-mlx
is a different bet:

- **Deno-native.** node-mlx doesn't run on Deno. This is the only option for the Deno
  ecosystem — secure-by-default permissions, JSR, URL imports, native TS.
- **No native build step.** It's pure `Deno.dlopen` over the `mlx-c` Homebrew bottle — no
  node-gyp, no prebuild matrix, no `.node` addon to compile or ship.
- **One self-contained binary.** `deno compile` bundles your app, and the dylib loads
  right beside it — a distributable **native Mac app with embedded local inference,
  written in TypeScript**. A native addon can't collapse into a single file like that.

Same MLX engine underneath; a fundamentally simpler distribution story. The goal is
durable infrastructure for the Deno/TS side, not a faster inference core.

## Quick start

```bash
brew install mlx-c
hf download HuggingFaceTB/SmolLM2-360M-Instruct
deno task chat HuggingFaceTB/SmolLM2-360M-Instruct "Name three primes."
```

```ts
import { chat, loadModel } from "@deno-mlx/models";

using m = await loadModel("HuggingFaceTB/SmolLM2-360M-Instruct");
for await (const { text } of chat(m, "Explain FFI in one sentence.")) {
  await Deno.stdout.write(new TextEncoder().encode(text));
}
```

Or run the streaming clipboard-summarizer demo: `deno task demo` →
<http://localhost:8787>.

## Packages

- **`@deno-mlx/core`** — raw `mlx-c` bindings, generated from the headers.
- **`@deno-mlx/tensor`** — a `Tensor` with `using`/`Symbol.dispose` lifetimes, async
  `eval()`, and a `tidy()` scope.
- **`@deno-mlx/models`** — streaming `generate()` (KV cache) and BERT embeddings.

Every model path is validated bit-exact against the reference stacks (`mlx_lm`,
transformers.js). Requires an Apple Silicon Mac and Deno 2.x.

**Status:** working v0 — chat + embeddings. Whisper (frontend done) and benchmarks are in
progress. MIT.
