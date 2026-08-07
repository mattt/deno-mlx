# Contributing

## Setup

```bash
brew install mlx-c
deno --version   # 2.x
hf download HuggingFaceTB/SmolLM2-135M-Instruct
hf download sentence-transformers/all-MiniLM-L6-v2
```

## Common tasks

```bash
deno task test
deno task lint
deno fmt
deno task publish:dry-run
deno task codegen   # regenerate FFI after mlx-c upgrades
```

## Guidelines

- Keep the three-layer boundary: `core` → `tensor` → `models`.
- Prefer async eval on inference hot paths.
- Do not expand into training, CUDA, or Whisper for 0.1.x unless agreed.
- Add tests for cancellation, disposal, and weight-layout changes.
- Apple Silicon only — fail fast on other platforms in user-facing tools.

## Pull requests

- Keep diffs focused.
- Ensure `deno fmt --check`, `deno lint`, and `deno test -P=mlx` pass locally.
- Do not publish to JSR from contributor workflows.
