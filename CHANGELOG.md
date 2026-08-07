# Changelog

## 0.1.0

### Added

- Nonblocking generation and embedding evaluation via Deno FFI threads
- `AbortSignal`, finish reasons, and PRNG key splitting for sampling
- Per-instance inference lock for safe concurrent chat/embed
- Sharded safetensors + local model directory loaders
- `chatMessages` multi-turn API
- Web (`Deno.serve`), CLI (`chat` / `embed` / `doctor`), and desktop examples
- Relocatable macOS bundling scripts for `libmlxc` + `libmlx`
- Developer ID sign / notarize scripts and GitHub Actions release workflow
- LICENSE, CI, publish dry-run readiness (packages not published yet)

### Changed

- Package versions set to `0.1.0`
- README claims corrected for multi-file native distribution
- Removed stale `--unstable-ffi` task flags on Deno 2
