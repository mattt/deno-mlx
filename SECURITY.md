# Security Policy

## Supported versions

Security fixes are considered for the latest 0.1.x release line on Apple Silicon macOS
only.

## Reporting a vulnerability

Please open a private GitHub security advisory on this repository, or email the maintainer
listed in the GitHub profile. Do not file public issues for vulnerabilities that involve
model exfiltration, sandbox escapes, or signing bypass.

## Scope notes

- Inference loads untrusted model weights from disk. Treat model directories as trusted
  input.
- The web/desktop demos bind to `127.0.0.1` by default. Do not expose them on public
  interfaces without authentication.
- Compiled artifacts may vendor native MLX libraries. Verify checksums from GitHub
  Releases before running downloaded binaries.
- FFI requires `--allow-ffi`. Prefer the named `-P=mlx` permission profile.
