#!/usr/bin/env -S deno run -A
/**
 * Stage relocatable libmlxc.dylib + libmlx.dylib for CLI / desktop artifacts.
 *
 * Copies from Homebrew (or DENO_MLX_DYLIB / DENO_MLX_LIBMLX),
 * rewrites install names to @loader_path,
 * and fails if any /opt/homebrew dependency remains.
 *
 *   deno task bundle:macos
 *   deno task bundle:macos -- --out dist
 */

import { mlxcVersion } from "../packages/core/mod.ts";

const outRoot = flag("--out") ?? "dist";
const libDir = `${outRoot}/lib`;
const frameworksDir = `${outRoot}/DenoMLX.app/Contents/Frameworks`;

const mlxcSrc = Deno.env.get("DENO_MLX_DYLIB") ??
  "/opt/homebrew/opt/mlx-c/lib/libmlxc.dylib";
const mlxSrc = Deno.env.get("DENO_MLX_LIBMLX") ??
  "/opt/homebrew/opt/mlx/lib/libmlx.dylib";

function flag(name: string): string | undefined {
  const i = Deno.args.indexOf(name);
  return i >= 0 ? Deno.args[i + 1] : undefined;
}

function mustExist(path: string): void {
  try {
    Deno.statSync(path);
  } catch {
    throw new Error(`Required native library not found: ${path}`);
  }
}

function run(cmd: string[], label: string): string {
  const p = new Deno.Command(cmd[0], {
    args: cmd.slice(1),
    stdout: "piped",
    stderr: "piped",
  }).outputSync();
  if (p.code !== 0) {
    throw new Error(
      `${label} failed: ${new TextDecoder().decode(p.stderr)}`,
    );
  }
  return new TextDecoder().decode(p.stdout);
}

function otoolDeps(path: string): string[] {
  const out = run(["otool", "-L", path], "otool");
  return out
    .split("\n")
    .slice(1)
    .map((l) => l.trim().split(" ")[0])
    .filter(Boolean);
}

function stagePair(destDir: string): void {
  Deno.mkdirSync(destDir, { recursive: true });
  const mlxcDst = `${destDir}/libmlxc.dylib`;
  const mlxDst = `${destDir}/libmlx.dylib`;
  Deno.copyFileSync(mlxcSrc, mlxcDst);
  Deno.copyFileSync(mlxSrc, mlxDst);

  // Make libmlx id relative and point libmlxc at @loader_path/libmlx.dylib.
  run(["install_name_tool", "-id", "@loader_path/libmlx.dylib", mlxDst], "id mlx");
  run(["install_name_tool", "-id", "@loader_path/libmlxc.dylib", mlxcDst], "id mlxc");

  for (const dep of otoolDeps(mlxcDst)) {
    if (dep.includes("libmlx.dylib") && !dep.startsWith("@loader_path")) {
      run(
        ["install_name_tool", "-change", dep, "@loader_path/libmlx.dylib", mlxcDst],
        "rewrite mlx dep",
      );
    }
  }

  for (const path of [mlxcDst, mlxDst]) {
    const bad = otoolDeps(path).filter((d) =>
      d.includes("/opt/homebrew") || d.includes("/usr/local/opt")
    );
    if (bad.length) {
      throw new Error(
        `${path} still depends on Homebrew paths:\n  ${bad.join("\n  ")}`,
      );
    }
  }
}

mustExist(mlxcSrc);
mustExist(mlxSrc);
console.log(`mlx-c pin: ${mlxcVersion}`);
console.log(`staging from:\n  ${mlxcSrc}\n  ${mlxSrc}`);

stagePair(libDir);
stagePair(frameworksDir);

// License notices for redistributed native libs.
Deno.mkdirSync(`${outRoot}/licenses`, { recursive: true });
const notice = `This release bundles prebuilt Apple MLX / mlx-c dynamic libraries.
See upstream licenses:
  https://github.com/ml-explore/mlx
  https://github.com/ml-explore/mlx-c
Pinned mlx-c version expected by bindings: ${mlxcVersion}
`;
Deno.writeTextFileSync(`${outRoot}/licenses/THIRD_PARTY_NATIVE.txt`, notice);

console.log(`staged:
  ${libDir}/libmlxc.dylib
  ${libDir}/libmlx.dylib
  ${frameworksDir}/libmlxc.dylib
  ${frameworksDir}/libmlx.dylib
`);
