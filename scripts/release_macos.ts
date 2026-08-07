#!/usr/bin/env -S deno run -A
/**
 * Build CLI + desktop layouts, bundle dylibs, optionally sign/notarize.
 *
 *   deno run -A scripts/release_macos.ts
 *   SIGN=1 NOTARIZE=1 deno run -A scripts/release_macos.ts
 */

import { mlxcVersion } from "../packages/core/mod.ts";

function run(cmd: string[], cwd?: string): void {
  console.log("$", cmd.join(" "));
  const p = new Deno.Command(cmd[0], {
    args: cmd.slice(1),
    cwd,
    stdout: "inherit",
    stderr: "inherit",
  }).outputSync();
  if (p.code !== 0) throw new Error(`failed: ${cmd.join(" ")}`);
}

const dist = "dist";
Deno.mkdirSync(`${dist}/bin`, { recursive: true });

// 1. Compile CLI
run([
  "deno",
  "compile",
  "-P=mlx",
  "--allow-write",
  "--allow-net",
  "--allow-run",
  "--target",
  "aarch64-apple-darwin",
  "--output",
  `${dist}/bin/deno-mlx`,
  "examples/cli/main.ts",
]);

// 2. Bundle dylibs into dist/lib and Frameworks
run(["deno", "run", "-A", "scripts/bundle_macos.ts", "--out", dist]);

// 3. Assemble CLI archive tree
const cliRoot = `${dist}/deno-mlx-cli`;
Deno.mkdirSync(`${cliRoot}/bin`, { recursive: true });
Deno.mkdirSync(`${cliRoot}/lib`, { recursive: true });
Deno.copyFileSync(`${dist}/bin/deno-mlx`, `${cliRoot}/bin/deno-mlx`);
Deno.copyFileSync(`${dist}/lib/libmlxc.dylib`, `${cliRoot}/lib/libmlxc.dylib`);
Deno.copyFileSync(`${dist}/lib/libmlx.dylib`, `${cliRoot}/lib/libmlx.dylib`);
Deno.copyFileSync(
  `${dist}/licenses/THIRD_PARTY_NATIVE.txt`,
  `${cliRoot}/THIRD_PARTY_NATIVE.txt`,
);
Deno.writeTextFileSync(
  `${cliRoot}/README.txt`,
  `deno-mlx CLI (Apple Silicon)

Run:
  ./bin/deno-mlx doctor
  ./bin/deno-mlx chat "Hello"

Native libraries live in ./lib and are resolved relative to the executable.
mlx-c pin: ${mlxcVersion}
`,
);

// 4. Minimal .app skeleton for desktop (executable points at compiled desktop or CLI server)
const app = `${dist}/DenoMLX.app`;
Deno.mkdirSync(`${app}/Contents/MacOS`, { recursive: true });
Deno.mkdirSync(`${app}/Contents/Frameworks`, { recursive: true });
Deno.mkdirSync(`${app}/Contents/Resources`, { recursive: true });
Deno.copyFileSync(
  `${dist}/bin/deno-mlx`,
  `${app}/Contents/MacOS/DenoMLX`,
);
Deno.copyFileSync(
  `${dist}/lib/libmlxc.dylib`,
  `${app}/Contents/Frameworks/libmlxc.dylib`,
);
Deno.copyFileSync(
  `${dist}/lib/libmlx.dylib`,
  `${app}/Contents/Frameworks/libmlx.dylib`,
);
Deno.writeTextFileSync(
  `${app}/Contents/Info.plist`,
  `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>DenoMLX</string>
  <key>CFBundleIdentifier</key><string>dev.deno-mlx.app</string>
  <key>CFBundleName</key><string>DenoMLX</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
</dict></plist>
`,
);

// 5. Checksums + metadata
const meta = {
  version: "0.1.0",
  mlxcVersion,
  deno: Deno.version.deno,
  target: "aarch64-apple-darwin",
  git: (() => {
    try {
      return new TextDecoder().decode(
        new Deno.Command("git", { args: ["rev-parse", "HEAD"] }).outputSync()
          .stdout,
      ).trim();
    } catch {
      return "unknown";
    }
  })(),
};
Deno.writeTextFileSync(`${dist}/build-metadata.json`, JSON.stringify(meta, null, 2));

run([
  "bash",
  "-c",
  `cd ${dist} && ditto -c -k --sequesterRsrc --keepParent deno-mlx-cli deno-mlx-cli-aarch64.zip`,
]);
run([
  "bash",
  "-c",
  `cd ${dist} && shasum -a 256 deno-mlx-cli-aarch64.zip DenoMLX.app/Contents/MacOS/DenoMLX > SHA256SUMS`,
]);

if (Deno.env.get("SIGN") === "1") {
  run([
    "deno",
    "run",
    "-A",
    "scripts/sign_macos.ts",
    `${cliRoot}/bin/deno-mlx`,
    `${cliRoot}/lib/libmlxc.dylib`,
    `${cliRoot}/lib/libmlx.dylib`,
    app,
  ]);
}

if (Deno.env.get("NOTARIZE") === "1") {
  run([
    "bash",
    "-c",
    `cd ${dist} && ditto -c -k --sequesterRsrc --keepParent deno-mlx-cli deno-mlx-cli-aarch64.zip`,
  ]);
  run([
    "deno",
    "run",
    "-A",
    "scripts/notarize_macos.ts",
    `${dist}/deno-mlx-cli-aarch64.zip`,
  ]);
}

console.log("release artifacts staged under dist/");
