#!/usr/bin/env -S deno run -A
/**
 * Developer ID sign nested macOS artifacts inside-out.
 *
 * Required env:
 *   MACOS_CODESIGN_IDENTITY  e.g. "Developer ID Application: Example (TEAMID)"
 * Optional:
 *   MACOS_ENTITLEMENTS       path to entitlements plist
 *
 *   deno run -A scripts/sign_macos.ts dist/bin/deno-mlx dist/lib/*.dylib
 *   deno run -A scripts/sign_macos.ts dist/DenoMLX.app
 */

const identity = Deno.env.get("MACOS_CODESIGN_IDENTITY");
if (!identity) {
  console.error("MACOS_CODESIGN_IDENTITY is required");
  Deno.exit(1);
}

const entitlements = Deno.env.get("MACOS_ENTITLEMENTS");
const targets = Deno.args;
if (targets.length === 0) {
  console.error("usage: sign_macos.ts <path>...");
  Deno.exit(2);
}

function run(cmd: string[]): void {
  const p = new Deno.Command(cmd[0], {
    args: cmd.slice(1),
    stdout: "inherit",
    stderr: "inherit",
  }).outputSync();
  if (p.code !== 0) throw new Error(`command failed: ${cmd.join(" ")}`);
}

function sign(path: string, deep = false): void {
  const args = [
    "codesign",
    "--force",
    "--options",
    "runtime",
    "--timestamp",
    "--sign",
    identity!,
  ];
  if (entitlements) args.push("--entitlements", entitlements);
  if (deep) args.push("--deep");
  args.push(path);
  console.log(`signing ${path}`);
  run(args);
  run(["codesign", "--verify", "--verbose=2", path]);
}

for (const t of targets) {
  const st = Deno.statSync(t);
  if (st.isDirectory && t.endsWith(".app")) {
    // Inside-out: Frameworks first, then binary, then bundle.
    const frameworks = `${t}/Contents/Frameworks`;
    try {
      for (const e of Deno.readDirSync(frameworks)) {
        if (e.name.endsWith(".dylib")) sign(`${frameworks}/${e.name}`);
      }
    } catch {
      // no Frameworks
    }
    const macos = `${t}/Contents/MacOS`;
    try {
      for (const e of Deno.readDirSync(macos)) {
        sign(`${macos}/${e.name}`);
      }
    } catch {
      // ignore
    }
    sign(t, true);
  } else {
    sign(t);
  }
}

console.log("signing complete");
