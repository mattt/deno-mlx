#!/usr/bin/env -S deno run -A
/**
 * Notarize and staple a macOS artifact (zip/dmg/pkg/app).
 *
 * Required env:
 *   APPLE_TEAM_ID
 *   APPLE_API_KEY_ID
 *   APPLE_API_ISSUER
 *   APPLE_API_KEY_PATH   path to AuthKey_XXX.p8
 *
 *   deno run -A scripts/notarize_macos.ts dist/deno-mlx-cli.zip
 *   deno run -A scripts/notarize_macos.ts dist/DenoMLX.dmg
 */

const teamId = Deno.env.get("APPLE_TEAM_ID");
const keyId = Deno.env.get("APPLE_API_KEY_ID");
const issuer = Deno.env.get("APPLE_API_ISSUER");
const keyPath = Deno.env.get("APPLE_API_KEY_PATH");
const target = Deno.args[0];

if (!teamId || !keyId || !issuer || !keyPath || !target) {
  console.error(
    "Need APPLE_TEAM_ID, APPLE_API_KEY_ID, APPLE_API_ISSUER, APPLE_API_KEY_PATH and a target path",
  );
  Deno.exit(2);
}

function run(cmd: string[]): string {
  const p = new Deno.Command(cmd[0], {
    args: cmd.slice(1),
    stdout: "piped",
    stderr: "piped",
  }).outputSync();
  const out = new TextDecoder().decode(p.stdout);
  const err = new TextDecoder().decode(p.stderr);
  if (p.code !== 0) throw new Error(`${cmd.join(" ")}\n${err || out}`);
  return out || err;
}

console.log(`submitting ${target}`);
run([
  "xcrun",
  "notarytool",
  "submit",
  target,
  "--key",
  keyPath,
  "--key-id",
  keyId,
  "--issuer",
  issuer,
  "--wait",
]);

console.log(`stapling ${target}`);
run(["xcrun", "stapler", "staple", target]);
run(["xcrun", "stapler", "validate", target]);

if (target.endsWith(".app") || target.endsWith(".dmg")) {
  run(["spctl", "-a", "-vv", "--type", "install", target]);
}

console.log("notarization complete");
