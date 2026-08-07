/**
 * deno-mlx CLI — chat, embed, and doctor.
 *
 *   deno task cli -- chat "Hello"
 *   deno task cli -- embed "a" "b"
 *   deno task cli -- doctor
 */

import { mlxcVersion, openMlxc, resolveMlxcPath } from "@deno-mlx/core";
import { chat, cosineSimilarity } from "@deno-mlx/models";
import {
  DEFAULT_CHAT_MODEL,
  DEFAULT_EMBED_MODEL,
  loadChatModel,
  loadEmbedModel,
  resolveChatSource,
  resolveEmbedSource,
} from "../shared/models.ts";

function usage(): never {
  console.log(`deno-mlx 0.1.0

Usage:
  deno-mlx chat [prompt] [--model-dir DIR] [--max-tokens N] [--temperature T] [--seed N]
  deno-mlx embed <text-a> <text-b> [--model-dir DIR]
  deno-mlx doctor

Environment:
  DENO_MLX_MODEL / DENO_MLX_MODEL_DIR
  DENO_MLX_EMBED_MODEL / DENO_MLX_EMBED_DIR
  DENO_MLX_DYLIB
`);
  Deno.exit(2);
}

function requireAppleSilicon(): void {
  if (Deno.build.os !== "darwin" || Deno.build.arch !== "aarch64") {
    console.error("deno-mlx requires Apple Silicon macOS (aarch64-apple-darwin).");
    Deno.exit(1);
  }
}

function parseArgs(args: string[]) {
  const positional: string[] = [];
  const flags: Record<string, string | boolean> = {};
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const next = args[i + 1];
      if (next && !next.startsWith("--")) {
        flags[key] = next;
        i++;
      } else {
        flags[key] = true;
      }
    } else {
      positional.push(a);
    }
  }
  return { positional, flags };
}

async function cmdChat(args: string[]) {
  const { positional, flags } = parseArgs(args);
  if (flags["model-dir"]) {
    Deno.env.set("DENO_MLX_MODEL_DIR", String(flags["model-dir"]));
  }
  const prompt = positional.join(" ") || "In one sentence, what is Deno?";
  const maxTokens = Number(flags["max-tokens"] ?? 100);
  const temperature = Number(flags.temperature ?? 0);
  const seed = flags.seed !== undefined ? Number(flags.seed) : undefined;

  const ac = new AbortController();
  Deno.addSignalListener("SIGINT", () => ac.abort());

  using m = await loadChatModel();
  const enc = new TextEncoder();
  const t0 = performance.now();
  let n = 0;
  for await (
    const t of chat(m, prompt, {
      maxTokens,
      temperature,
      seed,
      signal: ac.signal,
    })
  ) {
    if (t.id < 0) continue;
    await Deno.stdout.write(enc.encode(t.text));
    n++;
  }
  const dt = (performance.now() - t0) / 1000;
  console.log(`\n\n[${n} tokens, ${(n / Math.max(dt, 1e-6)).toFixed(1)} tok/s]`);
}

async function cmdEmbed(args: string[]) {
  const { positional, flags } = parseArgs(args);
  if (flags["model-dir"]) {
    Deno.env.set("DENO_MLX_EMBED_DIR", String(flags["model-dir"]));
  }
  if (positional.length < 2) usage();
  using e = await loadEmbedModel();
  const a = await e.embed(positional[0]);
  const b = await e.embed(positional[1]);
  console.log(`similarity: ${cosineSimilarity(a, b).toFixed(6)}`);
  console.log(`dimension: ${e.dimension}`);
}

function cmdDoctor() {
  console.log("deno-mlx doctor");
  console.log(`  os: ${Deno.build.os}`);
  console.log(`  arch: ${Deno.build.arch}`);
  console.log(`  deno: ${Deno.version.deno}`);
  console.log(
    `  target ok: ${Deno.build.os === "darwin" && Deno.build.arch === "aarch64"}`,
  );
  try {
    const path = resolveMlxcPath();
    console.log(`  libmlxc: ${path}`);
    console.log(`  mlx-c pin: ${mlxcVersion}`);
    openMlxc();
    console.log("  ffi: ok");
  } catch (err) {
    console.log(`  libmlxc: ERROR ${(err as Error).message}`);
  }
  const chatSrc = resolveChatSource();
  const embedSrc = resolveEmbedSource();
  console.log(`  chat source: ${chatSrc.kind} ${chatSrc.value}`);
  console.log(`  embed source: ${embedSrc.kind} ${embedSrc.value}`);
  console.log(`  default chat model: ${DEFAULT_CHAT_MODEL}`);
  console.log(`  default embed model: ${DEFAULT_EMBED_MODEL}`);
}

async function main() {
  requireAppleSilicon();
  const args = Deno.args[0] === "--" ? Deno.args.slice(1) : Deno.args;
  const [cmd, ...rest] = args;
  if (!cmd || cmd === "-h" || cmd === "--help") usage();
  switch (cmd) {
    case "chat":
      await cmdChat(rest);
      break;
    case "embed":
      await cmdEmbed(rest);
      break;
    case "doctor":
      cmdDoctor();
      break;
    default:
      usage();
  }
}

await main();
