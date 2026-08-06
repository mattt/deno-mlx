/**
 * Generate Deno FFI bindings from the mlx-c headers.
 *
 * Pipeline:
 *   1. clang -ast-dump=json over `mlx/c/mlx.h`  (a real parse, not regex)
 *   2. collect every `mlx_*` FunctionDecl, with return + parameter C types
 *   3. map each type via ./typemap.ts; a fn is bound only if all types map
 *   4. emit generated/{symbols,types,meta}.ts
 *   5. validate emitted names against the dylib's actual exports (`nm -gU`)
 *
 * This is a dev-time tool (deps on clang/nm are fine); the generated output is
 * committed and pinned to the mlx-c version it was produced from.
 *
 * Run: deno task codegen
 */

import { HANDLE, isStatusReturn, mapParam, mapResult, UNSUPPORTED } from "./typemap.ts";

interface CFunction {
  name: string;
  ret: string;
  params: string[];
}

const ENUM_HEADERS: Record<string, string[]> = {
  // name -> ordered member list, filled from the AST below
};

async function run(cmd: string, args: string[]): Promise<string> {
  const out = await new Deno.Command(cmd, {
    args,
    stdout: "piped",
    stderr: "piped",
  }).output();
  if (!out.success) {
    throw new Error(
      `${cmd} ${args.join(" ")} failed:\n${new TextDecoder().decode(out.stderr)}`,
    );
  }
  return new TextDecoder().decode(out.stdout);
}

async function brewPrefix(formula: string): Promise<string | null> {
  try {
    return (await run("brew", ["--prefix", formula])).trim();
  } catch {
    return null;
  }
}

/** Locate the mlx-c include dir and dylib (env overrides, else Homebrew). */
async function locate(): Promise<{ include: string; dylib: string }> {
  const envInc = Deno.env.get("DENO_MLX_INCLUDE");
  const envLib = Deno.env.get("DENO_MLX_DYLIB");
  const prefix = await brewPrefix("mlx-c");
  const include = envInc ?? (prefix ? `${prefix}/include` : null);
  const dylib = envLib ?? (prefix ? `${prefix}/lib/libmlxc.dylib` : null);
  if (!include || !dylib) {
    throw new Error(
      "Cannot locate mlx-c. Install `brew install mlx-c` or set " +
        "DENO_MLX_INCLUDE and DENO_MLX_DYLIB.",
    );
  }
  return { include, dylib };
}

/** Dump the umbrella header's AST as JSON via clang. */
async function dumpAst(include: string): Promise<unknown> {
  const probe = await Deno.makeTempFile({ suffix: ".c" });
  await Deno.writeTextFile(probe, '#include "mlx/c/mlx.h"\n');
  try {
    const json = await run("clang", [
      "-Xclang",
      "-ast-dump=json",
      "-fsyntax-only",
      "-I",
      include,
      probe,
    ]);
    return JSON.parse(json);
  } finally {
    await Deno.remove(probe);
  }
}

// deno-lint-ignore no-explicit-any
function collect(ast: any): { fns: CFunction[]; enums: typeof ENUM_HEADERS } {
  const fns: CFunction[] = [];
  const enums: typeof ENUM_HEADERS = {};
  // deno-lint-ignore no-explicit-any
  function walk(n: any) {
    if (n.kind === "FunctionDecl" && n.name?.startsWith("mlx_")) {
      const params = (n.inner ?? [])
        // deno-lint-ignore no-explicit-any
        .filter((c: any) => c.kind === "ParmVarDecl")
        // deno-lint-ignore no-explicit-any
        .map((c: any) => c.type.qualType as string);
      const full = (n.type?.qualType ?? "") as string;
      const ret = full.slice(0, full.lastIndexOf("(")).trim();
      fns.push({ name: n.name, ret, params });
    }
    if (n.kind === "EnumDecl" && n.name?.startsWith("mlx_")) {
      const members = (n.inner ?? [])
        // deno-lint-ignore no-explicit-any
        .filter((c: any) => c.kind === "EnumConstantDecl")
        // deno-lint-ignore no-explicit-any
        .map((c: any) => c.name as string);
      if (members.length) enums[n.name] = members;
    }
    for (const c of n.inner ?? []) walk(c);
  }
  walk(ast);
  // de-dup fns (umbrella can surface a decl more than once)
  const seen = new Set<string>();
  const unique = fns.filter((f) => !seen.has(f.name) && seen.add(f.name));
  return { fns: unique, enums };
}

/** Render a Deno FFI type literal as source text. */
function typeLit(t: unknown): string {
  if (typeof t === "string") return JSON.stringify(t);
  // struct: { struct: [...] }
  const s = t as { struct: unknown[] };
  return `{ struct: [${s.struct.map((x) => JSON.stringify(x)).join(", ")}] }`;
}

async function main() {
  const { include, dylib } = await locate();
  const version = (await brewCellarVersion()) ?? "unknown";
  console.log(`mlx-c include: ${include}`);
  console.log(`mlx-c dylib:   ${dylib}`);

  const ast = await dumpAst(include);
  const { fns, enums } = collect(ast);
  console.log(`parsed ${fns.length} mlx_ functions, ${Object.keys(enums).length} enums`);

  const bound: { name: string; params: unknown[]; result: unknown }[] = [];
  const skipped: { name: string; reason: string }[] = [];
  const statusReturning: string[] = [];

  for (const fn of fns) {
    const result = mapResult(fn.ret);
    if (result === UNSUPPORTED) {
      skipped.push({ name: fn.name, reason: `return type ${fn.ret}` });
      continue;
    }
    const params: unknown[] = [];
    let bad: string | null = null;
    for (const p of fn.params) {
      const m = mapParam(p);
      if (m === UNSUPPORTED) {
        bad = p;
        break;
      }
      params.push(m);
    }
    if (bad !== null) {
      skipped.push({ name: fn.name, reason: `param type ${bad}` });
      continue;
    }
    bound.push({ name: fn.name, params, result });
    if (isStatusReturn(fn.ret)) statusReturning.push(fn.name);
  }

  // ---- validate against the dylib's real exports --------------------------
  const nm = await run("nm", ["-gU", dylib]);
  const exported = new Set(
    nm.split("\n")
      .map((l) => l.match(/ T (_mlx_\w+)$/)?.[1]?.slice(1))
      .filter((x): x is string => !!x),
  );
  const missing = bound.filter((b) => !exported.has(b.name));
  if (missing.length) {
    throw new Error(
      `Generated ${missing.length} symbol(s) not exported by the dylib: ` +
        missing.slice(0, 5).map((m) => m.name).join(", "),
    );
  }
  const unbound = [...exported].filter(
    (e) => !bound.some((b) => b.name === e) && !skipped.some((s) => s.name === e),
  );

  await emit({ dylib, version, bound, skipped, statusReturning, enums });

  console.log(`\n✅ bound ${bound.length}, skipped ${skipped.length} (callback/vtable)`);
  console.log(
    `   dylib exports ${exported.size}; ${unbound.length} exports neither bound nor explicitly skipped`,
  );
  if (skipped.length) {
    console.log(`   skipped: ${skipped.map((s) => s.name).join(", ")}`);
  }
}

async function brewCellarVersion(): Promise<string | null> {
  try {
    const info = await run("brew", ["list", "--versions", "mlx-c"]);
    return info.trim().split(/\s+/)[1] ?? null;
  } catch {
    return null;
  }
}

interface EmitInput {
  dylib: string;
  version: string;
  bound: { name: string; params: unknown[]; result: unknown }[];
  skipped: { name: string; reason: string }[];
  statusReturning: string[];
  enums: typeof ENUM_HEADERS;
}

async function emit(x: EmitInput) {
  const dir = new URL("../generated/", import.meta.url).pathname;
  await Deno.mkdir(dir, { recursive: true });
  const banner = `// GENERATED by packages/core/codegen/generate.ts — DO NOT EDIT.\n` +
    `// Source: mlx-c ${x.version}. Regenerate with \`deno task codegen\`.\n\n`;

  // symbols.ts
  const syms = x.bound
    .map(
      (b) =>
        `  ${b.name}: { parameters: [${b.params.map(typeLit).join(", ")}], result: ${
          typeLit(b.result)
        } },`,
    )
    .join("\n");
  await Deno.writeTextFile(
    `${dir}symbols.ts`,
    `${banner}export const symbols = {\n${syms}\n} as const satisfies Deno.ForeignLibraryInterface;\n`,
  );

  // types.ts — enums as const objects + a status-set + handle alias
  const enumSrc = Object.entries(x.enums)
    .map(([name, members]) => {
      const ts = name
        .replace(/^mlx_/, "")
        .replace(/_$/, "") // AST tag name is e.g. `mlx_dtype_`
        .replace(/(^|_)(\w)/g, (_, __, c) => c.toUpperCase());
      const body = members.map((m, i) => `  ${m}: ${i},`).join("\n");
      return `/** mlx-c \`${name}\` */\nexport const ${ts} = {\n${body}\n} as const;\nexport type ${ts} = typeof ${ts}[keyof typeof ${ts}];`;
    })
    .join("\n\n");
  await Deno.writeTextFile(
    `${dir}types.ts`,
    `${banner}/** A single-pointer opaque mlx-c handle, as bytes returned/accepted by Deno FFI. */\nexport const HANDLE = ${
      typeLit(HANDLE)
    } as const;\n\n${enumSrc}\n`,
  );

  // meta.ts — which fns are status-returning, what was skipped
  await Deno.writeTextFile(
    `${dir}meta.ts`,
    `${banner}export const mlxcVersion = ${JSON.stringify(x.version)};\n\n` +
      `/** Functions whose C return is \`int\` (mlx-c status code; nonzero = error). */\n` +
      `export const statusReturning: ReadonlySet<string> = new Set(${
        JSON.stringify(x.statusReturning)
      });\n\n` +
      `/** Functions intentionally not bound in v0 (callbacks/vtables). */\n` +
      `export const skipped: ReadonlyArray<{ name: string; reason: string }> = ${
        JSON.stringify(x.skipped, null, 2)
      };\n`,
  );

  console.log(`emitted ${dir}{symbols,types,meta}.ts`);
}

if (import.meta.main) await main();
