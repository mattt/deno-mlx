import { assertEquals, assertThrows } from "jsr:@std/assert@^1";
import { resolveWeightFiles } from "./safetensors.ts";

Deno.test("resolveWeightFiles finds single model.safetensors", () => {
  const dir = Deno.makeTempDirSync();
  try {
    Deno.writeFileSync(`${dir}/model.safetensors`, new Uint8Array([1]));
    const r = resolveWeightFiles(dir);
    assertEquals(r.sharded, false);
    assertEquals(r.paths, [`${dir}/model.safetensors`]);
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test("resolveWeightFiles resolves shard index", () => {
  const dir = Deno.makeTempDirSync();
  try {
    Deno.writeTextFileSync(
      `${dir}/model.safetensors.index.json`,
      JSON.stringify({
        weight_map: {
          "a.weight": "model-00001-of-00002.safetensors",
          "b.weight": "model-00002-of-00002.safetensors",
        },
      }),
    );
    Deno.writeFileSync(
      `${dir}/model-00001-of-00002.safetensors`,
      new Uint8Array([1]),
    );
    Deno.writeFileSync(
      `${dir}/model-00002-of-00002.safetensors`,
      new Uint8Array([1]),
    );
    const r = resolveWeightFiles(dir);
    assertEquals(r.sharded, true);
    assertEquals(r.paths.length, 2);
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test("resolveWeightFiles errors on missing shard", () => {
  const dir = Deno.makeTempDirSync();
  try {
    Deno.writeTextFileSync(
      `${dir}/model.safetensors.index.json`,
      JSON.stringify({
        weight_map: { "a.weight": "model-00001-of-00002.safetensors" },
      }),
    );
    assertThrows(() => resolveWeightFiles(dir), Error, "Incomplete sharded");
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});
