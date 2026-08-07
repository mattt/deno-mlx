/**
 * deno-mlx desktop preview (experimental `deno desktop` WebView backend).
 *
 * Reuses the same loopback HTTP app as the web example.
 * On first launch,
 * downloads pinned demo models into ~/Library/Application Support/deno-mlx/models
 * when DENO_MLX_MODEL_DIR / cache is unset.
 *
 *   deno task desktop
 */

import { createAppState, makeHandler } from "../shared/handler.ts";
import {
  appModelsDir,
  DEFAULT_CHAT_MODEL,
  DEFAULT_EMBED_MODEL,
} from "../shared/models.ts";
import { ensureDemoModels } from "./models_download.ts";

if (Deno.build.os !== "darwin" || Deno.build.arch !== "aarch64") {
  console.error("deno-mlx desktop requires Apple Silicon macOS.");
  Deno.exit(1);
}

const PORT = Number(Deno.env.get("PORT") ?? 8787);
const HOST = "127.0.0.1";

if (!Deno.env.get("DENO_MLX_MODEL_DIR") && !Deno.env.get("DENO_MLX_MODEL")) {
  const chatDir = `${appModelsDir()}/${DEFAULT_CHAT_MODEL.replaceAll("/", "--")}`;
  const embedDir = `${appModelsDir()}/${DEFAULT_EMBED_MODEL.replaceAll("/", "--")}`;
  await ensureDemoModels([
    { repo: DEFAULT_CHAT_MODEL, dest: chatDir },
    { repo: DEFAULT_EMBED_MODEL, dest: embedDir },
  ]);
  Deno.env.set("DENO_MLX_MODEL_DIR", chatDir);
  Deno.env.set("DENO_MLX_EMBED_DIR", embedDir);
}

const state = await createAppState({ withEmbeddings: true });
const handler = makeHandler(state);

Deno.serve({ port: PORT, hostname: HOST }, handler);
console.log(`[deno-mlx desktop] http://${HOST}:${PORT}`);
console.log(
  "[deno-mlx desktop] Open this URL in a WebView via: deno desktop --backend webview examples/desktop/main.ts",
);
