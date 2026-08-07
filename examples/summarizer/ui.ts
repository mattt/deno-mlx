/** The demo's single-page UI, embedded so `deno compile` stays self-contained. */
export const HTML = String.raw`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>deno-mlx · local inference</title>
<style>
  :root {
    --bg: #f7f7f8; --card: #fff; --ink: #18181b; --muted: #71717a;
    --line: #e4e4e7; --accent: #6d4aff; --accent-ink: #fff; --pre: #fafafa;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f0f11; --card: #17171a; --ink: #ededf0; --muted: #a1a1aa;
      --line: #27272a; --accent: #8b6dff; --accent-ink: #0f0f11; --pre: #101012;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    display: flex; justify-content: center; padding: 5vh 20px;
  }
  main { width: 100%; max-width: 720px; }
  header { margin-bottom: 20px; }
  h1 {
    margin: 0; font-size: 20px; letter-spacing: -0.02em;
    display: flex; align-items: baseline; gap: 10px;
  }
  h1 .dot { color: var(--accent); }
  .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
  .sub code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .card {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 14px; padding: 16px;
  }
  textarea {
    width: 100%; min-height: 120px; resize: vertical; padding: 12px;
    border: 1px solid var(--line); border-radius: 10px; background: var(--pre);
    color: var(--ink); font: inherit; outline: none;
  }
  textarea:focus { border-color: var(--accent); }
  .row { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; align-items: center; }
  button {
    font: inherit; font-weight: 550; border-radius: 9px; padding: 8px 14px;
    border: 1px solid var(--line); background: transparent; color: var(--ink);
    cursor: pointer;
  }
  button:hover { border-color: var(--accent); }
  button.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
  button:disabled { opacity: 0.5; cursor: default; }
  .spacer { flex: 1; }
  .status { color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }
  .out {
    margin-top: 16px; padding: 14px; min-height: 64px; border-radius: 10px;
    background: var(--pre); border: 1px solid var(--line);
    white-space: pre-wrap; word-break: break-word;
  }
  .out:empty::before { content: "Output will stream here…"; color: var(--muted); }
  .cursor::after {
    content: "▋"; color: var(--accent);
    animation: blink 1s steps(2) infinite;
  }
  @keyframes blink { 50% { opacity: 0; } }
</style>
</head>
<body>
<main>
  <header>
    <h1><span class="dot">◆</span> deno-mlx</h1>
    <div class="sub">on-device inference in TypeScript · Apple Silicon · <code>__MODEL__</code></div>
  </header>
  <div class="card">
    <textarea id="input" placeholder="Paste text to summarize, or type a message to chat…"></textarea>
    <div class="row">
      <button id="paste">Paste clipboard</button>
      <button id="summarize" class="primary">Summarize</button>
      <button id="chat">Chat</button>
      <button id="embed">Similarity</button>
      <span class="spacer"></span>
      <span id="status" class="status"></span>
    </div>
    <div id="out" class="out"></div>
  </div>
</main>
<script>
  const $ = (id) => document.getElementById(id);
  const input = $("input"), out = $("out"), status = $("status");
  const buttons = ["paste", "summarize", "chat", "embed"].map($);
  const setBusy = (b) => { buttons.forEach((x) => x.disabled = b); out.classList.toggle("cursor", b); };

  async function run(mode) {
    const text = input.value.trim();
    if (!text) return;
    out.textContent = ""; status.textContent = "generating…"; setBusy(true);
    const t0 = performance.now();
    try {
      const res = await fetch("/generate", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ mode, text }),
      });
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        out.textContent += dec.decode(value, { stream: true });
        out.scrollTop = out.scrollHeight;
      }
      status.textContent = ((performance.now() - t0) / 1000).toFixed(1) + "s";
    } catch (e) {
      status.textContent = "error: " + e.message;
    } finally {
      setBusy(false);
    }
  }

  $("summarize").onclick = () => run("summarize");
  $("chat").onclick = () => run("chat");
  $("embed").onclick = async () => {
    const lines = input.value.split(/\n---\n/);
    if (lines.length < 2) {
      status.textContent = "enter two texts separated by a line with ---";
      return;
    }
    out.textContent = ""; status.textContent = "embedding…"; setBusy(true);
    try {
      const res = await fetch("/embed", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ a: lines[0].trim(), b: lines.slice(1).join("\n---\n").trim() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      out.textContent = "cosine similarity: " + data.similarity.toFixed(4) +
        "\ndimension: " + data.dim;
      status.textContent = "ok";
    } catch (e) {
      status.textContent = "error: " + e.message;
    } finally {
      setBusy(false);
    }
  };
  $("paste").onclick = async () => {
    status.textContent = "reading clipboard…";
    input.value = await (await fetch("/clipboard")).text();
    status.textContent = "";
    input.focus();
  };
</script>
</body>
</html>`;
