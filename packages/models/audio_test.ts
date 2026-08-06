/**
 * Whisper audio frontend test — deterministic and self-contained. The log-mel
 * of a fixed two-tone signal is checked against values captured from
 * mlx_whisper.audio.log_mel_spectrogram (matched there at max|Δ| ~ 4e-6).
 * Needs libmlxc.dylib and the shipped mel filterbank asset.
 */

import { assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1";
import { logMelSpectrogram, SAMPLE_RATE } from "./audio.ts";

function twoTone(): Float32Array {
  const sig = new Float32Array(SAMPLE_RATE); // 1 second
  for (let i = 0; i < sig.length; i++) {
    sig[i] = 0.5 * Math.sin((2 * Math.PI * 220 * i) / SAMPLE_RATE) +
      0.3 * Math.sin((2 * Math.PI * 440 * i) / SAMPLE_RATE);
  }
  return sig;
}

Deno.test("log-mel matches whisper reference", {
  sanitizeResources: false,
  sanitizeOps: false,
}, () => {
  using mel = logMelSpectrogram(twoTone());
  assertEquals(mel.shape, [100, 80]);

  const arr = mel.toFloat32Array();
  const row0 = [1.19725, 1.21705, 1.25884, 1.36451, 1.34897, 1.25971];
  row0.forEach((x, i) => assertAlmostEquals(arr[i], x, 1e-3));

  let sum = 0;
  for (const x of arr) sum += x;
  assertAlmostEquals(sum / arr.length, -0.29049, 1e-3); // mean
});
