/**
 * Whisper audio frontend: 16 kHz mono PCM -> log-mel spectrogram, matching
 * openai-whisper / mlx-whisper numerically.
 *
 * The STFT is done as a real DFT matmul (cos/sin bases) rather than a complex
 * FFT — exact for the fixed n_fft=400 and it keeps everything in real tensors.
 * The 80-bin mel filterbank is whisper's own (shipped as ./assets).
 */

import { Tensor } from "@deno-mlx/tensor";
import { addScalar, clipMin, log10, maxAll, maximum, mulScalar } from "./ops.ts";

export const SAMPLE_RATE = 16000;
const N_FFT = 400;
const HOP = 160;
const N_MELS = 80;
const FREQS = N_FFT / 2 + 1; // 201

/** Read WAV (16-bit or 32-bit float PCM, mono) into a Float32Array at 16 kHz. */
export function readWav(bytes: Uint8Array): Float32Array {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (str(dv, 0, 4) !== "RIFF" || str(dv, 8, 4) !== "WAVE") {
    throw new Error("not a RIFF/WAVE file");
  }
  let off = 12;
  let fmt = 1, channels = 1, rate = SAMPLE_RATE;
  let dataOff = -1, dataLen = 0;
  while (off + 8 <= dv.byteLength) {
    const id = str(dv, off, 4);
    const size = dv.getUint32(off + 4, true);
    const body = off + 8;
    if (id === "fmt ") {
      fmt = dv.getUint16(body, true);
      channels = dv.getUint16(body + 2, true);
      rate = dv.getUint32(body + 4, true);
    } else if (id === "data") {
      dataOff = body;
      dataLen = size;
    }
    off = body + size + (size & 1);
  }
  if (dataOff < 0) throw new Error("no data chunk");
  if (rate !== SAMPLE_RATE) {
    throw new Error(`expected ${SAMPLE_RATE} Hz mono WAV, got ${rate} Hz`);
  }
  const n = fmt === 3 ? dataLen / 4 : dataLen / 2;
  const frames = Math.floor(n / channels);
  const out = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    // average channels to mono
    let acc = 0;
    for (let ch = 0; ch < channels; ch++) {
      const k = i * channels + ch;
      acc += fmt === 3
        ? dv.getFloat32(dataOff + k * 4, true)
        : dv.getInt16(dataOff + k * 2, true) / 32768;
    }
    out[i] = acc / channels;
  }
  return out;
}

function str(dv: DataView, off: number, len: number): string {
  let s = "";
  for (let i = 0; i < len; i++) s += String.fromCharCode(dv.getUint8(off + i));
  return s;
}

// ---- precomputed, reused across calls ------------------------------------
let dftCos: Tensor | undefined; // [N_FFT, FREQS]
let dftSin: Tensor | undefined;
let melFb: Tensor | undefined; // [FREQS, N_MELS]
const hann = new Float32Array(N_FFT);
for (let n = 0; n < N_FFT; n++) hann[n] = 0.5 * (1 - Math.cos((2 * Math.PI * n) / N_FFT));

function dftBases(): { cos: Tensor; sin: Tensor } {
  if (!dftCos || !dftSin) {
    const cos = new Float32Array(N_FFT * FREQS);
    const sin = new Float32Array(N_FFT * FREQS);
    for (let t = 0; t < N_FFT; t++) {
      for (let k = 0; k < FREQS; k++) {
        const a = (2 * Math.PI * k * t) / N_FFT;
        cos[t * FREQS + k] = Math.cos(a);
        sin[t * FREQS + k] = Math.sin(a);
      }
    }
    dftCos = Tensor.fromFloat32(cos, [N_FFT, FREQS]);
    dftSin = Tensor.fromFloat32(sin, [N_FFT, FREQS]);
  }
  return { cos: dftCos, sin: dftSin };
}

function melFilters(): Tensor {
  if (!melFb) {
    const path = new URL("./assets/mel_filters_80.bin", import.meta.url);
    const raw = Deno.readFileSync(path); // [80, 201] float32, row-major
    const f = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    // transpose to [201, 80] so mag2[F,201] @ fb[201,80] -> [F,80]
    const t = new Float32Array(FREQS * N_MELS);
    for (let m = 0; m < N_MELS; m++) {
      for (let k = 0; k < FREQS; k++) t[k * N_MELS + m] = f[m * FREQS + k];
    }
    melFb = Tensor.fromFloat32(t, [FREQS, N_MELS]);
  }
  return melFb;
}

/** Reflect-pad without repeating the edge sample (numpy 'reflect'). */
function reflect(idx: number, n: number): number {
  while (idx < 0 || idx >= n) {
    if (idx < 0) idx = -idx;
    if (idx >= n) idx = 2 * (n - 1) - idx;
  }
  return idx;
}

/**
 * log-mel spectrogram, shape [frames, 80], matching whisper's
 * `log_mel_spectrogram` (center STFT, hann window, per-spectrogram max clamp).
 */
export function logMelSpectrogram(audio: Float32Array): Tensor {
  const pad = N_FFT / 2;
  const n = audio.length;
  // center STFT gives 1 + floor((n + 2*pad - N_FFT)/HOP) frames; whisper drops
  // the trailing frame, leaving floor((n + 2*pad - N_FFT)/HOP).
  const frames = Math.floor((n + 2 * pad - N_FFT) / HOP);

  // build windowed frames [frames, N_FFT] on CPU
  const framed = new Float32Array(frames * N_FFT);
  for (let f = 0; f < frames; f++) {
    const start = f * HOP - pad; // absolute index into audio (center STFT)
    for (let t = 0; t < N_FFT; t++) {
      framed[f * N_FFT + t] = audio[reflect(start + t, n)] * hann[t];
    }
  }

  using frameT = Tensor.fromFloat32(framed, [frames, N_FFT]);
  const { cos, sin } = dftBases();
  using re = frameT.matmul(cos); // [frames, FREQS]
  using im = frameT.matmul(sin);
  using mag2 = re.multiply(re).add(im.multiply(im));
  using mel = mag2.matmul(melFilters()); // [frames, N_MELS]

  using clamped = clipMin(mel, 1e-10);
  using logSpec = log10(clamped);
  using gmax = maxAll(logSpec);
  using thresh = addScalar(gmax, -8);
  using bounded = maximum(logSpec, thresh);
  using shifted = addScalar(bounded, 4);
  return mulScalar(shifted, 0.25);
}
