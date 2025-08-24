# README — Web MVP: Real-Time Speech↔Speech (RU↔EN)

No fluff. This is the developer-facing guide to stand up a browser UI that:

* captures **mic** or **tab/system** audio,
* streams 16 kHz mono PCM over **WebSocket** to FastAPI,
* runs **Canary AST → MT** and **Silero TTS** on the server,
* streams PCM back,
* plays it with Web Audio,
* lets the user pick input/output devices.

It assumes you have the project’s Python modules (you uploaded):

* `/mnt/data/canary_ast.py` (Canary wrapper)
* `/mnt/data/silero_tts.py` (Silero wrapper)
* `/mnt/data/realtime_s2s.py` (reference pipeline & utilities)

---

## 0) Requirements

**Runtime**

* Python 3.10+ (Linux/Windows/macOS). CUDA recommended (RTX A4000).
* Chrome 114+ (for AudioWorklet + `getDisplayMedia`/device selection).
* HTTPS (or `http://localhost`) is required for media capture.

**Python packages (server)**

```bash
pip install fastapi uvicorn[standard] numpy soundfile pydub
# Your model deps (install according to your environment):
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install nemo_toolkit['all']  # for Canary if not already installed
# pip install silero @ your existing install method
```

**Optional: Docker** (see §7)

---

## 1) Project Layout

```text
.
├─ server/
│  ├─ app.py                 # FastAPI app w/ WebSocket
│  ├─ audio_util.py          # (if you have it) helpers: to_mono16k_float32, wav_bytes_from_float32
│  ├─ canary_ast.py          # your wrapper (uploaded)
│  ├─ silero_tts.py          # your wrapper (uploaded)
│  └─ realtime_s2s.py        # reference script (uploaded)
└─ frontend/
   ├─ index.html
   ├─ app.js
   ├─ styles.css
   └─ worklets/
      ├─ encoder.worklet.js  # downmix + resample → Int16 16 kHz; posts 20–30 ms chunks
      └─ (optional) player.worklet.js  # streaming playback (or use AudioBufferSourceNode)
```

---

## 2) WebSocket Protocol

* **Text frames**: JSON control/events.
* **Binary frames (client→server)**: raw **16 kHz mono Int16LE** PCM chunks (20–30 ms each).
* **Binary frames (server→client)**: raw **16 kHz mono Int16LE** PCM chunks (TTS audio).

Initial control message (client→server, JSON):

```json
{
  "type": "control",
  "src": "ru",
  "tgt": "en",
  "mode": "mic",            // "mic" | "tab"
  "format": "s16le",
  "sample_rate": 16000,
  "vad_threshold": 0.5,     // optional
  "push_to_talk": false     // optional
}
```

Server events (JSON):

```json
{ "type": "ready" }
{ "type": "stats", "latency_ms": 780, "queue": 0 }
{ "type": "transcript", "src": "ru", "text": "..." }
{ "type": "error", "message": "..." }
```

Binary frames have **no header**. Direction is inferred by who sent them.

---

## 3) Backend (FastAPI)

`server/app.py`:

```python
import asyncio
import json
import logging
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

# Import your wrappers
from server.canary_ast import ModelManager as ASTManager  # adapt if your class differs
from server.silero_tts import TTSManager                  # adapt if your class differs

log = logging.getLogger("uvicorn.error")
app = FastAPI()
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# --- Global models (lazy init) ---
_ast: Optional[ASTManager] = None
_tts: Optional[TTSManager] = None

async def ensure_models(device: str = "cuda"):
    global _ast, _tts
    if _ast is None:
        _ast = ASTManager()      # adapt to your init signature
        _ast.load()              # warmup if needed
    if _tts is None:
        _tts = TTSManager()      # adapt to your init signature
        _tts.load(lang="en")     # or per-language later

# --- Simple PCM queue → processing task ---
async def process_loop(cfg, in_pcm_queue: asyncio.Queue, ws: WebSocket):
    """
    Pulls PCM frames from queue, aggregates into segments,
    runs Canary AST → TTS, streams TTS PCM back over WS.
    """
    src = cfg.get("src", "ru")
    tgt = cfg.get("tgt", "en")
    sr  = int(cfg.get("sample_rate", 16000))

    # naive segmenter: 0.6s windows with 0.1s hop (tune as needed or use VAD)
    seg_samples = int(0.6 * sr)
    buf = np.zeros(0, dtype=np.int16)

    while True:
        chunk = await in_pcm_queue.get()
        if chunk is None:
            break
        buf = np.concatenate([buf, chunk])
        if buf.shape[0] >= seg_samples:
            segment = buf[:seg_samples].astype(np.int16)
            buf = buf[seg_samples:]

            # --- AST → text ---
            # Expect float32 -1..1 for your AST; convert
            audio_f32 = (segment.astype(np.float32) / 32768.0).copy()
            # Your AST API may accept np.ndarray or wav path; adapt:
            text = _ast.translate(audio_f32, src_lang=src, tgt_lang=tgt)  # adapt to your signature
            if isinstance(text, dict) and "text" in text:
                text = text["text"]

            # --- TTS → PCM ---
            # Adapt TTS API to return 16k mono int16 bytes/array
            tts_pcm = _tts.tts(text=text, lang=tgt, sample_rate=sr)  # adapt method/signature
            # Ensure Int16LE bytes
            if isinstance(tts_pcm, np.ndarray):
                out_bytes = tts_pcm.astype(np.int16).tobytes()
            else:
                out_bytes = tts_pcm  # if already bytes

            await ws.send_bytes(out_bytes)
            await ws.send_text(json.dumps({"type": "transcript", "src": src, "text": text}))

@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    await ensure_models()

    cfg = {"src": "ru", "tgt": "en", "sample_rate": 16000}
    in_pcm_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    task = asyncio.create_task(process_loop(cfg, in_pcm_queue, ws))

    await ws.send_text(json.dumps({"type": "ready"}))

    try:
        while True:
            msg = await ws.receive()
            if "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "control":
                        cfg.update(data)
                        await ws.send_text(json.dumps({"type": "ack", "cfg": cfg}))
                except Exception as e:
                    await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            elif "bytes" in msg:
                # Binary PCM from client
                buf = msg["bytes"]
                # Convert to int16 mono array
                pcm = np.frombuffer(buf, dtype="<i2")  # little-endian int16
                try:
                    in_pcm_queue.put_nowait(pcm)
                except asyncio.QueueFull:
                    # drop to keep latency bounded
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        await in_pcm_queue.put(None)
        await task
```

> **Adapt the `_ast.translate(...)` and `_tts.tts(...)` calls to match your actual wrappers.** Your uploaded files are partially redacted; wire them here.

Run locally:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/`.

---

## 4) Frontend (HTML/CSS)

`frontend/index.html`:

```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Real-Time S2S MVP</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <main>
    <section class="controls">
      <div>
        <label>Source:</label>
        <select id="sourceSelect">
          <option value="mic">Microphone</option>
          <option value="tab">Tab/System (getDisplayMedia)</option>
        </select>
      </div>

      <div>
        <label>Mic:</label>
        <select id="micSelect"></select>
      </div>

      <div>
        <label>Speaker:</label>
        <select id="speakerSelect"></select>
      </div>

      <div>
        <label>Src→Tgt:</label>
        <select id="srcLang">
          <option value="ru">ru</option>
          <option value="en">en</option>
        </select>
        →
        <select id="tgtLang">
          <option value="en">en</option>
          <option value="ru">ru</option>
        </select>
      </div>

      <div>
        <label>VAD:</label>
        <input id="vadSlider" type="range" min="0" max="1" step="0.05" value="0.5" />
      </div>

      <div class="buttons">
        <button id="startBtn">Start</button>
        <button id="stopBtn" disabled>Stop</button>
        <button id="pttBtn" data-pressed="false">Push-to-Talk</button>
      </div>
    </section>

    <section class="meters">
      <canvas id="vu" width="200" height="20"></canvas>
      <span id="latency"></span>
    </section>

    <section class="transcripts">
      <div><h3>Source</h3><pre id="srcText"></pre></div>
      <div><h3>Target</h3><pre id="tgtText"></pre></div>
    </section>

    <!-- Hidden audio element to route to selected output device -->
    <audio id="outAudio" autoplay></audio>
  </main>

  <script src="app.js"></script>
</body>
</html>
```

`frontend/styles.css` — keep it minimal.

---

## 5) Frontend (JS + Worklets)

### 5.1 Encoder Worklet

`frontend/worklets/encoder.worklet.js`:

```js
// Downmix to mono, resample to 16 kHz, emit Int16LE frames of ~20–30 ms
// NOTE: simple linear resampler; replace with higher quality if needed.

const TARGET_SR = 16000;
const CHUNK_MS  = 20; // tweak 20..30
const samplesPerChunk = Math.round(TARGET_SR * (CHUNK_MS / 1000));

class EncoderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._accum = new Float32Array(0);
    this._srcRate = sampleRate; // worklet's rate (typically 48000)
  }

  _downmixStereoToMono(buf) {
    if (buf.numberOfChannels === 1) return buf.getChannelData(0);
    const L = buf.getChannelData(0), R = buf.getChannelData(1);
    const N = Math.min(L.length, R.length);
    const out = new Float32Array(N);
    for (let i = 0; i < N; i++) out[i] = 0.5 * (L[i] + R[i]);
    return out;
  }

  _linearResample(float32, fromSr, toSr) {
    if (fromSr === toSr) return float32;
    const ratio = toSr / fromSr;
    const N = Math.floor(float32.length * ratio);
    const out = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const srcPos = i / ratio;
      const i0 = Math.floor(srcPos);
      const t = srcPos - i0;
      const a = float32[i0] || 0;
      const b = float32[i0 + 1] || a;
      out[i] = a + (b - a) * t;
    }
    return out;
  }

  _f32ToInt16LEBytes(f32) {
    const out = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
      let s = Math.max(-1, Math.min(1, f32[i]));
      out[i] = (s * 0x7fff) | 0;
    }
    return new Uint8Array(out.buffer);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    // input[0] is an array of channels
    const mono = input.length === 1 ? input[0] : (() => {
      const L = input[0], R = input[1] || input[0];
      const N = Math.min(L.length, R.length);
      const out = new Float32Array(N);
      for (let i = 0; i < N; i++) out[i] = 0.5 * (L[i] + R[i]);
      return out;
    })();

    const resampled = this._linearResample(mono, this._srcRate, TARGET_SR);

    // accumulate
    const merged = new Float32Array(this._accum.length + resampled.length);
    merged.set(this._accum, 0);
    merged.set(resampled, this._accum.length);
    this._accum = merged;

    while (this._accum.length >= samplesPerChunk) {
      const chunkF32 = this._accum.slice(0, samplesPerChunk);
      this._accum = this._accum.slice(samplesPerChunk);
      const bytes = this._f32ToInt16LEBytes(chunkF32);
      this.port.postMessage(bytes, [bytes.buffer]); // transfer
    }

    return true;
  }
}

registerProcessor('encoder-processor', EncoderProcessor);
```

### 5.2 Frontend Orchestration

`frontend/app.js`:

```js
let audioCtx, ws, mediaStream, sourceNode, workletNode, analyser, destNode;
const outAudio = document.getElementById('outAudio');

const micSelect = document.getElementById('micSelect');
const speakerSelect = document.getElementById('speakerSelect');
const sourceSelect = document.getElementById('sourceSelect');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const pttBtn = document.getElementById('pttBtn');
const srcLang = document.getElementById('srcLang');
const tgtLang = document.getElementById('tgtLang');
const vadSlider = document.getElementById('vadSlider');
const srcText = document.getElementById('srcText');
const tgtText = document.getElementById('tgtText');
const vuCanvas = document.getElementById('vu');

async function listDevices() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  micSelect.innerHTML = devices
    .filter(d => d.kind === 'audioinput')
    .map(d => `<option value="${d.deviceId}">${d.label || d.deviceId}</option>`).join('');
  speakerSelect.innerHTML = devices
    .filter(d => d.kind === 'audiooutput')
    .map(d => `<option value="${d.deviceId}">${d.label || d.deviceId}</option>`).join('');
}

async function getInputStream() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  if (sourceSelect.value === 'tab') {
    mediaStream = await navigator.mediaDevices.getDisplayMedia({ audio: true, video: false });
  } else {
    const deviceId = micSelect.value || undefined;
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: deviceId ? { deviceId: { exact: deviceId } } : true
    });
  }
  return mediaStream;
}

async function setOutputDevice() {
  const sinkId = speakerSelect.value;
  if (outAudio.setSinkId) {
    await outAudio.setSinkId(sinkId);
  }
}

function drawVU() {
  if (!analyser) return;
  const ctx = vuCanvas.getContext('2d');
  const data = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(data);
  let peak = 0;
  for (let i = 0; i < data.length; i++) {
    const v = Math.abs((data[i] - 128) / 128);
    if (v > peak) peak = v;
  }
  ctx.clearRect(0, 0, vuCanvas.width, vuCanvas.height);
  ctx.fillStyle = '#0f0';
  ctx.fillRect(0, 0, Math.round(peak * vuCanvas.width), vuCanvas.height);
  requestAnimationFrame(drawVU);
}

async function start() {
  startBtn.disabled = true;
  stopBtn.disabled = false;

  await listDevices();
  await setOutputDevice();

  audioCtx = new AudioContext();

  await audioCtx.audioWorklet.addModule('worklets/encoder.worklet.js');
  const stream = await getInputStream();
  sourceNode = audioCtx.createMediaStreamSource(stream);

  workletNode = new AudioWorkletNode(audioCtx, 'encoder-processor');
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;

  // routing: source -> analyser -> worklet
  sourceNode.connect(analyser);
  analyser.connect(workletNode);

  // Playback routing: server PCM -> MediaStream -> <audio> (so we can setSinkId)
  destNode = audioCtx.createMediaStreamDestination();
  outAudio.srcObject = destNode.stream;

  // Socket
  ws = new WebSocket(`ws://${location.host}/ws/audio`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    ws.send(JSON.stringify({
      type: 'control',
      src: srcLang.value,
      tgt: tgtLang.value,
      mode: sourceSelect.value,
      sample_rate: 16000,
      format: 's16le',
      vad_threshold: parseFloat(vadSlider.value)
    }));
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'transcript') {
        // crude split between src/tgt panes
        srcText.textContent += (msg.src || 'src') + ': ' + msg.text + '\n';
        tgtText.textContent += (msg.src === tgtLang.value ? 'tgt' : 'mt') + ': ' + msg.text + '\n';
      }
      return;
    }
    // Binary: Int16LE 16k mono -> play
    const pcm = new Int16Array(ev.data);
    // Convert to Float32 and schedule
    const f32 = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) f32[i] = Math.max(-1, pcm[i] / 32768);
    const buffer = audioCtx.createBuffer(1, f32.length, 16000);
    buffer.copyToChannel(f32, 0, 0);
    const src = new AudioBufferSourceNode(audioCtx, { buffer });
    src.connect(destNode);
    src.start();
  };

  workletNode.port.onmessage = (ev) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(ev.data); // already Int16LE bytes from worklet
    }
  };

  drawVU();
}

function stop() {
  stopBtn.disabled = true;
  startBtn.disabled = false;
  if (ws) { ws.close(); ws = null; }
  if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
  if (workletNode) { workletNode.disconnect(); workletNode = null; }
  if (analyser) { analyser.disconnect(); analyser = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
}

startBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);
pttBtn.addEventListener('mousedown', () => pttBtn.dataset.pressed = 'true');
pttBtn.addEventListener('mouseup',   () => pttBtn.dataset.pressed = 'false');

navigator.mediaDevices.getUserMedia({ audio: true }).then(listDevices);
navigator.mediaDevices.addEventListener('devicechange', listDevices);
```

**Notes**

* Worklet resampler is linear → acceptable for MVP. Replace with higher-quality later.
* `AudioBufferSourceNode` chunk scheduling is simple and can click under heavy load; for smoother playback, implement a small FIFO scheduler or a Player Worklet with a ring buffer.

---

## 6) Browser/OS Notes (capture & output)

* `getUserMedia()` works on all modern browsers over HTTPS/localhost.
* `getDisplayMedia({ audio:true })`:

  * **Windows/ChromeOS** can capture **system audio**.
  * **macOS/Linux** generally capture **tab/window audio only**.
  * UX: browser shows a “Share audio” checkbox; users must tick it.
* Output device selection:

  * Use `<audio>.setSinkId(deviceId)` when available (Chrome/Edge).
  * Fallback: default output device.

---

## 7) Docker (optional)

`Dockerfile` (GPU layers omitted—keep your existing CUDA base if needed):

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY server/ server/
COPY frontend/ frontend/

RUN pip install --no-cache-dir fastapi uvicorn[standard] numpy soundfile pydub
# Install your model deps here as appropriate

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build & run:

```bash
docker build -t s2s-web .
docker run --rm -p 8000:8000 --gpus all -e CUDA_VISIBLE_DEVICES=0 s2s-web
```

---

## 8) Tuning & Troubleshooting

* **Chunk size**: 20–30 ms at 16 kHz → 320–480 samples (640–960 bytes). Smaller chunks = lower latency, higher overhead.
* **Queue & backpressure**: If `process_loop` can’t keep up, you’ll drop chunks (by design). Track latency and queue size via periodic `stats` events.
* **CORS/HTTPS**: For non-localhost deployment, terminate TLS (nginx/traefik) and proxy WS (`/ws/audio`).
* **Device labels empty**: You must call `getUserMedia` once before `enumerateDevices()` to unlock labels.
* **macOS “system audio”**: Not available to Chrome except tab/window. Use tab capture or a desktop wrapper (Electron/Tauri) with a virtual audio device if you truly need system-wide capture.

---

## 9) Roadmap (post-MVP)

* **WebRTC/`aiortc` transport** for jitter buffer, NAT traversal, A/V renegotiation.
* **Player Worklet** + ring buffer for gapless playback.
* **Proper VAD** in the browser (e.g., energy or WebNN) to reduce server load.
* **Better resampler** in the worklet (polyphase/soxr-quality).
* **Metrics**: per-stage timings (AST, MT, TTS), end-to-end latency, underrun/overrun counters.

---

## 10) Quick Start

```bash
# 1) Install server deps
python -m venv .venv && source .venv/bin/activate  # (or .venv\Scripts\activate on Windows)
pip install fastapi uvicorn[standard] numpy soundfile pydub
# plus your Canary/Silero deps

# 2) Run server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# 3) Open UI
open http://localhost:8000
# Choose "Microphone" (or "Tab/System"), select devices, Start.
```
