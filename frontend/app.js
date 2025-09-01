// /static/app.js
'use strict';

let ws;
let audioContext;
let workletNode;
let sourceNode;
let analyser;
let rafId;
let destNode;           // for routing playback to an <audio> element (so setSinkId works)
let playbackEl;
let micSelect, speakerSelect, langSelect, startBtn, stopBtn, transcriptsDiv, vuCanvas, vuCtx;
let sampleRateSelect, noiseSuppressionCheckbox, chunkSizeSelect, echoCancellationCheckbox, autoGainCheckbox;

const WS_URL = `ws://${location.host}/ws/audio`;

function uiSetRunning(running) {
  startBtn.disabled = running;
  stopBtn.disabled = !running;
  micSelect.disabled = running;
  langSelect.disabled = running;
}

async function ensureAudioPermission() {
  // Ask once so enumerateDevices returns labels
  const tmp = await navigator.mediaDevices.getUserMedia({ audio: true });
  tmp.getTracks().forEach(t => t.stop());
}

async function listDevices() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const mics = devices.filter(d => d.kind === 'audioinput');
  const outs = devices.filter(d => d.kind === 'audiooutput');

  // Mic select
  micSelect.innerHTML = '';
  mics.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d.deviceId;
    opt.textContent = d.label || `Mic ${micSelect.length + 1}`;
    micSelect.appendChild(opt);
  });

  // Speaker select
  speakerSelect.innerHTML = '';
  outs.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d.deviceId;
    opt.textContent = d.label || `Output ${speakerSelect.length + 1}`;
    speakerSelect.appendChild(opt);
  });
}

function parseLangPair() {
  const pair = langSelect.value; // e.g. "ru-en" | "en-ru"
  const [src, tgt] = pair.split('-');
  return { src, tgt, pair };
}

async function initAudio() {
  // Get selected parameters from UI
  const selectedSampleRate = parseInt(sampleRateSelect.value, 10) || 48000;
  const noiseSuppressionEnabled = !!noiseSuppressionCheckbox.checked;
  const echoCancellationEnabled = !!echoCancellationCheckbox.checked;
  const autoGainEnabled = !!autoGainCheckbox.checked;
  const selectedChunkMs = parseInt(chunkSizeSelect.value, 10) || 20;

  audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: selectedSampleRate });
  await audioContext.audioWorklet.addModule('/static/worklets/encoder.worklet.js');

  const constraints = {
    audio: {
      deviceId: micSelect.value ? { exact: micSelect.value } : undefined,
      channelCount: 1,
      sampleRate: selectedSampleRate,
      echoCancellation: echoCancellationEnabled,
      noiseSuppression: noiseSuppressionEnabled,
      autoGainControl: autoGainEnabled
    }
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);

  sourceNode = audioContext.createMediaStreamSource(stream);

  // Encoder worklet (do NOT connect to destination → avoids echo)
  workletNode = new AudioWorkletNode(audioContext, 'encoder-worklet', {
    processorOptions: { chunkMs: selectedChunkMs, gateThreshold: 0.02, hangoverMs: 500 }
  });
  sourceNode.connect(workletNode);

  // VU meter tap
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  sourceNode.connect(analyser);
  drawVu();

  // Route playback through a MediaStreamDestination so we can pick output device via <audio>
  destNode = audioContext.createMediaStreamDestination();
  playbackEl.srcObject = destNode.stream;

  // Wire encoder → WS
  workletNode.port.onmessage = (evt) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const data = evt.data;
    if (data && data.type === 'flush') {
      ws.send(JSON.stringify({ type: 'flush' }));
      return;
    }
    const payload = (data instanceof ArrayBuffer)
      ? data
      : (data instanceof Float32Array || data instanceof Int16Array)
        ? data.buffer
        : data;
    ws.send(payload);
  };
}

function stopAudio() {
  try { cancelAnimationFrame(rafId); } catch {}
  try { analyser?.disconnect(); } catch {}
  try { sourceNode?.disconnect(); } catch {}
  try { workletNode?.disconnect(); } catch {}
  try { destNode?.disconnect(); } catch {}
  if (audioContext) {
    // Stop all tracks
    const tracks = audioContext?.destination?.stream?.getTracks?.() || [];
    tracks.forEach(t => t.stop());
    audioContext.close();
  }
  analyser = sourceNode = workletNode = destNode = audioContext = null;
}

function connectWS() {
  return new Promise((resolve, reject) => {
    ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      const { src, tgt, pair } = parseLangPair();
      ws.send(JSON.stringify({ type: 'config', srcLang: src, dstLang: tgt, langPair: pair }));
      resolve();
    };

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'transcript') {
            const p = document.createElement('p');
            p.textContent = msg.text;
            transcriptsDiv.appendChild(p);
            transcriptsDiv.scrollTop = transcriptsDiv.scrollHeight;
          }
        } catch { /* ignore */ }
      } else {
        // Expect 16k mono PCM int16 from server; play it
        playPcmInt16Mono16k(event.data);
      }
    };

    ws.onclose = () => {
      // no-op; UI controls handle state
      ws = null;
    };

    ws.onerror = (e) => {
      reject(e);
    };
  });
}

function closeWS() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    try { ws.close(1000, 'client-stop'); } catch {}
  }
  ws = null;
}

function playPcmInt16Mono16k(ab) {
  if (!audioContext) return;
  const i16 = new Int16Array(ab);
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768.0;

  // Create a buffer tagged as 16k; browser will resample into context's rate
  const buffer = audioContext.createBuffer(1, f32.length, 16000);
  buffer.copyToChannel(f32, 0);
  const src = audioContext.createBufferSource();
  src.buffer = buffer;

  // Route to selectable output
  src.connect(destNode);
  src.start();
}

function drawVu() {
  const w = vuCanvas.width, h = vuCanvas.height;
  const data = new Uint8Array(analyser.fftSize);
  const draw = () => {
    analyser.getByteTimeDomainData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      const v = (data[i] - 128) / 128.0;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / data.length); // 0..~1
    const pct = Math.min(1, rms * 2.5);

    vuCtx.clearRect(0, 0, w, h);
    vuCtx.fillStyle = '#4caf50';
    vuCtx.fillRect(0, 0, Math.floor(pct * w), h);

    rafId = requestAnimationFrame(draw);
  };
  draw();
}

async function applyOutputDevice() {
  if (!('setSinkId' in HTMLMediaElement.prototype)) return;
  if (!speakerSelect.value) return;
  try {
    await playbackEl.setSinkId(speakerSelect.value);
  } catch (e) {
    console.warn('setSinkId failed or not allowed:', e);
  }
}

async function start() {
  uiSetRunning(true);
  transcriptsDiv.textContent = '';
  await initAudio();
  await connectWS();
  await applyOutputDevice();
}

function stop() {
  closeWS();
  stopAudio();
  uiSetRunning(false);
}

async function onMicChange() {
  const running = !!audioContext;
  if (running) {
    // Re-init audio path with new device; keep the same WS session
    stopAudio();
    await initAudio();
  }
}

async function onSpeakerChange() {
  await applyOutputDevice();
}

window.addEventListener('DOMContentLoaded', async () => {
  // Grab elements
  transcriptsDiv = document.getElementById('transcripts');
  micSelect = document.getElementById('micSelect');
  speakerSelect = document.getElementById('speakerSelect');
  langSelect = document.getElementById('langSelect');
  startBtn = document.getElementById('startBtn');
  stopBtn = document.getElementById('stopBtn');
  playbackEl = document.getElementById('playback');
  vuCanvas = document.getElementById('vuCanvas');
  vuCtx = vuCanvas.getContext('2d');
  sampleRateSelect = document.getElementById('sampleRateSelect');
  noiseSuppressionCheckbox = document.getElementById('noiseSuppressionCheckbox');
  echoCancellationCheckbox = document.getElementById('echoCancellationCheckbox');
  autoGainCheckbox = document.getElementById('autoGainCheckbox');
  chunkSizeSelect = document.getElementById('chunkSizeSelect');

  // Permissions + devices
  await ensureAudioPermission();
  await listDevices();

  // Handlers
  startBtn.onclick = start;
  stopBtn.onclick = stop;
  micSelect.onchange = onMicChange;
  speakerSelect.onchange = onSpeakerChange;

  // Clean shutdown on page exit
  window.addEventListener('beforeunload', stop);
});
