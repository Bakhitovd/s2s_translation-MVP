// /static/app.js
'use strict';

let ws;
let audioContext;
let workletNode;
let sourceNode;
let analyser;
let rafId;
let destNode;                 // for routing playback to an <audio> element (so setSinkId works)
let playbackEl;
let inputStream;              // ACTIVE INPUT (mic or display) so we can stop tracks cleanly

let micSelect, speakerSelect, langSelect, startBtn, stopBtn, transcriptsDiv, vuCanvas, vuCtx;
let sampleRateSelect, noiseSuppressionCheckbox, chunkSizeSelect, echoCancellationCheckbox, autoGainCheckbox;
let sourceSelect, muteWhileDisplay;

const WS_URL = `ws://${location.host}/ws/audio`;

function uiSetRunning(running) {
  startBtn.disabled = running;
  stopBtn.disabled = !running;
  micSelect.disabled = running || sourceSelect.value === 'display';
  langSelect.disabled = running;
  sourceSelect.disabled = running;
}

async function ensureAudioPermission() {
  // Ask once so enumerateDevices returns labels
  try {
    const tmp = await navigator.mediaDevices.getUserMedia({ audio: true });
    tmp.getTracks().forEach(t => t.stop());
  } catch {
    // ignore – user can still use display capture
  }
}

async function listDevices() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const mics = devices.filter(d => d.kind === 'audioinput');
  const outs = devices.filter(d => d.kind === 'audiooutput');

  // Mic select
  micSelect.innerHTML = '';
  mics.forEach((d, i) => {
    const opt = document.createElement('option');
    opt.value = d.deviceId;
    opt.textContent = d.label || `Mic ${i + 1}`;
    micSelect.appendChild(opt);
  });

  // Speaker select
  speakerSelect.innerHTML = '';
  outs.forEach((d, i) => {
    const opt = document.createElement('option');
    opt.value = d.deviceId;
    opt.textContent = d.label || `Output ${i + 1}`;
    speakerSelect.appendChild(opt);
  });
}

function parseLangPair() {
  const pair = langSelect.value; // e.g. "ru-en" | "en-ru"
  const [src, tgt] = pair.split('-');
  return { src, tgt, pair };
}

function isDisplayCapture() {
  return sourceSelect?.value === 'display';
}

async function getInputStream(selectedSampleRate, echoCancellationEnabled, noiseSuppressionEnabled, autoGainEnabled) {
  if (!isDisplayCapture()) {
    // Microphone
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
    return navigator.mediaDevices.getUserMedia(constraints);
  }

  // Display (Tab / Window / Screen) with audio
  if (!navigator.mediaDevices.getDisplayMedia) {
    alert('Display capture with audio is not supported in this browser.');
    throw new Error('getDisplayMedia unsupported');
  }

  const displayConstraints = {
    video: true, // required by spec
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      // Chrome hint: reduce local playback of the captured tab to mitigate echo
      suppressLocalAudioPlayback: true
    }
  };

  const stream = await navigator.mediaDevices.getDisplayMedia(displayConstraints);

  // If user didn’t enable sharing audio, there may be no audio tracks
  if (!stream.getAudioTracks().length) {
    alert('No audio captured. In the picker, choose a tab and enable "Share tab audio" (or "Share system audio" on Windows).');
  }

  // Auto-stop when user ends sharing via browser UI
  stream.getTracks().forEach(t => t.addEventListener('ended', () => stop()));

  return stream;
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

  // Acquire input
  inputStream = await getInputStream(selectedSampleRate, echoCancellationEnabled, noiseSuppressionEnabled, autoGainEnabled);
  sourceNode = audioContext.createMediaStreamSource(inputStream);

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

  // Stop input tracks (mic or display)
  if (inputStream) {
    try { inputStream.getTracks().forEach(t => t.stop()); } catch {}
  }

  // Clear playback routing
  if (playbackEl && playbackEl.srcObject) {
    try { playbackEl.srcObject = null; } catch {}
  }

  if (audioContext) {
    try { audioContext.close(); } catch {}
  }

  analyser = sourceNode = workletNode = destNode = audioContext = null;
  inputStream = null;
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

    ws.onclose = () => { ws = null; };
    ws.onerror = (e) => { reject(e); };
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

  // Avoid feedback if we capture this tab or screen and user asked to mute playback
  if (isDisplayCapture() && muteWhileDisplay?.checked) return;

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
  // Only meaningful when mic is the source
  if (isDisplayCapture()) return;
  const running = !!audioContext;
  if (running) {
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
  sourceSelect = document.getElementById('sourceSelect');
  muteWhileDisplay = document.getElementById('muteWhileDisplay');

  sourceSelect.onchange = async () => {
    const running = !!audioContext;
    if (running) {
      stopAudio();
      await initAudio();        // re-init with new source
    }
    // Toggle mic dropdown usability
    micSelect.disabled = sourceSelect.value === 'display';
  };

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
