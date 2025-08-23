let ws;
let audioContext;
let processor;
let source;
let stream;

const logDiv = document.getElementById("log");
const player = document.getElementById("player");
const inputSelect = document.getElementById("inputSelect");
const outputSelect = document.getElementById("outputSelect");

// Populate device lists
async function populateDevices() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  inputSelect.innerHTML = "";
  outputSelect.innerHTML = "";
  devices.forEach((d) => {
    if (d.kind === "audioinput") {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || `Input ${inputSelect.length + 1}`;
      inputSelect.appendChild(opt);
    } else if (d.kind === "audiooutput") {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || `Output ${outputSelect.length + 1}`;
      outputSelect.appendChild(opt);
    }
  });
}
navigator.mediaDevices.addEventListener("devicechange", populateDevices);
populateDevices();

function log(msg) {
  const p = document.createElement("p");
  p.textContent = msg;
  logDiv.appendChild(p);
}

document.getElementById("startBtn").onclick = async () => {
  const mode = document.getElementById("modeSelect").value;
  ws = new WebSocket(`ws://${location.host}/ws/translate`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    ws.send(JSON.stringify({ mode }));
    log("WebSocket opened");
  };

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      log("JSON: " + event.data);
    } else {
      const blob = new Blob([event.data], { type: "audio/wav" });
      player.src = URL.createObjectURL(blob);
      if (outputSelect.value && typeof player.setSinkId === "function") {
        player.setSinkId(outputSelect.value).then(() => {
          player.play();
        }).catch((err) => {
          log("Error setting output device: " + err);
          player.play();
        });
      } else {
        player.play();
      }
    }
  };

  const constraints = { audio: { deviceId: inputSelect.value ? { exact: inputSelect.value } : undefined } };
  stream = await navigator.mediaDevices.getUserMedia(constraints);
  audioContext = new AudioContext({ sampleRate: 48000 });
  source = audioContext.createMediaStreamSource(stream);

  processor = audioContext.createScriptProcessor(4096, 1, 1);
  source.connect(processor);
  processor.connect(audioContext.destination);

  processor.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    const int16 = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      int16[i] = Math.max(-1, Math.min(1, input[i])) * 0x7fff;
    }
    ws.send(int16.buffer);
  };
};

document.getElementById("stopBtn").onclick = () => {
  if (processor) {
    processor.disconnect();
    source.disconnect();
    processor = null;
  }
  if (ws) {
    ws.send("__flush__");
    log("Sent __flush__, waiting for server response...");
    // Do not close immediately; wait for server to send audio
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  log("Stopped (WS still open until audio received)");
};
