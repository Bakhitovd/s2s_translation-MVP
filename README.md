# EN↔RU Real-Time Speech-to-Speech MVP

Local, offline, real-time English↔Russian speech translation demo.  
**No cloud.** Canary AST (NVIDIA) + Silero TTS.  
FastAPI backend, WebSocket streaming, Chrome-only frontend.  
Runs on Windows with CUDA GPU (RTX A4000+ recommended).

---

## Quick Start

### 1. Prerequisites

- Windows 11, CUDA-capable GPU (RTX A4000+)
- Docker (recommended) or Python 3.10+ with CUDA toolkit
- Chrome browser

### 2. Run with Docker

```sh
docker compose up --build
```

Or, build and run manually:

```sh
cd docker
docker build -t s2s-mvp .
docker run --gpus all -p 8000:8000 s2s-mvp
```

### 3. Run Locally (Dev)

```sh
pip install -r server/requirements.txt
# Install torch+cuda matching your GPU (see Dockerfile for version)
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 4. Open the Frontend

- Visit [http://localhost:8000](http://localhost:8000) in Chrome.
- Allow microphone access.
- Select mic, output device, and translation direction.
- **Choose Mode:**  
  - **ASR+MT (default):** Canary ASR → M2M100 MT → Silero TTS  
  - **AST (direct):** Canary AST (speech-to-translation) → Silero TTS
- Click **Start** to begin streaming.
- Speak; translated audio will play through the selected output.
- Live transcripts update below.

---

## Features

- Real-time EN↔RU speech translation (Canary AST + Silero TTS, or ASR+MT pipeline)
- All processing local/offline (no cloud)
- Chrome-only frontend: device selection, VU meter, live transcripts, mode selector
- WebSocket `/ws/audio` protocol: JSON control + raw 16kHz mono Int16LE PCM
- Dockerized for reproducibility

---

## Pipeline Modes

**Mode Selector:**  
In the frontend, select the pipeline mode:
- **ASR+MT (default):**  
  - Step 1: Canary runs in ASR mode (speech-to-text, source language)  
  - Step 2: M2M100 translates the transcript to the target language  
  - Step 3: Silero TTS synthesizes speech in the target language  
  - The UI shows both the ASR transcript (italic, grey) and the final translation.
- **AST (direct):**  
  - Canary runs in AST mode (direct speech-to-translation)  
  - Silero TTS synthesizes the translated speech  
  - The UI shows only the translated text.

**Language Codes:**  
- Supported: `"en"` (English), `"ru"` (Russian)
- Direction: set via the "Direction" dropdown (RU→EN or EN→RU)
- Both modes require explicit source/target language selection.

---

## Troubleshooting

- **AudioWorklet not supported:** Use latest Chrome.
- **No audio output:** Check device selection and permissions.
- **High latency:** Ensure CUDA is available and models are loaded.
- **404 for /favicon.ico:** Non-critical, can be ignored.

---

## Project Structure

- `server/app.py` — FastAPI app, WebSocket, static frontend
- `server/audio_util.py` — Resampling, PCM utils
- `server/canary_ast.py`, `server/silero_tts.py`, `server/mt.py` — Model wrappers
- `frontend/` — index.html, app.js, styles.css, worklets/
- `docker/` — Dockerfile, compose, CUDA setup

---

## License

MIT
