# Next Steps — Web MVP (EN↔RU Real-Time Speech→Speech)

This document captures the plan for evolving the MVP from its current state to the full real-time streaming design described in the 8‑25‑2025 spec.

---

## Current State

- **Server**
  - `canary_ast.py`: wraps NVIDIA Canary v2 (AST).
  - `silero_tts.py`: wraps Silero TTS, outputs ~24 kHz float32.
  - `audio_util.py`: has 48k→16k resampler and WAV encoder.
  - **Missing**: `server/app.py` with FastAPI + WebSocket `/ws/audio`.
- **Frontend**
  - `index.html` + `app.js`: legacy MVP using ScriptProcessorNode, 48 kHz, `/ws/translate`, WAV blob playback.
  - No AudioWorklet encoder, no real-time PCM streaming, no device selection UI.
- **Docs**
  - `next_step_8-25-2025.md` provides reference implementation for server/app.py, encoder.worklet.js, and updated frontend.

---

## Gaps

1. No FastAPI app or WebSocket endpoint (`/ws/audio`).
2. Frontend encoder missing (no AudioWorklet, no 16 kHz resampling).
3. Playback is blob-based, not streaming PCM.
4. Protocol mismatch: current frontend uses `/ws/translate` + `__flush__`; spec requires `/ws/audio` with JSON control + raw PCM both ways.
5. TTS sample rate mismatch: Silero outputs 24 kHz, must resample to 16 kHz before sending.
6. Device selection and tab/system capture not implemented.

---

## Plan

### 1. Backend: FastAPI WebSocket
- Create `server/app.py`:
  - Mount `frontend/` as static.
  - WebSocket `/ws/audio`.
  - Lazy-init global Canary AST + Silero TTS.
  - Handle JSON control messages and binary PCM frames.
  - Processing loop: segment audio, run AST → TTS, resample to 16 kHz, send Int16LE PCM back, plus transcript JSON.

### 2. Backend Utilities
- Extend `audio_util.py`:
  - `resample_float32(audio, from_sr, to_sr)`.
  - `float32_to_int16le_bytes(f32)`.

### 3. Frontend: Encoder Worklet + Orchestration
- Add `frontend/worklets/encoder.worklet.js`:
  - Downmix stereo → mono, resample to 16 kHz, chunk 20–30 ms, post Int16LE bytes.
- Replace `frontend/app.js`:
  - Device enumeration, mic/tab capture, AudioWorklet pipeline.
  - WebSocket `/ws/audio`, send control JSON, handle transcripts, play PCM chunks.
- Replace `frontend/index.html`:
  - Controls for source, mic, speaker, langs, VAD, start/stop/PTT.
  - Transcript panes, VU meter, hidden `<audio>` for output routing.
- Add minimal `styles.css`.

### 4. Requirements & Docker
- Verify `requirements.txt` includes `fastapi`, `uvicorn`, `numpy<2`, `scipy`, `soundfile`, `pydub`.
- Torch pinned in Dockerfile.
- Ensure Dockerfile exposes port 8000 and runs `uvicorn server.app:app`.

### 5. Testing
- Run: `uvicorn server.app:app --host 0.0.0.0 --port 8000`.
- Open `http://localhost:8000/`.
- Test mic and tab/system capture.
- Validate:
  - Continuous streaming audio.
  - Transcripts update.
  - Output device selection works.
  - Latency ≤ 2–3 s per utterance.

### 6. Metrics & Backpressure
- Track queue drops.
- Optionally send `{type:"stats"}` events with latency and queue depth.

---

## Deliverables

- `server/app.py` with `/ws/audio`.
- Updated `server/audio_util.py`.
- `frontend/worklets/encoder.worklet.js`.
- Updated `frontend/app.js` and `index.html`.
- `frontend/styles.css`.
- Updated README quick start.

---

## Definition of Done

- Visiting `http://localhost:8000` shows new UI.
- Selecting mic or tab/system starts real-time streaming.
- Hear translated audio routed to selected output device.
- Transcripts update live.
- Latency ~2–3 s per utterance on RTX A4000 with CUDA.
- Binary frames are raw 16 kHz mono Int16LE PCM (no WAV blobs).
- Stable under continuous speech for several minutes.

---

## Notes

- Use `InternalModelResult.text` from Canary AST.
- Resample Silero output to 16 kHz before sending.
- Chunk size ~20–30 ms (320–480 samples).
- If AudioWorklet unavailable, disable start button with message.
- Default langs: ru→en; configurable via UI.

---

## Next Action

Implement the above in small, reversible steps:
1. Add `server/app.py`.
2. Extend `audio_util.py`.
3. Add encoder worklet.
4. Replace frontend files.
5. Update README.
6. Test end-to-end.
