# Project Rules (This MVP)

## Scope (MUST)
- Speech→speech MVP, **local-only** (no cloud calls).
- Pipeline: **Canary AST (nvidia/canary-1b-v2)** → **Silero TTS**.
- Web app: **FastAPI** + **WebSockets**, Chrome-only client with AudioWorklet for mic PCM.
- Dockerized backend with GPU; A4000 expected. Single-user, no auth.
- I/O: Mic by default; system/meeting audio via OS loopback (Stereo Mix / VB-CABLE).

## Non-goals (NOW)
- No voice cloning. No multi-tenant. No fancy UI. No persistent storage beyond caches.

## Tech Choices (LOCKED)
- AST: NeMo `ASRModel.from_pretrained("nvidia/canary-1b-v2")`, 16 kHz mono input.
- TTS: Silero via `torch.hub.load('snakers4/silero-models','silero_tts', ...)`.
- Audio in browser: 48 kHz float → int16 frames → WS, server downsample to 16 kHz.
- WS protocol: binary = PCM/WAV, text = JSON control/status.
- Chrome cannot capture system audio directly; use loopback devices.

## Definition of Done (DoD)
- `docker compose up` brings API on :8000 with GPU; `GET /` serves frontend.
- Start/Stop works; speaking English returns Russian audio; speaking Russian returns English.
- End-of-utterance handled (idle ~900ms or explicit flush); latency acceptable for demo.
- No network calls to 3rd-party AI; models load locally; attribution noted in docs.
- A short RUNBOOK and AUDIO_ROUTING docs exist.

## Code Quality
- Lint: ruff/flake8 acceptable; basic pre-commit optional.
- Structure under /server: app.py, canary_ast.py, silero_tts.py, audio_util.py, etc.
- Clear logs: model load start/end, inference timings per stage.

## Risk/Fallback
- If Canary load fails, fail fast with actionable error; do **not** add Whisper fallback unless task requests.
- If TTS fails, return transcript text to client as last resort.

## Contributor Workflow
- Each task PR: include a checklist mapping to this DoD; update /memory on decisions.
