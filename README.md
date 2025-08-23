# s2s_translation-MVP

## Overview
Local EN↔RU speech-to-speech MVP.  
Implements utterance-based streaming (idle ~900ms or explicit `__flush__`) with a FastAPI WebSocket backend, Canary AST (`nvidia/canary-1b-v2`) for ASR/translation, and Silero TTS for synthesis.  
Frontend is a minimal Chrome-only UI for microphone / loopback capture and playback.  
Runs locally in Docker with GPU support.

## How to Run
1. Ensure Docker Desktop is running with GPU support (NVIDIA Container Toolkit).
2. From the project root:
```bash
cd docker
docker compose up --build
```
3. Open http://localhost:8000 in Chrome.

## Docs
- RUNBOOK: `docs/RUNBOOK.md` — operational instructions and manual acceptance test.
- Audio routing: `docs/AUDIO_ROUTING.md` — Windows loopback (Stereo Mix / VB-CABLE) guidance.

## File Structure
- `server/` — FastAPI backend, audio utilities, model wrappers (ModelManager, TTSManager)
- `frontend/` — Minimal HTML/JS client
- `docker/` — Dockerfile and docker-compose.yml
- `docs/` — RUNBOOK and AUDIO_ROUTING instructions
- `.clinerules/` — project rules, memory, and implementation constraints
- `tests/` — Unit and integration tests (heavy models are mocked)

## Models / Runtime Notes
- ASR/Translation: NeMo Canary AST — `nvidia/canary-1b-v2` (expects 16 kHz mono input)
- TTS: Silero TTS via package wrapper; server keeps `speaker = None` so the Silero package chooses a sensible default speaker per language. If TTS synthesis fails, server falls back to returning transcript text to the client.
- Models are cached under the compose-mounted volume `/root/.cache/huggingface`.

## Development & Testing
- Tests mock ModelManager and TTSManager to avoid downloading heavy models in CI.
- Use the provided sample audio in `audio_sample/` for local verification.
- For local manual testing, follow the RUNBOOK acceptance test steps.

## Changelog (recent)
- 2025-08-23: Added RUNBOOK and clarified audio routing. Updated `.clinerules/memory/01-architecture.md` to require `speaker=None` defaults and to document model manager usage.

## Attribution
- **Canary AST**: nvidia/canary-1b-v2 (NeMo Toolkit)
- **Silero TTS**: snakers4/silero-models
