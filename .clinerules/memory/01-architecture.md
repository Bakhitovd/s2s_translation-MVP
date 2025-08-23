# Architecture (High-Level)

Browser (Chrome):
  - getUserMedia (mic or loopback device) → AudioWorklet → int16 frames over WS
  - receive WAV blobs (one per utterance) → play via <audio> or AudioContext

Server (FastAPI in Docker):
  - WS endpoint: `ws://localhost:8000/ws/translate`
  - Buffer PCM until idle ~900ms or explicit `"__flush__"` control message
  - Downsample 48 kHz → 16 kHz mono for Canary AST ingestion
  - Use ModelManager (Canary AST) and TTSManager (Silero) classes to manage lifecycle and make mocking in tests straightforward
  - TTS: keep `speaker = None` in the protocol/server code so SileroTTS chooses a sensible default speaker per language (package default / first available). This reduces brittle speaker-id mappings and matches what is wired in server/silero_tts.py.
  - TTS failure behavior: if Silero synthesis fails, server will fall back to returning the transcript text to the client (no audio), with a clear status message so frontend can surface fallback behavior.
  - Server responses: WAV blob per utterance plus accompanying metadata JSON { src_lang, tgt_lang, text, timings?, inference_time_ms }

Models / runtime:
  - Canary AST (NeMo): `nvidia/canary-1b-v2` — expects 16 kHz mono input, returns translated text. Loaded via ModelManager.
  - Silero TTS: selected per language by TTSManager; `speaker=None` allows the library/package to pick the default speaker for the model. Sample rates vary (8/16/48k) — server resamples/normalizes as needed.
  - Both managers expose load() and inference methods to allow stubbing/mocking in unit tests and CI (no heavy model downloads in tests).

Infra:
  - Image: pytorch + CUDA runtime
  - GPU access: `--gpus all` (NVIDIA Container Toolkit)
  - Cache HF models under mounted volume: `/root/.cache/huggingface`
  - Frontend served from `/frontend` (Chrome-only client)

Notes:
  - Keep server wiring minimal and local-only: avoid extra configuration complexity; prefer safe defaults (speaker=None) and explicit fallbacks for production robustness.
  - Tests must mock ModelManager and TTSManager load/inference methods; CI must not download large models.
