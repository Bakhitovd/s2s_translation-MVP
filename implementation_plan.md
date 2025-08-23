# Implementation Plan

[Overview]
Single sentence describing the overall goal.

Deliver a local, utterance-based EN↔RU speech→speech MVP that wires the Canary AST model to Silero TTS via a FastAPI WebSocket backend, served in Docker with GPU support, and a minimal Chrome frontend for mic/loopback capture and playback.

This implementation fills the gap between the existing project documentation and a runnable codebase by providing exact file-level changes, typed interfaces, concrete function and class responsibilities, Docker integration details, and tests. The scope adheres to the project's locked tech choices: NeMo Canary AST (nvidia/canary-1b-v2) for AST, Silero via torch.hub for TTS, a FastAPI server exposing a duplex WebSocket at /ws/translate, AudioWorklet in Chrome to stream 48 kHz int16 audio frames, server-side resampling to 16 kHz mono, and utterance-level flushing (idle ~900ms or explicit "__flush__"). The approach minimizes complexity by implementing utterance-based end-of-utterance detection and clear fallbacks (return transcript text if TTS fails).

[Types]  
Single sentence describing the type system changes.

Introduce a small, explicit set of typed data structures for WebSocket control messages, AST results, TTS payloads, and audio envelopes to make the pipeline deterministic and testable.

Detailed type definitions:

- WsControl (JSON control message sent by client at WS open)
  - src: "auto" | "en" | "ru"  (string) — source language preference; default "auto"
  - tgt: "en" | "ru" | null     (string|null) — explicit target language; when null and src != "auto", infer opposite
  - mode: "auto" | "en->ru" | "ru->en"  (string) — frontend selects this; fallback to src/tgt pair
  - Example validation rules:
    - If mode is provided, it overrides src/tgt.
    - If src == "auto" and mode omitted, server attempts language detection heuristic (start with en->ru).

- WsAstResult (server->client JSON after processing)
  - src: "en" | "ru"  (string) — detected or used source language
  - tgt: "en" | "ru"  (string) — used target language
  - text: string      — translated text returned by Canary AST
  - timestamps: optional Array<{start: float, end: float, text: string}> — included if timestamped transcription used (future extension)

- AudioEnvelope (binary WebSocket frames)
  - Binary frames: raw Int16 PCM chunks produced by AudioWorklet (48 kHz, interleaved if >1 channel)
  - Server expects to receive concatenated PCM Int16 bytes and will treat them as 48 kHz stereo (client chooses device channels)

- InternalModelResult (internal)
  - text: string
  - segments?: same shape as timestamps above
  - meta: { inference_time_ms: number, model_load_ms?: number }

- WAVBlob (server->client binary)
  - Standard WAV file bytes (PCM_16 subtype) at TTS sample rate (8k/16k depending on chosen speaker)
  - Metadata sent separately via WsAstResult JSON

Validation rules:
- WS JSON messages must be small and validated on receipt; unknown fields ignored.
- Binary frames must be multiples of 2 bytes (int16); server will drop malformed frames and log an error.

[Files]
Single sentence describing file modifications.

Create the canonical server/frontend/docker files described in docs/init_8-23-2025.md and add strict entrypoints, while modifying only necessary files at the repository root.

Detailed breakdown:

- New files to be created (full paths and purpose)
  - server/app.py
    - FastAPI app, static mount for frontend, WS endpoint /ws/translate, main orchestration loop and utterance handling (handle_utterance).
  - server/canary_ast.py
    - Model loader and wrapper functions for Canary AST: load_canary(), ast_translate().
  - server/silero_tts.py
    - TTS loader/wrapper: load_tts(), synth().
  - server/audio_util.py
    - Utilities: to_mono16k_float32(bytes, in_sr=48000, channels=2), wav_bytes_from_float32(np_array, sr), and small helpers to validate PCM frames.
  - server/requirements.txt
    - Pin minimal dependencies required by the plan (see Dependencies section).
  - frontend/index.html
    - Minimal HTML UI (Start/Stop, Mode select, log).
  - frontend/app.js
    - Browser side: AudioWorklet, PCM capture, WS client; explicit __flush__ on Stop.
  - docker/Dockerfile
    - GPU-enabled image building PyTorch, NeMo and system deps (ffmpeg, libsndfile1).
  - docker/docker-compose.yml
    - Compose file mapping ports and volume for HF cache and enabling GPU devices.
  - docs/RUNBOOK.md
    - Short runbook and audio routing instructions (extract from docs/init_8-23-2025.md).
  - docs/AUDIO_ROUTING.md
    - System/loopback setup notes.
  - implementation_plan.md
    - This file (already created).
  - tests/test_audio_util.py
    - Unit tests for resampling and WAV framing helpers.
  - tests/test_ws_protocol.py
    - Integration-level tests for WS message validation and utterance handling (mocking model wrappers).

- Existing files to be modified (with specific changes)
  - README.md
    - Add "How to run" short commands referencing docker/docker-compose.yml and add attribution for Canary-1b-v2 and Silero TTS.
  - .gitignore
    - Ensure typical additions: /docker/hf_cache, /server/__pycache__/, .venv/
  - docs/init_8-23-2025.md
    - No functional change; keep as canonical design doc (reference only).

- Files to be deleted or moved
  - None in current plan. Keep diffs additive.

- Configuration file updates
  - Add server/requirements.txt with pinned package versions.
  - docker/Dockerfile and docker/docker-compose.yml to be added under docker/.
  - Optionally add a minimal .env.example with TORCH_CUDA_ARCH_LIST if needed.

[Functions]
Single sentence describing function modifications.

Introduce a small set of new functions mapped to the files above with explicit signatures; modify only orchestration-level helpers in server/app.py to integrate them.

Detailed breakdown:

- New functions (name, signature, file path, purpose)
  - load_canary() -> ASRModel
    - server/canary_ast.py
    - Loads NeMo ASRModel.from_pretrained("nvidia/canary-1b-v2") lazily, logs timings, raises descriptive exceptions on failure.
  - ast_translate(wav_array_or_path, src_lang: str, tgt_lang: str, timestamps: bool=False) -> InternalModelResult
    - server/canary_ast.py
    - Accepts numpy float32 array or path; returns text and optional segments (timestamps). Wraps and normalizes model outputs.
  - load_tts(lang: str, speaker: str) -> (model, symbols, sr, apply_tts, device)
    - server/silero_tts.py
    - Caches model instances per (lang, speaker), moves model to CUDA if available.
  - synth(text: str, lang: str='ru', speaker: str='baya') -> (sr: int, audio_np: np.ndarray)
    - server/silero_tts.py
    - Returns numpy float32 array for WAV writing.
  - to_mono16k_float32(pcm_bytes: bytes, in_sr: int=48000, channels: int=2) -> (sr:int, audio_np: np.ndarray)
    - server/audio_util.py
    - Converts raw int16 bytes into mono 16k float32 numpy array.
  - wav_bytes_from_float32(audio_np: np.ndarray, sr: int) -> bytes
    - server/audio_util.py
    - Writes WAV (PCM_16) bytes into memory and returns bytes.
  - validate_ws_control(payload: dict) -> WsControl
    - server/app.py (or server/utils)
    - Validates and normalizes initial WS JSON config.
  - handle_utterance(ws: WebSocket, buf: bytearray, mode: str)
    - server/app.py
    - Orchestrates resampling, AST call, TTS call, and sending of WAV + metadata. Should be async and non-blocking; CPU-bound TTS can be awaited (or dispatched to threadpool if long).
  - run_uvicorn() -> None (entrypoint helper)
    - docker entry invoked as uvicorn server.app:app --host 0.0.0.0 --port 8000.

- Modified functions (exact name, current file path, required changes)
  - None pre-existing in repo; the plan creates the server code. If README contains example server run, update the example to match docker paths (edit README.md).

- Removed functions
  - None.

[Classes]
Single sentence describing class modifications.

Introduce lightweight manager classes to encapsulate model lifecycles and caching, improving testability and separation of concerns.

Detailed breakdown:

- New classes (name, file path, key methods, inheritance)
  - ModelManager (server/canary_ast.py)
    - Methods:
      - __init__(self)
      - load(self) -> None
      - translate(self, audio_or_path, src_lang, tgt_lang, timestamps=False) -> InternalModelResult
    - Purpose: wrap global model state, manage lazy load, log timings, provide deterministic interface for tests.
  - TTSManager (server/silero_tts.py)
    - Methods:
      - __init__(self)
      - load(self, lang, speaker) -> None
      - synth(self, text, lang, speaker) -> (sr, np.array)
    - Purpose: cache and manage Silero TTS instances; handle device placement.
  - AudioBuffer (server/app.py or server/audio_util.py)
    - Methods:
      - __init__(self)
      - append(self, bytes_chunk: bytes)
      - flush(self) -> bytes
      - length(self) -> int
    - Purpose: small utility around bytearray for clearer semantics and testing.

- Modified classes
  - None.

- Removed classes
  - None.

[Dependencies]
Single sentence describing dependency modifications.

Pin and add explicit Python dependencies required for NeMo, Silero, FastAPI WebSockets, audio processing and test tooling; keep versions compatible with GPU containers.

Details of packages and versions (suggested):
- fastapi==0.115.0
- uvicorn[standard]==0.30.0
- websockets==12.0
- soundfile==0.12.1
- numpy==1.26.4
- scipy==1.13.1
- torchaudio==2.3.1
- torch==2.3.1+cu12.x (installed via appropriate wheel in Dockerfile base image)
- nemo_toolkit[asr]==2.x compatible with CUDA 12.1 (installed with pip in Dockerfile as shown in docs)
- pytest==8.4.3 (or latest 8.x) — test runner
- pytest-asyncio==0.22.0 — test async WS handlers
- (dev) black/ruff optionally for linting

Integration requirements:
- Docker base image must match CUDA version for NeMo and PyTorch (docs suggest pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime).
- Docker Compose must expose a volume for Hugging Face cache (hf_cache) to avoid repeated downloads.
- The container must run with --gpus all (NVIDIA Container Toolkit) enabled.

[Testing]
Single sentence describing testing approach.

Combine unit tests for audio utilities and type validation with small integration tests that mock model wrappers and exercise the WebSocket handler, plus a simple manual end-to-end smoke test run under Docker.

Test file requirements and strategies:
- Unit tests
  - tests/test_audio_util.py
    - Validate to_mono16k_float32 behavior with synthetic waveforms (48k stereo sine wave -> 16k mono)
    - Validate wav_bytes_from_float32 produces readable WAV with soundfile
  - tests/test_silero_wrapper.py (mocked)
    - Mock torch.hub.load to return a fake apply_tts; verify synth() returns expected shape and dtype
- Integration tests (async)
  - tests/test_ws_protocol.py
    - Use pytest-asyncio and websockets or FastAPI TestClient with WebSocket support to simulate a client:
      - Connect, send JSON control message, send a small binary chunk, send "__flush__", assert that server sends back JSON and binary WAV blob
    - Mock canary_ast.ast_translate and silero_tts.synth to deterministic values to avoid heavy model loads during CI
- Manual smoke test (runbook)
  - docker/docker-compose.yml build & up; open http://localhost:8000 and follow UI steps; verify speech->translated audio.
- Acceptance tests (optional)
  - A short CLI script under tests/cli_smoke.py that posts a pre-recorded WAV to handle_utterance and verifies outputs (not executed in CI due to model heaviness).

[Implementation Order]
Single sentence describing the implementation sequence.

Implement incrementally: core audio utilities and WS protocol first, stubbed model wrappers and tests second, then integrate real model loaders, Dockerize, and finish with frontend and runbook.

Numbered steps:

1. Create server directory and core utilities
   - Add server/audio_util.py with to_mono16k_float32 and wav_bytes_from_float32
   - Add unit tests tests/test_audio_util.py
   - Run unit tests locally (pytest)

2. Implement WS protocol skeleton
   - Add server/app.py with WebSocket endpoint, AudioBuffer utility, validate_ws_control, and an async handle_utterance that calls placeholder model wrappers (return deterministic text)
   - Add tests/test_ws_protocol.py mocking model calls to verify binary and JSON messages

3. Add model wrapper scaffolding (non-blocking)
   - Add server/canary_ast.py with ModelManager stub that raises a descriptive error if load is attempted (so tests can mock it)
   - Add server/silero_tts.py with TTSManager stub returning a short sine wave; tests verify synth shape

4. Wire end-to-end behavior locally using stubs
   - Start FastAPI app via uvicorn and test with frontend (will play stubbed audio)
   - Fix any WS quirks (binary/text framing, flush behavior)

5. Replace stubs with real model loaders
   - Implement load_canary() calling ASRModel.from_pretrained and ast_translate() that accepts numpy arrays; add careful logging around model load time
   - Implement load_tts() using torch.hub.load and synth() that returns float32 numpy arrays; ensure device placement and caching

6. Dockerize
   - Add docker/Dockerfile and docker/docker-compose.yml per docs/init_8-23-2025.md; ensure requirements.txt is used
   - Build and run image with GPU support and HF cache volume
   - Validate canary weights are cached under volume

7. Frontend & AudioWorklet
   - Add frontend/index.html and frontend/app.js as per docs
   - Test mic capture, AudioWorklet framing and explicit "__flush__" behavior; verify playback of server WAV blobs

8. Runbook and docs
   - Add docs/RUNBOOK.md and docs/AUDIO_ROUTING.md; update README.md with "docker compose up" commands and attributions

9. Performance and logging improvements
   - Add inference timing logs in ModelManager and TTSManager
   - Add basic health check/GET / returning a small HTML page (already served from frontend index)

10. Final acceptance
    - Run full stack under Docker on machine with GPU (A4000), verify EN→RU and RU→EN demo flows, measure latency, confirm no external network calls in inference paths, and ensure DoD items are satisfied.

Notes on edge cases & constraints:
- Canary model loading may fail due to insufficient memory; ModelManager must catch exceptions, log, and return a clear error to the client (e.g., JSON {error: "..."}).
- TTS synthesis failure should not block the client: return JSON with text and an optional "tts_failed": true flag; client should play nothing or fallback to text-to-speech via client speechSynthesis (browser) in worst case.
- Keep web frontend minimal and Chrome-only; provide clear troubleshooting in RUNBOOK for loopback audio.
