# Pipeline Tuning Report: EN↔RU Speech-to-Speech MVP

This report describes all tunable parameters in the MVP pipeline, their effects, and how to adjust them to reduce noise or improve quality.

---

## 1. Frontend (Audio Capture & Encoding)

**File(s):** frontend/app.js, frontend/worklets/encoder.worklet.js, frontend/index.html

### Tunable Parameters

- **Input Device (micSelect):**  
  Selectable via UI. Choose the best quality microphone or system input.
- **Output Device (speakerSelect):**  
  Selectable via UI. Route playback to desired output.
- **Language Direction (langSelect):**  
  RU→EN or EN→RU. Affects ASR and TTS models.
- **AudioContext Sample Rate:**  
  Fixed at 48000 Hz (can be changed in app.js, but must match worklet and backend expectations).
- **getUserMedia Constraints:**  
  - `channelCount`: 1 (mono)
  - `sampleRate`: 48000
  - `echoCancellation`, `noiseSuppression`, `autoGainControl`: all true (can be toggled in app.js for more/less aggressive noise filtering).
- **Encoder Worklet:**
  - `targetSampleRate`: 16000 Hz (hardcoded, matches backend/model)
  - `chunkSize`: 20 ms (320 samples at 16kHz, can be changed for latency/quality tradeoff)
  - **Downmixing:** All channels averaged to mono.
- **VU Meter:**  
  `analyser.fftSize = 2048` (can be tuned for VU meter responsiveness).

---

## 2. WebSocket Streaming & Protocol

**File(s):** frontend/app.js, server/app.py

### Tunable Parameters

- **WebSocket URL:**  
  `ws://<host>/ws/audio` (fixed)
- **Control Messages:**  
  - `type: "config"`: srcLang, dstLang, langPair
  - Protocol can be extended for pause, resume, flush, etc.
- **PCM Chunk Size:**  
  Determined by encoder worklet chunking (20 ms by default).
- **Segment Size (Backend):**  
  `SEGMENT_MS = 1000` (1s, in server/app.py). Lower for lower latency, higher for stability.

---

## 3. Backend (Segmentation, ASR, TTS, Resampling)

**File(s):** server/app.py, server/audio_util.py, server/canary_ast.py, server/silero_tts.py

### Tunable Parameters

- **Segmentation:**
  - `SEGMENT_MS`: 1000 ms (can be reduced for lower latency, at the cost of more frequent model calls)
  - `SAMPLE_RATE`: 16000 Hz (fixed for model compatibility)
- **Canary AST (ASR+Translation):**
  - `src_lang`, `tgt_lang`: Set by frontend
  - `timestamps`: Can be enabled for segment alignment (not exposed in UI)
  - **Model:** "nvidia/canary-1b-v2" (can be changed in canary_ast.py)
- **Silero TTS:**
  - `sample_rate`: 24000 Hz (set in TTSManager, can be changed)
  - `device`: "cpu" or "cuda" (set in TTSManager)
  - `lang`: "ru" or "en" (set by backend)
  - `speaker`: "aidar" (RU), "en_0" (EN), can be overridden in code
  - **Model_id:** "v4_ru" (RU), "v3_en" (EN), can be changed for other Silero models
- **Resampling:**
  - All TTS output is resampled to 16kHz mono float32 before sending to frontend
  - Resampling uses scipy's `resample_poly` (quality can be tuned by changing up/down factors or using a different method)
- **PCM Conversion:**
  - float32 audio is clipped to [-1, 1] and scaled to Int16LE

---

## 4. Playback & Output

**File(s):** frontend/app.js, frontend/index.html

### Tunable Parameters

- **Output Device:**  
  Selectable via UI (`setSinkId` on `<audio>` element)
- **Playback Buffering:**  
  Not explicitly buffered; browser resamples 16kHz PCM to AudioContext rate

---

## 5. Additional Notes

- **Noise Suppression:**  
  Try toggling `noiseSuppression` and `autoGainControl` in getUserMedia constraints for different environments.
- **Latency vs. Quality:**  
  Lowering `SEGMENT_MS` and `chunkSize` reduces latency but may increase CPU usage and instability.
- **Model Selection:**  
  Canary and Silero models can be swapped for different quality/speed tradeoffs.
- **Device:**  
  Running Silero TTS on "cuda" (if available) can reduce synthesis latency.

---

## How to Tune

- **Reduce Noise:**  
  - Use a high-quality mic and select it in the UI.
  - Enable/disable `noiseSuppression` and `autoGainControl` to see which works best for your setup.
  - Lower input gain if clipping occurs (watch VU meter).
- **Reduce Latency:**  
  - Lower `SEGMENT_MS` in server/app.py (e.g., 500 ms).
  - Lower `chunkSize` in encoder.worklet.js (e.g., 10 ms).
  - Use "cuda" for Silero TTS if available.
- **Improve Quality:**  
  - Try different Silero speakers (edit server/silero_tts.py).
  - Use the latest Canary/Silero models.
  - Tune resampling method in audio_util.py if artifacts are present.

---

## Summary Table

| Parameter                | Location                | Effect                        | How to Change                |
|--------------------------|-------------------------|-------------------------------|------------------------------|
| Input Device             | UI (micSelect)          | Input quality                 | Select in UI                 |
| Output Device            | UI (speakerSelect)      | Playback routing              | Select in UI                 |
| Language Direction       | UI (langSelect)         | ASR/TTS language              | Select in UI                 |
| Sample Rate (Frontend)   | app.js                  | Input/encoding quality        | Edit AudioContext init       |
| Noise Suppression        | app.js                  | Input noise                   | Edit getUserMedia constraints|
| Chunk Size (Frontend)    | encoder.worklet.js      | Latency, CPU                  | Edit chunkSize               |
| Segment Size (Backend)   | server/app.py           | Latency, CPU                  | Edit SEGMENT_MS              |
| TTS Device               | server/silero_tts.py    | Synthesis speed               | Edit TTSManager(device=...)  |
| TTS Speaker              | server/silero_tts.py    | Voice quality                 | Edit speaker param           |
| Resampling Method        | audio_util.py           | Audio artifacts               | Edit resample_float32        |
