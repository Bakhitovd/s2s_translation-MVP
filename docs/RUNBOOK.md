# RUNBOOK

## Overview
This runbook describes how to build, run, and test the local EN↔RU speech-to-speech MVP.

## Prerequisites
- Windows 11 host with Docker Desktop + WSL2
- NVIDIA GPU (A4000 or similar) with NVIDIA Container Toolkit installed
- Chrome browser

## Build and Run
```bash
cd docker
docker compose up --build
```

This will:
- Build the Docker image with PyTorch + CUDA + NeMo + Silero dependencies
- Mount Hugging Face cache volume under `docker/hf_cache`
- Expose FastAPI server on http://localhost:8000

## Usage
1. Open Chrome and navigate to http://localhost:8000
2. Select translation mode (EN→RU, RU→EN, or Auto)
3. Click **Start** to begin streaming microphone audio
4. Speak into the microphone
5. Click **Stop** to flush the utterance and receive translated audio playback

## Logs
- Server logs are visible in Docker console
- Model load times and inference timings are logged with `uvicorn.error`

## Troubleshooting
- If Canary AST fails to load, check GPU memory availability
- If TTS fails, the server will return transcript text with `"tts_failed": true`
- Ensure Chrome is used (other browsers not supported)
- For system audio capture, configure loopback device (see AUDIO_ROUTING.md)
