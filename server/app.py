import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("s2s.server")

app = FastAPI()

# Canary AST model manager (lazy global)
canary_mgr = None

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/")
async def root():
    index_path = os.path.join(frontend_path, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# Mount the frontend directory as static files at /static, serve index.html at "/"
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    import json
    import numpy as np
    from server.audio_util import resample_float32, float32_to_int16le_bytes

    await websocket.accept()
    log.info("WebSocket connection opened: %s", websocket.client)
    config = None
    pcm_buffer = bytearray()
    total_bytes = 0
    try:
        SEGMENT_MS = 1000  # 1 second segments for demo
        SAMPLE_RATE = 16000
        BYTES_PER_SAMPLE = 2  # Int16LE
        SEGMENT_SIZE = SAMPLE_RATE * BYTES_PER_SAMPLE * SEGMENT_MS // 1000

        while True:
            try:
                msg = await websocket.receive()
            except RuntimeError:
                log.info("WebSocket disconnect detected (client: %s)", websocket.client)
                break
            if "bytes" in msg:
                chunk = msg["bytes"]
                pcm_buffer.extend(chunk)
                total_bytes += len(chunk)
                log.debug("Received PCM chunk: %d bytes (total: %d)", len(chunk), total_bytes)
                # Segment by fixed size (1s for demo)
                while len(pcm_buffer) >= SEGMENT_SIZE:
                    segment = pcm_buffer[:SEGMENT_SIZE]
                    del pcm_buffer[:SEGMENT_SIZE]
                    log.info("Segment ready: %d bytes (%.2fs)", len(segment), len(segment) / (SAMPLE_RATE * BYTES_PER_SAMPLE))
                    # Convert Int16LE PCM to float32 [-1,1]
                    audio_np = np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0
                    # Lazy-load Canary AST
                    global canary_mgr
                    if canary_mgr is None:
                        from server.canary_ast import ModelManager
                        canary_mgr = ModelManager()
                        canary_mgr.load()
                        log.info("Canary AST model loaded")
                    src_lang = config.get("srcLang", "ru") if config else "ru"
                    tgt_lang = config.get("dstLang", "en") if config else "en"
                    log.info("Calling Canary AST: src=%s tgt=%s audio=%.2fs", src_lang, tgt_lang, len(audio_np)/SAMPLE_RATE)
                    result = canary_mgr.translate(audio_np, src_lang, tgt_lang)
                    log.info("Canary AST result: %r", result.text)
                    # Run Silero TTS
                    global tts_mgr
                    if "tts_mgr" not in globals() or tts_mgr is None:
                        from server.silero_tts import TTSManager
                        tts_mgr = TTSManager(sample_rate=24000, device="cpu")
                        log.info("Silero TTS manager loaded")
                    tts_lang = tgt_lang
                    tts_text = result.text
                    log.info("Calling Silero TTS: lang=%s text=%r", tts_lang, tts_text)
                    tts_sr, tts_audio = tts_mgr.synth(tts_text, tts_lang)
                    log.info("Silero TTS result: %d samples @ %d Hz", len(tts_audio), tts_sr)
                    # Resample to 16kHz mono float32
                    tts_audio_16k = resample_float32(tts_audio, tts_sr, SAMPLE_RATE)
                    # Convert to Int16LE PCM
                    tts_pcm = float32_to_int16le_bytes(tts_audio_16k)
                    log.info("Sending TTS audio: %d bytes (%.2fs)", len(tts_pcm), len(tts_audio_16k)/SAMPLE_RATE)
                    await websocket.send_bytes(tts_pcm)
                    # Also send transcript JSON
                    await websocket.send_text(json.dumps({"type": "transcript", "text": tts_text}))
                    log.info("Sent transcript JSON: %r", tts_text)
            elif "text" in msg:
                try:
                    ctrl = json.loads(msg["text"])
                    log.info("Received control message: %s", ctrl)
                    if ctrl.get("type") == "config":
                        config = ctrl
                        log.info("Session config: %s", config)
                        await websocket.send_text(json.dumps({"type": "ack", "config": config}))
                    # Handle other control types: pause, resume, flush, etc.
                except Exception as e:
                    log.error("Error parsing control message: %s", e)
                    await websocket.send_text(json.dumps({"type": "error", "error": str(e)}))
    except WebSocketDisconnect:
        log.info("WebSocket disconnected: %s", websocket.client)
    except Exception as e:
        log.error("Exception in WebSocket handler: %s", e)
    finally:
        log.info("WebSocket session closed (client: %s, total_bytes: %d)", websocket.client, total_bytes)

if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)
