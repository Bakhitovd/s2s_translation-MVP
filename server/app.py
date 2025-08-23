import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from server.audio_util import AudioBuffer, to_mono16k_float32, wav_bytes_from_float32
from server.canary_ast import ModelManager as ASTManager
from server.silero_tts import TTSManager

logger = logging.getLogger("uvicorn.error")

app = FastAPI()

# Serve frontend index at GET /
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

# Mount static files (JS, CSS, etc.)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


def validate_ws_control(payload: dict):
    """
    Validate and normalize initial WS JSON config.
    """
    src = payload.get("src", "auto")
    tgt = payload.get("tgt")
    mode = payload.get("mode")

    if mode:
        if mode == "en->ru":
            src, tgt = "en", "ru"
        elif mode == "ru->en":
            src, tgt = "ru", "en"
        elif mode == "auto":
            src, tgt = "auto", None
    else:
        if src == "auto" and not tgt:
            src, tgt = "en", "ru"

    return {"src": src, "tgt": tgt, "mode": mode or f"{src}->{tgt}"}


async def handle_utterance(ws: WebSocket, buf: AudioBuffer, mode: str):
    """
    Orchestrates resampling, AST call, TTS call, and sending of WAV + metadata.
    """
    pcm_bytes = buf.flush()
    if not pcm_bytes:
        logger.warning("handle_utterance called with empty buffer")
        return

    logger.info(f"Received {len(pcm_bytes)} bytes from client")

    # Convert to mono 16k float32
    try:
        sr, audio_np = to_mono16k_float32(pcm_bytes)
        logger.info(f"Resampled audio to {sr} Hz, {len(audio_np)} samples")
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return

    # Run AST
    try:
        ast_result = ast_manager.translate(audio_np, src_lang="en", tgt_lang="ru")
        text = ast_result.text
        logger.info(f"AST result: {text}")
    except Exception as e:
        logger.error(f"AST failed: {e}")
        # Fallback: proceed with placeholder text so TTS (or tone) can still respond
        text = "(ast_failed)"
        await ws.send_text(json.dumps({"src": "en", "tgt": "ru", "text": text, "ast_failed": True}))

    # Run TTS
    try:
        sr_tts, audio_tts = tts_manager.synth(text, lang="ru", speaker="baya")
        wav_bytes = wav_bytes_from_float32(audio_tts, sr_tts)
        logger.info(f"TTS generated {len(wav_bytes)} bytes")
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        await ws.send_text(json.dumps({"src": "en", "tgt": "ru", "text": text, "tts_failed": True}))
        # Fallback: send a short tone so user hears acknowledgement
        import numpy as np
        duration = 0.6
        t = np.linspace(0, duration, int(16000 * duration), endpoint=False)
        tone = 0.3 * np.sin(2 * np.pi * 880 * t).astype(np.float32)
        wav_bytes = wav_bytes_from_float32(tone, 16000)
        await ws.send_bytes(wav_bytes)
        return

    # Send JSON metadata
    await ws.send_text(
        json.dumps({"src": "en", "tgt": "ru", "text": text})
    )
    logger.info("Sent JSON metadata to client")

    # Send WAV blob
    await ws.send_bytes(wav_bytes)
    logger.info("Sent WAV audio to client")

    try:
        await ws.send_text(json.dumps({"status": "utterance_complete"}))
        logger.info("Notified client utterance complete")
    except Exception as e:
        logger.error(f"Failed to notify client: {e}")


@app.websocket("/ws/translate")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    buf = AudioBuffer()
    mode = "auto"

    try:
        while True:
            msg = await ws.receive()
            if "text" in msg:
                if msg["text"] == "__flush__":
                    await handle_utterance(ws, buf, mode)
                else:
                    payload = json.loads(msg["text"])
                    cfg = validate_ws_control(payload)
                    mode = cfg["mode"]
            elif "bytes" in msg:
                buf.append(msg["bytes"])
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

# Initialize global model managers
ast_manager = ASTManager()
tts_manager = TTSManager()
try:
    logger.info("Loading Canary AST model...")
    ast_manager.load()
    logger.info("Canary AST model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load AST model: {e}", exc_info=True)

try:
    logger.info("Loading Silero TTS models...")
    tts_manager.load("ru", "baya")
    tts_manager.load("en", "lj_16khz")
    logger.info("Silero TTS models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load TTS models: {e}", exc_info=True)
