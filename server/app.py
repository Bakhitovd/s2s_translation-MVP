import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import re

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

# Health check
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# Mount static files (JS, CSS, etc.)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


import re
from typing import Any

# Strip Canary control tokens like <|en|><|pnc|> etc.
CTRL_TOKEN_RE = re.compile(r"<\|[^|>]+?\|>")
# Fallback: extract ... text='...' ... from Hypothesis.__repr__()
HYP_REPR_TEXT_RE = re.compile(r"text=(?:'|\")(.+?)(?:'|\")")

def _ast_to_clean_text(ast_out: Any) -> str:
    """
    Normalize NeMo Canary outputs to a clean string.
    Handles: Hypothesis, list[Hypothesis], dict, str.
    Falls back to parsing repr() if needed.
    """
    # Unwrap lists/tuples by best-first
    if isinstance(ast_out, (list, tuple)):
        for cand in ast_out:
            s = _ast_to_clean_text(cand)
            if s:
                return s
        return ""

    # Direct string
    if isinstance(ast_out, str):
        s = ast_out
    else:
        # dict-like
        if isinstance(ast_out, dict) and isinstance(ast_out.get("text"), str):
            s = ast_out["text"]
        else:
            # Attribute 'text' if present
            t = getattr(ast_out, "text", None)
            if isinstance(t, str):
                s = t
            else:
                # Fallback: parse repr() of Hypothesis
                r = repr(ast_out)
                m = HYP_REPR_TEXT_RE.search(r)
                s = m.group(1) if m else ""

    if isinstance(s, bytes):
        s = s.decode("utf-8", "ignore")

    # Remove control tokens and trim
    s = CTRL_TOKEN_RE.sub("", s).strip()
    return s

def _looks_latin(s: str) -> bool:
    return bool(s) and all(ord(c) < 128 for c in s)


def validate_ws_control(payload: dict) -> dict:
    """
    Validate and normalize initial WS JSON config.
    Accepts either:
      - {"mode": "en->ru"} or {"mode": "ru->en"} or {"mode": "auto"}
      - {"src": "...", "tgt": "..."}
    Returns a dict with concrete src/tgt and canonical mode.
    """
    src = payload.get("src")
    tgt = payload.get("tgt")
    mode = payload.get("mode")

    if mode:
        if mode == "en->ru":
            src, tgt = "en", "ru"
        elif mode == "ru->en":
            src, tgt = "ru", "en"
        elif mode == "auto":
            # Default pairing when auto is requested
            src, tgt = "en", "ru"
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    else:
        # If explicit src/tgt not given, default
        src = (src or "en").lower()
        tgt = (tgt or "ru").lower()

    if src not in {"en", "ru"} or tgt not in {"en", "ru"} or src == tgt:
        raise ValueError(f"Invalid language pair: {src}->{tgt}")

    return {"src": src, "tgt": tgt, "mode": f"{src}->{tgt}"}

async def handle_utterance(ws: WebSocket, buf: AudioBuffer, lang_cfg: dict):
    """
    Orchestrates resampling, AST call, TTS call, and sending of WAV + metadata.
    lang_cfg: {"src": "...", "tgt": "...", "mode": "..."}
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
        logger.error(f"Audio conversion failed: {e}", exc_info=True)
        return

    src_lang = lang_cfg["src"]
    tgt_lang = lang_cfg["tgt"]

    # Run AST
    try:
        ast_out = ast_manager.translate(audio_np, src_lang=src_lang, tgt_lang=tgt_lang)
        raw_text = _ast_to_clean_text(ast_out)
        logger.info(f"AST text: {raw_text!r}")
        if not raw_text:
            logger.warning("AST produced empty text; sending tone and metadata.")
            await ws.send_text(json.dumps({"src": src_lang, "tgt": tgt_lang, "text": "", "ast_empty": True}))
            # Audible tone so the client gets feedback
            import numpy as np
            duration = 0.4
            t = np.linspace(0, duration, int(16000 * duration), endpoint=False)
            tone = 0.3 * np.sin(2 * np.pi * 660 * t).astype(np.float32)
            wav_bytes = wav_bytes_from_float32(tone, 16000)
            await ws.send_bytes(wav_bytes)
            return
        if tgt_lang == "ru" and _looks_latin(raw_text):
            logger.warning("AST returned ASCII text for ru target; passing through to TTS anyway.")
    except Exception as e:
        logger.error(f"AST failed: {e}", exc_info=True)
        raw_text = "(ast_failed)"
        await ws.send_text(json.dumps({"src": src_lang, "tgt": tgt_lang, "text": raw_text, "ast_failed": True}))
        # Fallback tone
        import numpy as np
        duration = 0.4
        t = np.linspace(0, duration, int(16000 * duration), endpoint=False)
        tone = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        wav_bytes = wav_bytes_from_float32(tone, 16000)
        await ws.send_bytes(wav_bytes)
        return

    # Run TTS
    try:
        speaker = "baya" if tgt_lang == "ru" else "en_0"
        sr_tts, audio_tts = tts_manager.synth(raw_text, lang=tgt_lang, speaker=speaker)
        wav_bytes = wav_bytes_from_float32(audio_tts, sr_tts)
        logger.info(f"TTS generated {len(wav_bytes)} bytes")
        if len(wav_bytes) <= 44:
            logger.warning("TTS produced empty WAV payload; replacing with tone.")
            import numpy as np
            duration = 0.4
            t = np.linspace(0, duration, int(16000 * duration), endpoint=False)
            tone = 0.3 * np.sin(2 * np.pi * 520 * t).astype(np.float32)
            wav_bytes = wav_bytes_from_float32(tone, 16000)
    except Exception as e:
        logger.error(f"TTS failed: {e}", exc_info=True)
        await ws.send_text(json.dumps({"src": src_lang, "tgt": tgt_lang, "text": raw_text, "tts_failed": True}))
        # Fallback: short tone
        import numpy as np
        duration = 0.4
        t = np.linspace(0, duration, int(16000 * duration), endpoint=False)
        tone = 0.3 * np.sin(2 * np.pi * 880 * t).astype(np.float32)
        wav_bytes = wav_bytes_from_float32(tone, 16000)
        await ws.send_bytes(wav_bytes)
        return

    # Send JSON metadata
    await ws.send_text(json.dumps({"src": src_lang, "tgt": tgt_lang, "text": raw_text}))
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
    lang_cfg = {"src": "en", "tgt": "ru", "mode": "en->ru"}  # sane default

    try:
        while True:
            msg = await ws.receive()
            if "text" in msg:
                txt = msg["text"]
                if txt == "__flush__":
                    await handle_utterance(ws, buf, lang_cfg)
                elif txt == "__ping__":
                    await ws.send_text("__pong__")
                else:
                    try:
                        payload = json.loads(txt)
                        lang_cfg = validate_ws_control(payload)
                        await ws.send_text(json.dumps({"status": "cfg_ok", **lang_cfg}))
                    except Exception as e:
                        err = {"status": "cfg_error", "error": str(e)}
                        await ws.send_text(json.dumps(err))
                        logger.warning(f"Bad WS control payload: {e}")
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
    # Preload the exact speakers you use at runtime
    tts_manager.load("ru", "baya")
    tts_manager.load("en", None)  # v3_en

    logger.info("Silero TTS models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load TTS models: {e}", exc_info=True)


if __name__ == "__main__":
    # Run directly: python -m server.app  OR  python server/app.py
    import os
    import uvicorn

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = os.getenv("RELOAD", "").lower() in {"1", "true", "yes"}

    # Match your Docker CMD target
    uvicorn.run("server.app:app", host=host, port=port, reload=reload_flag, log_level="info")
