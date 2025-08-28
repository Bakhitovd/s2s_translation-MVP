# server/app.py
import json
import asyncio
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketState

from server import canary_ast, silero_tts, audio_util

app = FastAPI()

# Mount frontend as static at /static
frontend_path = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/")
async def root():
    index_path = frontend_path / "index.html"
    return FileResponse(index_path)


# Lazy globals
_ast_manager = None
_tts_manager = None


async def init_models():
    """Lazy-load heavy models once per process."""
    global _ast_manager, _tts_manager
    if _ast_manager is None:
        _ast_manager = canary_ast.ModelManager()
        _ast_manager.load()
    if _tts_manager is None:
        _tts_manager = silero_tts.TTSManager()


@app.websocket("/ws/audio")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await init_models()

    # Default direction; overridable via {type:"start", pair:"ru-en"} or src/tgt
    src_lang = "ru"
    tgt_lang = "en"

    try:
        loop = asyncio.get_running_loop()

        while True:
            msg = await ws.receive()  # dict with keys: type, text/bytes
            mtype = msg.get("type")

            # Stop reading immediately on disconnect → no post-disconnect .receive() calls
            if mtype == "websocket.disconnect":
                break

            if "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    continue

                if data.get("type") == "start":
                    pair = data.get("pair")
                    if pair and "-" in pair:
                        src_lang, tgt_lang = pair.split("-", 1)
                    else:
                        src_lang = data.get("src", src_lang) or src_lang
                        tgt_lang = data.get("tgt", tgt_lang) or tgt_lang

                # Ack for UI state machines
                await ws.send_text(json.dumps({"type": "ack"}))
                continue

            if "bytes" in msg and msg["bytes"]:
                # Int16LE PCM -> float32 mono in [-1,1]
                pcm_bytes = msg["bytes"]
                audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                # Offload blocking work to a thread (don’t block the event loop)
                result = await loop.run_in_executor(
                    None, lambda: _ast_manager.translate(audio, src_lang=src_lang, tgt_lang=tgt_lang)
                )
                text = getattr(result, "text", "")

                sr, tts_audio = await loop.run_in_executor(
                    None, lambda: _tts_manager.synth(text, lang=tgt_lang)
                )

                # Resample TTS to 16k mono and send back PCM int16
                tts_audio_16k = audio_util.resample_float32(tts_audio, sr, 16000)
                pcm_out = audio_util.float32_to_int16le_bytes(tts_audio_16k)

                await ws.send_text(json.dumps({"type": "transcript", "text": text}))
                await ws.send_bytes(pcm_out)

    except WebSocketDisconnect:
        # Normal path on tab close / refresh
        pass
    except Exception as e:
        # Log and swallow; don’t crash the process
        import logging
        logging.getLogger("uvicorn.error").exception("Error in WebSocket loop: %s", e)
    finally:
        # Avoid double-close: only close if still connected
        try:
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
