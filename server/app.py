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
# Silero TTS manager (lazy global)
tts_mgr = None
# MT manager (lazy global)
mt_mgr = None

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.on_event("startup")
def preload_models():
    global canary_mgr, tts_mgr, mt_mgr
    from server.canary_ast import ModelManager
    from server.silero_tts import TTSManager
    from server.mt import mt_mgr as global_mt_mgr
    canary_mgr = ModelManager()
    canary_mgr.load()
    tts_mgr = TTSManager(sample_rate=24000, device="cpu")
    mt_mgr = global_mt_mgr
    log.info("All models preloaded at startup.")

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
    utter_buf = bytearray()
    total_bytes = 0
    last_text = ""
    pipeline_mode = "cascade"  # default

    try:
        SAMPLE_RATE = 16000
        BYTES_PER_SAMPLE = 2  # Int16LE
        MAX_UTTER_SEC = 8
        MAX_UTTER_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * MAX_UTTER_SEC

        async def process_and_respond(pcm_bytes: bytes):
            audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            global canary_mgr, tts_mgr, mt_mgr
            src_lang = config.get("srcLang", "ru") if config else "ru"
            tgt_lang = config.get("dstLang", "en") if config else "en"
            mode = pipeline_mode

            # Cascade: ASR → MT → TTS
            if mode == "cascade":
                log.info("Pipeline: ASR+MT. src=%s tgt=%s audio=%.2fs", src_lang, tgt_lang, len(audio_np)/SAMPLE_RATE)
                asr_result = canary_mgr.transcribe(audio_np, src_lang)
                asr_text = (asr_result.text or "").strip()
                log.info("Canary ASR result: %r", asr_result.__dict__)
                await websocket.send_text(json.dumps({"type": "asr", "text": asr_text}))
                # MT
                try:
                    mt_text = mt_mgr.translate_text(src_lang, tgt_lang, asr_text)
                except Exception as e:
                    log.error("MT error: %s", e)
                    await websocket.send_text(json.dumps({"type": "error", "error": f"MT error: {e}"}))
                    return asr_text
                text = (mt_text or "").strip()
                log.info("MT result: %r", text)
            else:
                # AST (direct speech-to-translation)
                log.info("Pipeline: AST. src=%s tgt=%s audio=%.2fs", src_lang, tgt_lang, len(audio_np)/SAMPLE_RATE)
                result = canary_mgr.translate(audio_np, src_lang, tgt_lang)
                text = (result.text or "").strip()
                log.info("Canary AST result: %r", text)

            # Duplicate/short-text suppression
            def _norm_words(s: str):
                return [w for w in "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in s).split() if w]
            nw = _norm_words(text)
            suppressed = False
            if len("".join(nw)) < 4:
                suppressed = True
            else:
                stoplist = {"yes","yeah","ok","okay","uh","mmm","thanks","thank","thankyou","да","угу","спасибо"}
                if " ".join(nw) in stoplist:
                    suppressed = True
                else:
                    prev = _norm_words(last_text)
                    if prev:
                        inter = len(set(nw) & set(prev))
                        union = len(set(nw) | set(prev))
                        sim = inter / union if union else 0.0
                        if sim >= 0.9:
                            suppressed = True
            # Send transcript (with filtered flag)
            await websocket.send_text(json.dumps({"type": "transcript", "text": text, "filtered": suppressed}))
            if suppressed:
                return text
            # Run Silero TTS
            if tts_mgr is None:
                from server.silero_tts import TTSManager
                tts_mgr = TTSManager(sample_rate=24000, device="cpu")
                log.info("Silero TTS manager loaded")
            tts_sr, tts_audio = tts_mgr.synth(text, tgt_lang)
            tts_audio_16k = resample_float32(tts_audio, tts_sr, SAMPLE_RATE)
            tts_pcm = float32_to_int16le_bytes(tts_audio_16k)
            log.info("Sending TTS audio: %d bytes (%.2fs)", len(tts_pcm), len(tts_audio_16k)/SAMPLE_RATE)
            await websocket.send_bytes(tts_pcm)
            return text

        while True:
            try:
                msg = await websocket.receive()
            except RuntimeError:
                log.info("WebSocket disconnect detected (client: %s)", websocket.client)
                break
            if "bytes" in msg:
                chunk = msg["bytes"]
                utter_buf.extend(chunk)
                total_bytes += len(chunk)
                log.debug("Received PCM chunk: %d bytes (total: %d)", len(chunk), total_bytes)
                if len(utter_buf) >= MAX_UTTER_BYTES:
                    log.info("Max utterance reached: %d bytes, processing", len(utter_buf))
                    text_out = await process_and_respond(bytes(utter_buf))
                    last_text = text_out
                    utter_buf.clear()
            elif "text" in msg:
                try:
                    ctrl = json.loads(msg["text"])
                    log.info("Received control message: %s", ctrl)
                    if ctrl.get("type") == "config":
                        config = ctrl
                        pipeline_mode = ctrl.get("pipeline", "cascade")
                        log.info("Session config: %s", config)
                        await websocket.send_text(json.dumps({"type": "ack", "config": config}))
                    elif ctrl.get("type") == "flush":
                        if len(utter_buf) > 0:
                            log.info("Flush received: processing %d bytes", len(utter_buf))
                            text_out = await process_and_respond(bytes(utter_buf))
                            last_text = text_out
                            utter_buf.clear()
                        else:
                            log.info("Flush received with empty buffer; ignoring")
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
