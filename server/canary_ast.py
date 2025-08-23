import logging
from typing import Any, Dict
import time
import numpy as np
from pydub import AudioSegment

logger = logging.getLogger("uvicorn.error")


class InternalModelResult:
    def __init__(self, text: str, segments=None, meta: Dict[str, Any] = None):
        self.text = text
        self.segments = segments or []
        self.meta = meta or {}


class ModelManager:
    """
    Stubbed ModelManager for Canary AST.
    In tests, this will be patched. In production, load() will initialize NeMo model.
    """

    def __init__(self):
        self.model = None

    def load(self):
        import time
        from nemo.collections.asr.models import ASRModel
        start = time.time()
        try:

            self.model = ASRModel.from_pretrained("nvidia/canary-1b-v2")

        except Exception as e:

            logger.error(f"Failed to load Canary AST: {e}")
            raise

        end = time.time()
        logger.info(f"Loaded Canary AST in {end - start:.2f}s")



    def translate(self, audio_or_path, src_lang: str, tgt_lang: str, timestamps: bool = False) -> InternalModelResult:
        
        if self.model is None:
            raise RuntimeError("Canary AST model not loaded")
        start = time.time()


        if isinstance(audio_or_path, str):
            import soundfile as sf
            audio_np, sr = sf.read(audio_or_path, dtype="float32")
        else:
            audio_np = audio_or_path

        try:
            text = self.model.transcribe([audio_np])[0]
        except Exception as e:
            logger.error(f"AST inference failed: {e}")
            raise
        end = time.time()
        return InternalModelResult(
            text=text,
            meta={"inference_time_ms": int((end - start) * 1000)}
        )
