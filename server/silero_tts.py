import logging
import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from silero_tts.silero_tts import SileroTTS

log = logging.getLogger(__name__)


class TTSManager:
    """Minimal Silero TTS manager with sane defaults and tiny cache."""

    def __init__(self, sample_rate: int = 24000, device: str = "cpu"):
        self._cache: dict[tuple[str, Optional[str]], SileroTTS] = {}
        self.sample_rate = sample_rate
        self.device = device

    def load(self, lang: str, speaker: Optional[str] = None) -> SileroTTS:
        # Defaults that don't explode:
        # - RU: common multi-speaker voices -> pick "baya"
        # - EN: v3_en single-speaker -> pass None
        if speaker is None:
            speaker = "baya" if lang == "ru" else None

        key = (lang, speaker)
        if key in self._cache:
            return self._cache[key]

        t0 = time.time()
        model_id = SileroTTS.get_latest_model(lang)  # keep it simple
        tts = SileroTTS(
            model_id=model_id,
            language=lang,
            speaker=speaker,              # None is valid for single-speaker EN
            sample_rate=self.sample_rate,
            device=self.device,
        )
        self._cache[key] = tts
        log.info("Silero loaded: lang=%s speaker=%s in %.2fs", lang, speaker, time.time() - t0)
        return tts

    def synth(self, text: str, lang: str, speaker: Optional[str] = None) -> Tuple[int, np.ndarray]:
        tts = self.load(lang, speaker)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.wav"
            tts.tts(text, str(out))
            audio, sr = sf.read(str(out), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        return int(sr), audio.astype(np.float32)
