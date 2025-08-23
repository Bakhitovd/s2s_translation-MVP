import logging
import time
from pathlib import Path
from typing import Optional, Tuple
import tempfile

import numpy as np
import soundfile as sf
from silero_tts.silero_tts import SileroTTS

logger = logging.getLogger(__name__)

class TTSManager:
    def __init__(self, default_sample_rate: int = 48000, device: str = "cpu"):
        # cache by (lang, speaker)
        self.models: dict[tuple[str, Optional[str]], SileroTTS] = {}
        self.default_sample_rate = default_sample_rate
        self.device = device

    def load(self, lang: str, speaker: Optional[str] = None) -> SileroTTS:
        key = (lang, speaker)
        if key in self.models:
            return self.models[key]

        start = time.time()
        model_id = SileroTTS.get_latest_model(lang)  # latest for the language
        tts = SileroTTS(
            model_id=model_id,
            language=lang,
            speaker=speaker,               # None => package picks default
            sample_rate=self.default_sample_rate,
            device=self.device,
        )
        self.models[key] = tts
        logger.info("Loaded SileroTTS %s, speaker=%s in %.2fs",
                    lang, speaker, time.time() - start)
        return tts

    def synth(self, text: str, lang: str = "ru", speaker: Optional[str] = None) -> Tuple[int, np.ndarray]:
        """
        Returns (sample_rate, mono float32 PCM).
        """
        tts = self.load(lang, speaker)

        # choose a concrete speaker if still None
        if speaker is None:
            speakers = tts.get_available_speakers()
            if not speakers:
                raise RuntimeError(f"No speakers available for language '{lang}'")
            speaker = speakers[0]  # take the first available

        # write to a temp wav (required by silero-tts API), then read back
        sr = int(tts.sample_rate) if tts.sample_rate else self.default_sample_rate
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.wav"
            # IMPORTANT: silero-tts requires an output file path
            tts.tts(text, str(out_path))  # writes the wav to disk
            audio, read_sr = sf.read(str(out_path), dtype="float32")
            if read_sr != sr:
                sr = read_sr  # trust file metadata

        return sr, audio.astype(np.float32)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mgr = TTSManager()
    sr, audio = mgr.synth("Привет, это тест.", lang="ru")
    print(f"Generated {len(audio)} samples @ {sr} Hz")

    # save proof
    sf.write("output_ru.wav", audio, sr)
