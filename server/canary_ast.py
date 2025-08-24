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


if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys
    import re

    # Strip Canary control tokens like <|en|><|pnc|> etc.
    CTRL_TOKEN_RE = re.compile(r"<\|[^|>]+?\|>")
    # Fallback extractor from repr(Hypothesis(...))
    HYP_REPR_TEXT_RE = re.compile(r"text=(?:'|\")(.+?)(?:'|\")")

    def extract_text(obj) -> str:
        """Return a clean string from NeMo Hypothesis / dict / str."""
        # common shapes: str; dict with 'text'; object with .text
        if isinstance(obj, str):
            s = obj
        elif isinstance(obj, dict) and isinstance(obj.get("text"), str):
            s = obj["text"]
        else:
            t = getattr(obj, "text", None)
            if isinstance(t, str):
                s = t
            else:
                # last resort: parse repr
                r = repr(obj)
                m = HYP_REPR_TEXT_RE.search(r)
                s = m.group(1) if m else r
        # remove control tokens
        try:
            s = CTRL_TOKEN_RE.sub("", s).strip()
        except Exception:
            s = str(s)
        return s

    def to_serializable(x):
        """Conservative JSON serializer."""
        if x is None or isinstance(x, (str, int, float, bool)):
            return x
        if isinstance(x, (list, tuple)):
            return [to_serializable(i) for i in x]
        if isinstance(x, dict):
            return {str(k): to_serializable(v) for k, v in x.items()}
        # objects with a useful .__dict__
        d = getattr(x, "__dict__", None)
        if isinstance(d, dict) and d:
            return {k: to_serializable(v) for k, v in d.items()}
        # fallback
        return str(x)

    def load_audio_as_float32(path: str, target_sr: int = 16000) -> np.ndarray:
        """Load audio via pydub/ffmpeg → mono 16k float32 in [-1, 1]."""
        seg = AudioSegment.from_file(path)
        seg = seg.set_frame_rate(target_sr).set_channels(1).set_sample_width(2)  # 16-bit PCM
        samples = np.array(seg.get_array_of_samples())
        if samples.dtype.kind in ("i", "u"):
            audio = (samples.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        else:
            audio = samples.astype(np.float32)
        return audio

    parser = argparse.ArgumentParser(description="Smoke test for Canary AST ModelManager.translate()")
    parser.add_argument("audio", nargs="?", help="Path to audio file (wav/mp3/flac/ogg/m4a...)")
    parser.add_argument("--src", default="en")
    parser.add_argument("--tgt", default="ru")
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--raw", action="store_true", help="Print text only")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    mm = ModelManager()
    try:
        logger.info("Loading Canary AST model...")
        mm.load()
        logger.info("Canary AST model loaded successfully.")
    except Exception:
        logger.exception("Failed to load Canary AST.")
        sys.exit(2)

    if args.audio:
        if not os.path.exists(args.audio):
            logger.error(f"Audio file not found: {args.audio}")
            sys.exit(3)
        try:
            audio_np = load_audio_as_float32(args.audio, target_sr=16000)
        except Exception:
            logger.exception("Failed to load/convert audio.")
            sys.exit(4)
    else:
        logger.warning("No audio provided; generating a 0.5s 660 Hz tone.")
        dur = 0.5
        t = np.linspace(0, dur, int(16000 * dur), endpoint=False)
        audio_np = (0.3 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)

    try:
        t0 = time.time()
        res = mm.translate(audio_np, src_lang=args.src, tgt_lang=args.tgt, timestamps=args.timestamps)
        wall_ms = int((time.time() - t0) * 1000)
    except Exception:
        logger.exception("Inference failed.")
        sys.exit(5)

    # Always coerce to clean string
    text_str = extract_text(res.text)

    if args.raw:
        print(text_str or "")
        sys.exit(0)

    out = {
        "text": text_str,
        "meta": {
            **to_serializable(res.meta or {}),
            "wall_time_ms": wall_ms,
            "src_lang": args.src,
            "tgt_lang": args.tgt,
            "timestamps_requested": bool(args.timestamps),
        },
        "segments": to_serializable(res.segments or []),
    }

    print(json.dumps(out, ensure_ascii=False, indent=2, default=to_serializable))
