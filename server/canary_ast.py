import logging
import time
from typing import Any, Dict, Union
import numpy as np
import re
from typing import Any
import soundfile as sf

log = logging.getLogger(__name__)
# Strip Canary control tokens like <|en|><|pnc|> etc.
_CTRL_TOKEN_RE = re.compile(r"<\|[^|>]+?\|>")
# Fallback: pull text='...' from Hypothesis.__repr__()
_HYP_REPR_TEXT_RE = re.compile(r"text=(?:'|\")(.+?)(?:'|\")")


def _to_clean_text(obj: Any) -> str:
    """Return a clean str for Canary outputs: str | dict{text} | obj.text | list[...]."""
    # Unwrap list/tuple
    if isinstance(obj, (list, tuple)):
        return _to_clean_text(obj[0]) if obj else ""
    # Direct string
    if isinstance(obj, str):
        s = obj
    # Dict with 'text'
    elif isinstance(obj, dict) and isinstance(obj.get("text"), str):
        s = obj["text"]
    else:
        # Object with .text
        t = getattr(obj, "text", None)
        if isinstance(t, str):
            s = t
        else:
            # Fallback: parse repr(Hypothesis(...))
            r = repr(obj)
            m = _HYP_REPR_TEXT_RE.search(r)
            s = m.group(1) if m else r
    # Remove control tokens and trim
    try:
        return _CTRL_TOKEN_RE.sub("", s).strip()
    except Exception:
        return str(s).strip()
    

class InternalModelResult:
    def __init__(self, text: str,segments=None, meta: Dict[str, Any] = None):
        self.text = text
        self.segments = segments or []
        self.meta = meta or {}


class ModelManager:
    """Minimal wrapper around NVIDIA Canary for speech → translation (AST) and ASR."""

    def __init__(self):
        self.model = None

    def load(self):
        from nemo.collections.asr.models import ASRModel
        t0 = time.time()
        self.model = ASRModel.from_pretrained("nvidia/canary-1b-v2")
        log.info("Loaded Canary in %.2fs", time.time() - t0)

    def translate(self, audio_or_path, src_lang: str, tgt_lang: str, timestamps: bool = False) -> InternalModelResult:
        if self.model is None:
            raise RuntimeError("Canary AST model not loaded")

        if isinstance(audio_or_path, str):
            audio_np, _ = sf.read(audio_or_path, dtype="float32")
        else:
            audio_np = audio_or_path

        t0 = time.time()
        raw = self.model.transcribe(
            [audio_np],
            task="ast",               # force translation (not plain ASR)
            source_lang=src_lang,
            target_lang=tgt_lang,
            timestamps=timestamps,
        )[0]
        text = _to_clean_text(raw)   # <— ALWAYS a str now
        return InternalModelResult(text=text, meta={"inference_ms": int((time.time() - t0) * 1000)})

    def transcribe(self, audio_or_path, lang: str, timestamps: bool = False) -> InternalModelResult:
        """ASR-only: returns transcript in source language (no translation)."""
        if self.model is None:
            raise RuntimeError("Canary AST model not loaded")

        if isinstance(audio_or_path, str):
            audio_np, _ = sf.read(audio_or_path, dtype="float32")
        else:
            audio_np = audio_or_path

        t0 = time.time()
        raw = self.model.transcribe(
            [audio_np],
            task="asr",               # ASR mode (no translation)
            source_lang=lang,
            target_lang=lang,
            timestamps=timestamps,
        )[0]
        print(raw.score)
        text = _to_clean_text(raw)
        return InternalModelResult(text=text, meta={"inference_ms": int((time.time() - t0) * 1000)})
