#!/usr/bin/env python3
"""
speech2speech.py — English↔Russian speech-to-speech using your Canary + Silero wrappers.

Examples:
  # English -> Russian
  python speech2speech.py input_en.mp3 --src en --tgt ru -o out_ru.wav

  # Russian -> English
  python speech2speech.py input_ru.wav --src ru --tgt en -o out_en.wav

Requirements (besides your two modules):
  pip install pydub soundfile numpy
  # Also ensure FFmpeg is installed and on PATH (required by pydub).
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from server.canary_ast import ModelManager as ASTModelManager
from server.silero_tts import TTSManager

logger = logging.getLogger("speech2speech")

CTRL_TOKEN_RE = re.compile(r"<\|[^|>]+?\|>")
HYP_REPR_TEXT_RE = re.compile(r"text=(?:'|\")(.+?)(?:'|\")")


def extract_clean_text(obj: Any) -> str:
    """
    Robustly get a clean string from Canary results:
    - str directly
    - dict with 'text'
    - object with .text
    - fallback parse of repr(Hypothesis(...))
    Then strip control tokens like <|en|><|pnc|> etc.
    """
    if isinstance(obj, str):
        s = obj
    elif isinstance(obj, dict) and isinstance(obj.get("text"), str):
        s = obj["text"]
    else:
        t = getattr(obj, "text", None)
        if isinstance(t, str):
            s = t
        else:
            r = repr(obj)
            m = HYP_REPR_TEXT_RE.search(r)
            s = m.group(1) if m else r
    try:
        return CTRL_TOKEN_RE.sub("", s).strip()
    except Exception:
        return str(s)


def load_audio_as_float32(path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load ANY common audio format via pydub/ffmpeg and convert to mono 16k float32 [-1, 1].
    """
    seg = AudioSegment.from_file(path)
    seg = seg.set_frame_rate(target_sr).set_channels(1).set_sample_width(2)  # 16-bit PCM
    samples = np.array(seg.get_array_of_samples())
    if samples.dtype.kind in ("i", "u"):
        audio = (samples.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    else:
        audio = samples.astype(np.float32)
    return audio


def main():
    ap = argparse.ArgumentParser(description="EN↔RU speech→speech using Canary AST + Silero TTS")
    ap.add_argument("input", help="Input audio file (wav/mp3/flac/ogg/m4a...)")
    ap.add_argument("-o", "--output", help="Output WAV file (default: derive from input + _en/_ru)")
    ap.add_argument("--src", choices=["en", "ru"], required=True, help="Source language")
    ap.add_argument("--tgt", choices=["en", "ru"], required=True, help="Target language")
    ap.add_argument("--spk-ru", default="baya", help="Silero speaker for Russian (default: baya)")
    ap.add_argument("--spk-en", default="en_0", help="Silero speaker for English (default: en_0)")
    ap.add_argument("--tts-device", default="cpu", help="Silero TTS device (cpu/cuda; default: cpu)")
    ap.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    ap.add_argument("--json", action="store_true", help="Print a JSON summary at the end")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not os.path.exists(args.input):
        logger.error("Input not found: %s", args.input)
        sys.exit(3)

    # Derive output path if missing
    if args.output:
        out_wav = args.output
    else:
        stem, _ = os.path.splitext(args.input)
        suffix = "_ru" if args.tgt == "ru" else "_en"
        out_wav = f"{stem}{suffix}.wav"

    # --- Load models ---
    ast = ASTModelManager()
    logger.info("Loading Canary AST...")
    t0 = time.time()
    ast.load()
    logger.info("Canary ready in %.2fs", time.time() - t0)

    tts = TTSManager(device=args.tts_device)
    logger.info("Loading Silero TTS voices...")
    # Preload the exact speakers we will use
    tts.load("ru", args.spk_ru)
    tts.load("en", args.spk_en)
    logger.info("Silero TTS ready.")

    # --- Prepare audio ---
    logger.info("Reading/normalizing input audio...")
    audio_np = load_audio_as_float32(args.input, target_sr=16000)

    # --- AST (speech -> text in target language, as provided by your wrapper) ---
    logger.info("Running Canary AST (%s -> %s)...", args.src, args.tgt)
    t1 = time.time()
    res = ast.translate(audio_np, src_lang=args.src, tgt_lang=args.tgt, timestamps=False)
    ast_ms = int((time.time() - t1) * 1000)
    text = extract_clean_text(res.text)
    if not text:
        logger.warning("AST produced empty text; writing a short tone so you still get an output file.")
        # 0.4s tone as fallback
        dur = 0.4
        t = np.linspace(0, dur, int(16000 * dur), endpoint=False)
        tone = 0.25 * np.sin(2 * np.pi * 660 * t).astype(np.float32)
        sf.write(out_wav, tone, 16000)
        if args.json:
            print(json.dumps({
                "ok": True,
                "note": "ast_empty_fallback_tone",
                "output_wav": out_wav,
                "meta": {"ast_time_ms": ast_ms, "src": args.src, "tgt": args.tgt}
            }, ensure_ascii=False, indent=2))
        return

    # --- TTS (text -> speech in target language) ---
    spk = args.spk_ru if args.tgt == "ru" else args.spk_en
    logger.info("Synthesizing TTS (lang=%s, speaker=%s)...", args.tgt, spk)
    t2 = time.time()
    sr_tts, audio_tts = tts.synth(text, lang=args.tgt, speaker=spk)
    tts_ms = int((time.time() - t2) * 1000)
    # Guard: if Silero produced header-only
    if isinstance(audio_tts, np.ndarray) and audio_tts.size <= 1:
        logger.warning("TTS produced too-short audio; replacing with a short tone.")
        dur = 0.4
        t = np.linspace(0, dur, int(16000 * dur), endpoint=False)
        audio_tts = 0.25 * np.sin(2 * np.pi * 520 * t).astype(np.float32)
        sr_tts = 16000

    # --- Save ---
    sf.write(out_wav, audio_tts.astype(np.float32), int(sr_tts))
    logger.info("Saved: %s", out_wav)

    if args.json:
        print(json.dumps({
            "ok": True,
            "text": text,
            "output_wav": out_wav,
            "meta": {
                "ast_time_ms": ast_ms,
                "tts_time_ms": tts_ms,
                "src": args.src,
                "tgt": args.tgt,
                "speaker": spk
            }
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
