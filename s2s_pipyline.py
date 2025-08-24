#!/usr/bin/env python3
import argparse
import logging
import os
import sys

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from server.canary_ast import ModelManager as ASTModelManager
from server.silero_tts import TTSManager

log = logging.getLogger("s2s")


def load_audio_float32(path: str, target_sr: int = 16000) -> np.ndarray:
    seg = AudioSegment.from_file(path).set_frame_rate(target_sr).set_channels(1).set_sample_width(2)
    a = np.array(seg.get_array_of_samples())
    return (a.astype(np.float32) / 32768.0).clip(-1.0, 1.0) if a.dtype.kind in ("i", "u") else a.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="EN↔RU speech→speech using Canary (AST) + Silero (TTS)")
    ap.add_argument("input", help="Audio file (wav/mp3/ogg/m4a/...)")
    ap.add_argument("--src", choices=["en", "ru"], required=True)
    ap.add_argument("--tgt", choices=["en", "ru"], required=True)
    ap.add_argument("-o", "--output", help="Output WAV (default: <input>_<tgt>.wav)")
    ap.add_argument("--device", default="cpu", help="Silero device: cpu/cuda")
    ap.add_argument("--spk-ru", default="baya", help="RU speaker (default: baya)")
    ap.add_argument("--spk-en", default=None, help="EN speaker (default: None for v3_en)")
    ap.add_argument("--log", default=os.getenv("LOG_LEVEL", "INFO"))
    args = ap.parse_args()

    logging.basicConfig(level=args.log.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not os.path.exists(args.input):
        log.error("Input not found: %s", args.input)
        sys.exit(2)
    out_wav = args.output or f"{os.path.splitext(args.input)[0]}_{args.tgt}.wav"

    # Models
    ast = ASTModelManager()
    log.info("Loading Canary AST...")
    ast.load()
    tts = TTSManager(device=args.device)
    if args.tgt == "ru":
        tts.load("ru", args.spk_ru)
    else:
        tts.load("en", args.spk_en)  # None is fine (single-speaker EN)

    # Audio → AST
    x = load_audio_float32(args.input, 16000)
    log.info("Translating %s → %s ...", args.src, args.tgt)
    res = ast.translate(x, src_lang=args.src, tgt_lang=args.tgt, timestamps=False)
    text = (res.text or "").strip()
    if not text:
        log.warning("AST produced empty text; emitting a short tone.")
        dur = 0.4
        t = np.linspace(0, dur, int(16000 * dur), endpoint=False)
        tone = 0.25 * np.sin(2 * np.pi * 660 * t).astype(np.float32)
        sf.write(out_wav, tone, 16000)
        print(out_wav)
        return

    # TTS → WAV
    spk = args.spk_ru if args.tgt == "ru" else args.spk_en
    sr, y = tts.synth(text, lang=args.tgt, speaker=spk)
    sf.write(out_wav, y, sr)
    log.info("Saved: %s", out_wav)
    print(out_wav)


if __name__ == "__main__":
    main()