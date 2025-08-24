#!/usr/bin/env python3
"""
Real-time EN↔RU speech→speech using Canary (AST) + Silero (TTS)
Mic input → 2–3s delay → Speaker output.

Requirements:
  pip install sounddevice numpy soundfile scipy pydub  (pydub is already in your project)
Project deps:
  - server.canary_ast.ModelManager  (your existing file)
  - server.silero_tts.TTSManager    (your existing file)
  - server.audio_util.to_mono16k_float32, wav_bytes_from_float32 (your existing file)

Run (examples):
  python realtime_s2s.py --src ru --tgt en
  python realtime_s2s.py --src en --tgt ru --in-dev 1 --out-dev 3 --device cuda
"""

import argparse
import logging
import math
import queue
import signal
import sys
import threading
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

# Use your existing modules (same imports style as your batch script)
from server.canary_ast import ModelManager as ASTModelManager
from server.silero_tts import TTSManager
from server.audio_util import to_mono16k_float32

log = logging.getLogger("realtime-s2s")


# --------------------------- Utilities ---------------------------

def dbfs_int16(x: np.ndarray) -> float:
    """RMS dBFS for int16 mono/stereo frames."""
    if x.size == 0:
        return -120.0
    # normalize to [-1,1]
    xf = x.astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(xf**2) + 1e-12)
    return 20.0 * math.log10(rms + 1e-12)


def resample_float32(y: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    """High-quality resample using polyphase."""
    if in_sr == out_sr:
        return y.astype(np.float32, copy=False)
    g = math.gcd(in_sr, out_sr)
    up = out_sr // g
    dn = in_sr // g
    return resample_poly(y.astype(np.float32, copy=False), up, dn).astype(np.float32)


# --------------------------- Segmenter ---------------------------

class EnergyVADSegmenter:
    """
    Streaming segmenter (energy-based):

    - 30 ms frames @ 48 kHz
    - start when dBFS > threshold for ≥1 frame
    - end when dBFS < threshold for 'silence_ms' AND length ≥ min_chunk_ms
    - hard cut at max_chunk_ms
    """

    def __init__(
        self,
        sr: int = 48000,
        channels: int = 1,
        frame_ms: int = 30,
        min_chunk_ms: int = 600,
        max_chunk_ms: int = 2000,
        silence_ms: int = 220,
        threshold_db: float = -42.0,   # adjust on noisy mics: [-48 .. -35]
        calibrate_ms: int = 400,       # initial noise floor estimate
    ):
        self.sr = sr
        self.channels = channels
        self.samples_per_frame = int(sr * frame_ms / 1000)
        self.min_frames = max(1, int(min_chunk_ms / frame_ms))
        self.max_frames = max(2, int(max_chunk_ms / frame_ms))
        self.silence_frames = max(1, int(silence_ms / frame_ms))
        self.threshold_db_default = threshold_db
        self.calibrate_frames = max(1, int(calibrate_ms / frame_ms))

        self._cur: deque[np.ndarray] = deque()
        self._cur_db: deque[float] = deque()
        self._silence_run = 0
        self._active = False
        self._frames_seen = 0
        self._threshold_db = threshold_db
        self._calib_buf: list[float] = []

        # outputs: raw int16 bytes for the finalized chunk
        self.out_q: "queue.Queue[bytes]" = queue.Queue()

        # latency/telemetry
        self._chunk_started_at: Optional[float] = None

    def push(self, frame_i16: np.ndarray):
        """frame_i16: shape (N,) mono or (N, channels) int16"""
        # Ensure 2D shape for unified handling
        if frame_i16.ndim == 1:
            mono = frame_i16
        else:
            mono = frame_i16.mean(axis=1).astype(np.int16)

        # If sizes drift, drop extras (shouldn't happen with fixed blocksize)
        if mono.shape[0] != self.samples_per_frame:
            if mono.shape[0] > self.samples_per_frame:
                mono = mono[: self.samples_per_frame]
            else:
                pad = self.samples_per_frame - mono.shape[0]
                mono = np.pad(mono, (0, pad), mode="constant")

        self._frames_seen += 1
        level = dbfs_int16(mono)

        # Calibrate initial noise floor to auto-pick threshold if needed
        if self._frames_seen <= self.calibrate_frames:
            self._calib_buf.append(level)
            if self._frames_seen == self.calibrate_frames:
                noise = np.percentile(self._calib_buf, 80)
                # threshold ~ noise + 8 dB, clamped
                self._threshold_db = max(self.threshold_db_default, noise + 8.0)
                log.info("VAD calibrated: noise=%.1f dBFS → threshold=%.1f dBFS", noise, self._threshold_db)
            return

        speaking = level > self._threshold_db

        if not self._active:
            if speaking:
                # start
                self._active = True
                self._cur.clear()
                self._cur_db.clear()
                self._silence_run = 0
                self._chunk_started_at = time.time()
                self._cur.append(mono.copy())
                self._cur_db.append(level)
        else:
            # append while active
            self._cur.append(mono.copy())
            self._cur_db.append(level)

            # silence tracking
            if speaking:
                self._silence_run = 0
            else:
                self._silence_run += 1

            # Decide to cut
            frames_len = len(self._cur)
            should_end = False
            reason = ""

            if frames_len >= self.max_frames:
                should_end = True
                reason = "max"
            elif frames_len >= self.min_frames and self._silence_run >= self.silence_frames:
                should_end = True
                reason = "silence"

            if should_end:
                arr = np.concatenate(list(self._cur)).astype(np.int16)
                self._active = False
                self._cur.clear()
                self._cur_db.clear()
                self._silence_run = 0

                # Emit bytes
                self.out_q.put(arr.tobytes())

                dur = frames_len * self.samples_per_frame / self.sr
                if self._chunk_started_at:
                    log.info("Segment finalized (%.2fs, reason=%s, start→cut=%.2fs)",
                             dur, reason, time.time() - self._chunk_started_at)
                self._chunk_started_at = None


# --------------------------- Player ---------------------------

class StreamPlayer:
    """Single output stream @ 48kHz mono float32; buffers audio chunks, crossfading slightly."""

    def __init__(self, sr_out: int = 48000, device: Optional[int] = None, fade_ms: int = 10):
        self.sr = sr_out
        self.device = device
        self.fade = int(sr_out * fade_ms / 1000)
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._cur: Optional[np.ndarray] = None
        self._idx = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._stream: Optional[sd.OutputStream] = None

    def start(self):
        self._stream = sd.OutputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=0,
            callback=self._cb
        )
        self._stream.start()

    def _cb(self, outdata, frames, time_info, status):
        if status:
            log.warning("Output status: %s", status)
        with self._lock:
            # if no current buffer, try to fetch one
            if self._cur is None or self._idx >= self._cur.shape[0]:
                try:
                    self._cur = self.q.get_nowait()
                    self._idx = 0
                except queue.Empty:
                    outdata[:] = 0.0
                    return

            end = min(self._idx + frames, self._cur.shape[0])
            chunk = self._cur[self._idx:end]
            out = np.zeros((frames,), dtype=np.float32)
            out[:chunk.shape[0]] = chunk

            self._idx = end

            # If we will finish inside this callback, try to prefetch and crossfade
            if self._idx >= self._cur.shape[0]:
                try:
                    nxt = self.q.get_nowait()
                    # Crossfade last 'fade' samples of current with first 'fade' of next
                    f = min(self.fade, chunk.shape[0], nxt.shape[0])
                    if f > 0:
                        # overlap region mapped to output tail
                        out[-f:] = (np.linspace(1.0, 0.0, f, dtype=np.float32) * out[-f:]
                                    + np.linspace(0.0, 1.0, f, dtype=np.float32) * nxt[:f])
                        nxt = nxt[f:]
                    self._cur = nxt if nxt.size else None
                    self._idx = 0
                except queue.Empty:
                    self._cur = None
                    self._idx = 0

        outdata[:, 0] = out

    def enqueue(self, y48: np.ndarray):
        if y48.ndim > 1:
            y48 = y48[:, 0]
        # safety clip
        y48 = np.clip(y48, -1.0, 1.0).astype(np.float32)
        self.q.put(y48)

    def stop(self):
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


# --------------------------- Pipeline Threads ---------------------------

def translator_worker(
    vad: EnergyVADSegmenter,
    ast: ASTModelManager,
    tts: TTSManager,
    src: str,
    tgt: str,
    out_player: StreamPlayer,
    in_sr: int,
    in_channels: int,
    tts_out_sr: int = 48000,
):
    """
    Pull finalized speech chunks from VAD, AST-translate them, TTS, and enqueue to player.
    """
    while True:
        try:
            pcm_bytes = vad.out_q.get(timeout=0.5)
        except queue.Empty:
            continue

        t0 = time.time()
        # Convert to 16k float32 mono (re-use your util)
        sr16, x16 = to_mono16k_float32(pcm_bytes, in_sr=in_sr, channels=in_channels)

        # AST inference (blocking)
        try:
            res = ast.translate(x16, src_lang=src, tgt_lang=tgt, timestamps=False)
            text = (res.text or "").strip()
        except Exception as e:
            log.exception("AST failed: %s", e)
            text = ""

        t1 = time.time()
        if not text:
            log.info("Empty translation (%.2fs end-to-end to AST). Skipping TTS.", t1 - t0)
            continue

        # TTS
        try:
            spk = None if tgt == "en" else "baya"  # simple default; override via your TTSManager if needed
            sr_tts, y = tts.synth(text, lang=tgt, speaker=spk)
        except Exception as e:
            log.exception("TTS failed: %s", e)
            continue

        # Resample to output SR for smooth playback mixing
        y48 = resample_float32(y, sr_tts, tts_out_sr)

        # Enqueue to speaker
        out_player.enqueue(y48)

        t2 = time.time()
        log.info("Chunk done | len_in=%.2fs | AST=%.2fs | TTS=%.2fs | total=%.2fs | text='%s'",
                 len(pcm_bytes) / (in_sr * in_channels * 2),
                 t1 - t0, t2 - t1, t2 - t0,
                 (text[:120] + "…") if len(text) > 120 else text)


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Real-time EN↔RU speech→speech (Canary AST + Silero TTS)")
    ap.add_argument("--src", choices=["en", "ru"], required=True)
    ap.add_argument("--tgt", choices=["en", "ru"], required=True)
    ap.add_argument("--device", default="cpu", help="Silero device for TTS: cpu/cuda")
    ap.add_argument("--in-dev", type=int, default=None, help="Input device index (sounddevice)")
    ap.add_argument("--out-dev", type=int, default=None, help="Output device index (sounddevice)")
    ap.add_argument("--in-ch", type=int, default=1, help="Mic channels (1 or 2)")
    ap.add_argument("--log", default="INFO")
    ap.add_argument("--threshold-db", type=float, default=-42.0, help="VAD threshold dBFS (higher = less sensitive)")
    ap.add_argument("--min-ms", type=int, default=600)
    ap.add_argument("--max-ms", type=int, default=2000)
    ap.add_argument("--silence-ms", type=int, default=220)
    args = ap.parse_args()

    logging.basicConfig(level=args.log.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.src == args.tgt:
        log.error("src and tgt must differ.")
        sys.exit(2)

    # Models
    log.info("Loading Canary AST…")
    ast = ASTModelManager()
    ast.load()

    tts = TTSManager(device=args.device)
    if args.tgt == "ru":
        tts.load("ru", "baya")
    else:
        tts.load("en", None)

    # VAD/Segmenter
    seg = EnergyVADSegmenter(
        sr=48000, channels=args.in_ch,
        threshold_db=args.threshold_db,
        min_chunk_ms=args.min_ms,
        max_chunk_ms=args.max_ms,
        silence_ms=args.silence_ms,
    )

    # Player
    player = StreamPlayer(sr_out=48000, device=args.out_dev)
    player.start()

    # Translator worker
    worker = threading.Thread(
        target=translator_worker,
        args=(seg, ast, tts, args.src, args.tgt, player, 48000, args.in_ch, 48000),
        daemon=True
    )
    worker.start()

    # Mic stream
    block_frames = seg.samples_per_frame  # 30ms @ 48k
    log.info("Opening mic @ 48kHz, block=%d frames, ch=%d (in_dev=%s, out_dev=%s)",
             block_frames, args.in_ch, args.in_dev, args.out_dev)

    qerr = queue.Queue()

    def on_audio_indata(indata, frames, time_info, status):
        if status:
            log.warning("Input status: %s", status)
        try:
            # indata shape: (frames, channels), dtype int16
            seg.push(indata.copy())
        except Exception as e:
            qerr.put(e)

    with sd.InputStream(
        samplerate=48000,
        channels=args.in_ch,
        dtype="int16",
        blocksize=block_frames,
        device=args.in_dev,
        callback=on_audio_indata,
    ):
        log.info("Streaming… Press Ctrl+C to stop.")
        stop = threading.Event()

        def handle_sigint(sig, frame):
            stop.set()
        signal.signal(signal.SIGINT, handle_sigint)

        while not stop.is_set():
            time.sleep(0.2)
            # bubble up callback errors
            if not qerr.empty():
                raise qerr.get()

    player.stop()
    log.info("Stopped.")

if __name__ == "__main__":
    main()
