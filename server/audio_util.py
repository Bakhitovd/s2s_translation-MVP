import io
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def to_mono16k_float32(pcm_bytes: bytes, in_sr: int = 48000, channels: int = 2):
    """
    Convert raw int16 PCM bytes (from AudioWorklet, 48kHz stereo) into mono 16kHz float32 numpy array.

    Args:
        pcm_bytes: Raw PCM16 bytes (little-endian).
        in_sr: Input sample rate (default 48000).
        channels: Number of channels in input (default 2).

    Returns:
        (sr, audio_np): Tuple of target sample rate (16000) and mono float32 numpy array.
    """
    if len(pcm_bytes) % 2 != 0:
        raise ValueError("PCM byte length must be multiple of 2")

    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

    if channels > 1:
        audio_int16 = audio_int16.reshape(-1, channels)
        audio_int16 = audio_int16.mean(axis=1).astype(np.int16)

    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    # Resample from in_sr to 16k
    gcd = np.gcd(in_sr, 16000)
    up = 16000 // gcd
    down = in_sr // gcd
    audio_resampled = resample_poly(audio_float32, up, down).astype(np.float32)

    return 16000, audio_resampled


def wav_bytes_from_float32(audio_np: np.ndarray, sr: int) -> bytes:
    """
    Encode a float32 numpy array as WAV PCM16 bytes.

    Args:
        audio_np: Float32 numpy array (-1.0 to 1.0).
        sr: Sample rate.

    Returns:
        WAV file bytes.
    """
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class AudioBuffer:
    """
    Small utility around bytearray for clearer semantics and testing.
    """

    def __init__(self):
        self._buf = bytearray()

    def append(self, bytes_chunk: bytes):
        self._buf.extend(bytes_chunk)

    def flush(self) -> bytes:
        data = bytes(self._buf)
        self._buf.clear()
        return data

    def length(self) -> int:
        return len(self._buf)
