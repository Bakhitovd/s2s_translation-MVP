import io
import numpy as np
import soundfile as sf
import pytest

from server.audio_util import to_mono16k_float32, wav_bytes_from_float32, AudioBuffer


def test_to_mono16k_float32_downsamples_and_mixes():
    # Create 48kHz stereo sine wave
    sr_in = 48000
    t = np.linspace(0, 1, sr_in, endpoint=False)
    sine_left = 0.5 * np.sin(2 * np.pi * 440 * t)
    sine_right = 0.5 * np.sin(2 * np.pi * 880 * t)
    stereo = np.stack([sine_left, sine_right], axis=1)
    pcm_bytes = (stereo * 32767).astype(np.int16).tobytes()

    sr_out, audio_out = to_mono16k_float32(pcm_bytes, in_sr=sr_in, channels=2)

    assert sr_out == 16000
    assert isinstance(audio_out, np.ndarray)
    assert audio_out.dtype == np.float32
    # Should be roughly 1/3 length due to resampling
    assert abs(len(audio_out) - (len(t) // 3)) < 10


def test_wav_bytes_from_float32_roundtrip():
    sr = 16000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    wav_bytes = wav_bytes_from_float32(sine, sr)
    buf = io.BytesIO(wav_bytes)
    data, sr_read = sf.read(buf, dtype="float32")

    assert sr_read == sr
    assert np.allclose(data[:100], sine[:100], atol=1e-3)


def test_audio_buffer_append_and_flush():
    buf = AudioBuffer()
    buf.append(b"abc")
    buf.append(b"def")
    assert buf.length() == 6
    data = buf.flush()
    assert data == b"abcdef"
    assert buf.length() == 0
