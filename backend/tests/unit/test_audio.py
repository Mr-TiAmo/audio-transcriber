"""local_asr.audio 共享音频工具的单测。

这些测试原先挂在 test_postprocess_pipeline.py 下、通过 importlib 加载
脚本间接测试；音频函数下沉到 local_asr.audio 后迁移至此直接测试，
不再依赖脚本命名空间。
"""

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

from local_asr.audio import (
    _pcm_bytes,
    audio_duration,
    can_convert_audio,
    is_pcm16k_mono_wav,
    prepare_pcm16k,
    slice_wav,
)


def _write_wav(path: Path, *, rate: int = 16000, channels: int = 1,
               sampwidth: int = 2, frames: bytes = b"\0\0" * 100) -> Path:
    """生成一个参数可控的 PCM WAV，供各测试复用。"""
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(sampwidth)
        stream.setframerate(rate)
        stream.writeframes(frames)
    return path


# ---------------------------------------------------------------------------
# _pcm_bytes：PyAV 帧 padding 回归
# ---------------------------------------------------------------------------

def test_pcm_bytes_strips_frame_padding():
    """PyAV 的 planes[0] 按 64 字节对齐，直接写出会把音频拉长。

    s16 单声道每样本 2 字节，samples=100 时有效数据只有 200 字节，
    缓冲区尾部是对齐 padding，必须截掉。
    """
    frame = SimpleNamespace(planes=[bytes(range(256)) * 2], samples=100)
    assert _pcm_bytes(frame) == frame.planes[0][:200]


# ---------------------------------------------------------------------------
# audio_duration
# ---------------------------------------------------------------------------

def test_audio_duration_reads_wav_via_wave(tmp_path):
    """规范 WAV 走标准库 wave 分支：1 秒 16 kHz 单声道。"""
    audio = _write_wav(tmp_path / "one_sec.wav", frames=b"\0\0" * 16000)
    assert audio_duration(audio) == 1.0


# ---------------------------------------------------------------------------
# is_pcm16k_mono_wav
# ---------------------------------------------------------------------------

def test_is_pcm16k_mono_wav_accepts_target_format(tmp_path):
    audio = _write_wav(tmp_path / "ok.wav")
    assert is_pcm16k_mono_wav(audio) is True


def test_is_pcm16k_mono_wav_rejects_wrong_rate(tmp_path):
    audio = _write_wav(tmp_path / "8k.wav", rate=8000)
    assert is_pcm16k_mono_wav(audio) is False


def test_is_pcm16k_mono_wav_rejects_stereo(tmp_path):
    audio = _write_wav(tmp_path / "stereo.wav", channels=2, frames=b"\0\0\0\0" * 100)
    assert is_pcm16k_mono_wav(audio) is False


def test_is_pcm16k_mono_wav_rejects_non_wav(tmp_path):
    """非 WAV 内容（wave 打不开）按「不是目标格式」处理，不抛异常。"""
    bogus = tmp_path / "bogus.wav"
    bogus.write_bytes(b"not a riff file at all")
    assert is_pcm16k_mono_wav(bogus) is False


# ---------------------------------------------------------------------------
# prepare_pcm16k
# ---------------------------------------------------------------------------

def test_prepare_pcm16k_keeps_target_format_input(tmp_path):
    """已是目标格式时直接返回原路径，不产生任何副本。"""
    audio = _write_wav(tmp_path / "ok.wav")
    assert prepare_pcm16k(audio, tmp_path / "out.wav") == audio
    assert not (tmp_path / "out.wav").exists()


# ---------------------------------------------------------------------------
# slice_wav
# ---------------------------------------------------------------------------

def test_slice_wav_extracts_requested_range(tmp_path):
    """按毫秒区间切片：1 秒音频取 [250ms, 750ms) 应得 0.5 秒。"""
    src = _write_wav(tmp_path / "full.wav", frames=bytes(range(256)) * 125)  # 1s
    dst = tmp_path / "part.wav"
    slice_wav(src, 250, 750, dst)
    with wave.open(str(dst), "rb") as part:
        assert part.getframerate() == 16000
        assert part.getnchannels() == 1
        assert part.getsampwidth() == 2
        assert part.getnframes() == 8000  # 0.5s * 16000
        # 内容应等于源文件第 4000~12000 帧（250ms 起）
        assert part.readframes(8000) == (bytes(range(256)) * 125)[8000:24000]


def test_slice_wav_clamps_end_beyond_duration(tmp_path):
    """end_ms 超出文件时长时 readframes 自动截断，不报错。"""
    src = _write_wav(tmp_path / "full.wav", frames=b"\0\0" * 16000)  # 1s
    dst = tmp_path / "part.wav"
    slice_wav(src, 500, 9999, dst)
    with wave.open(str(dst), "rb") as part:
        assert part.getnframes() == 8000


# ---------------------------------------------------------------------------
# can_convert_audio
# ---------------------------------------------------------------------------

def test_can_convert_audio_returns_bool():
    """只校验返回类型：结果取决于本机是否装了 ffmpeg/PyAV。"""
    assert isinstance(can_convert_audio(), bool)
