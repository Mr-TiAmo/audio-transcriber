"""音频质量快速检测（纯标准库，不依赖 numpy/scipy）。

在 VAD 之前运行，提取 RMS 能量、过零率、峰值比等时域特征，
用于判断录音类型（录音棚/现场/嘈杂），为策略决策提供输入。

当前 ``decide_strategy`` 只消费 ``duration_s``；其余字段是预留特征，
供后续按录音环境扩展决策维度（如嘈杂环境提高 overlap），计算成本
O(n)（1200s 16k mono PCM 约 0.3s），保留不构成负担。
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioProfile:
    """VAD 前的快速音频特征。"""

    duration_s: float
    sample_rate: int
    channels: int
    rms_energy: float          # RMS 能量，<0.001 几乎静音
    zero_crossing_rate: float  # 过零率（口语 > 朗读）
    peak_ratio: float          # 峰值/RMS 比（动态范围）
    is_studio: bool            # 是否疑似录音棚（低噪声、高动态范围）


def _read_pcm_frames(path: Path) -> tuple[bytes, int, int, int]:
    """读取 PCM WAV → (raw bytes, nchannels, sampwidth, framerate)。"""
    wf = wave.open(str(path))
    nch = wf.getnchannels()
    sw = wf.getsampwidth()
    rate = wf.getframerate()
    nframes = wf.getnframes()
    data = wf.readframes(nframes)
    wf.close()
    return data, nch, sw, rate


def profile_audio(pcm_path: Path, *, frame_ms: int = 25) -> AudioProfile:
    """读取 16k mono PCM WAV，计算快速音频特征。

    不做 FFT（避免依赖 scipy），仅用时域分析：
    - RMS 能量：帧级标准差
    - 过零率：相邻采样点符号变化率
    - 峰值比：峰值振幅 / RMS（高值 → 高动态范围 → 录音棚）
    """
    data, nch, sw, rate = _read_pcm_frames(pcm_path)
    duration = len(data) / (nch * sw * rate) if rate > 0 else 0.0

    # 解码为 int16 数组（Python 内存视图，不做拷贝）
    fmt = {2: "<h", 4: "<i"}.get(sw, "<h")
    samples = len(data) // sw
    frame_samples = int(rate * frame_ms / 1000)
    num_frames = max(1, samples // frame_samples)

    rms_values: list[float] = []
    zcr_values: list[float] = []
    peak = 0.0

    for i in range(num_frames):
        start = i * frame_samples
        end = min(start + frame_samples, samples)
        chunk = data[start * sw : end * sw]
        vals = [
            struct.unpack(fmt, chunk[j : j + sw])[0]
            for j in range(0, len(chunk), sw)
        ]
        if not vals:
            continue
        # RMS
        rms = math.sqrt(sum(v * v for v in vals) / len(vals))
        rms_values.append(rms)
        # ZCR
        crossings = sum(
            1 for a, b in zip(vals, vals[1:]) if (a >= 0) != (b >= 0)
        )
        zcr_values.append(crossings / max(1, len(vals) - 1))
        # Peak
        frame_peak = max(abs(v) for v in vals)
        if frame_peak > peak:
            peak = frame_peak

    avg_rms = sum(rms_values) / max(1, len(rms_values))
    avg_zcr = sum(zcr_values) / max(1, len(zcr_values))
    peak_ratio = peak / max(avg_rms, 1.0)

    # 录音棚判断：低噪声（RMS 标准差小）+ 高动态范围（peak/RMS 大）
    rms_std = math.sqrt(
        sum((v - avg_rms) ** 2 for v in rms_values) / max(1, len(rms_values))
    )
    is_studio = (rms_std < avg_rms * 0.5) and (peak_ratio > 8) and (avg_rms > 100)

    return AudioProfile(
        duration_s=duration,
        sample_rate=rate,
        channels=nch,
        rms_energy=avg_rms,
        zero_crossing_rate=avg_zcr,
        peak_ratio=peak_ratio,
        is_studio=is_studio,
    )