"""VAD 后特征提取。

从 VAD 段列表提取结构化特征：停顿密度、语速、语音连续性等，
为策略决策引擎提供数值输入。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from local_asr.asr.segmentation import VadSegment


@dataclass
class VADProfile:
    """VAD 后提取的结构化特征，用于策略决策。"""

    total_duration_s: float          # 音频总时长
    speech_ratio: float              # 有效语音占比 (0-1)
    num_segments: int                # VAD 段数
    num_gaps: int                    # 段间停顿数 (segments - 1)
    gaps_per_minute: float           # ★ 每分停顿密度
    mean_gap_ms: float               # 平均停顿时长
    median_gap_ms: float             # 中位数停顿时长
    gap_std_ms: float                # 停顿变异度
    mean_segment_s: float            # 平均语音段长
    min_segment_s: float             # 最短语音段
    max_segment_s: float             # 最长语音段
    is_continuous: bool              # 是否连续语音（朗读风格）


def extract_vad_profile(
    segments: Sequence[VadSegment],
    total_ms: int,
) -> VADProfile:
    """从 VAD 结果提取结构化特征。

    关键特征 gaps_per_minute：
    - 实测财经直播 5min: 44 gaps / 5.53min ≈ 8.0 gaps/min → 口语
    - 实测 LiPS 朗读 20min: ~286 gaps / 20min   ≈ 14.3 splits/min
      但实际停顿 >40ms 的只有 ~20 个 → 真正的 gaps/min ≈ 1.0
    - 实测播客 32min: ~31 个 >40ms gaps / 32min ≈ 1.0 gaps/min
      但 VAD 检测到的原始段间停顿 >390 个

    注意: gaps_per_minute 计算的是原始 VAD 段间停顿数（含短间隙），
    在策略决策时需要与 balanced_min_silence_ms=40 对齐，
    只统计 ≥40ms 的停顿。

    阈值来自实测:
    - < 3 gaps/min  → 朗读/演讲风格
    - 3-8 gaps/min  → 普通语音
    - > 8 gaps/min  → 对话/直播风格
    """
    if not segments:
        return VADProfile(
            total_duration_s=0, speech_ratio=0, num_segments=0,
            num_gaps=0, gaps_per_minute=0,
            mean_gap_ms=0, median_gap_ms=0, gap_std_ms=0,
            mean_segment_s=0, min_segment_s=0, max_segment_s=0,
            is_continuous=False,
        )

    total_s = total_ms / 1000.0
    seg_durations = [s.end_ms - s.start_ms for s in segments]
    gaps = [
        segments[i + 1].start_ms - segments[i].end_ms
        for i in range(len(segments) - 1)
    ]
    # 只统计有效停顿 (>0ms)
    positive_gaps = [g for g in gaps if g > 0]

    speech_total = sum(seg_durations)
    speech_ratio = speech_total / max(total_ms, 1)

    import statistics
    mean_gap = statistics.mean(positive_gaps) if positive_gaps else 0.0
    median_gap = statistics.median(positive_gaps) if positive_gaps else 0.0
    gap_std = statistics.stdev(positive_gaps) if len(positive_gaps) >= 2 else 0.0

    mean_seg = statistics.mean(seg_durations) / 1000.0 if seg_durations else 0.0
    min_seg = min(seg_durations) / 1000.0 if seg_durations else 0.0
    max_seg = max(seg_durations) / 1000.0 if seg_durations else 0.0

    # gaps_per_minute: 只统计 ≥40ms（与 balanced_min_silence_ms 对齐）
    significant_gaps = [g for g in positive_gaps if g >= 40]
    gaps_per_min = len(significant_gaps) / (total_s / 60.0) if total_s > 0 else 0.0

    # 连续性判断: 停顿密度低 + 语音段长且均匀
    is_continuous = (
        gaps_per_min < 3
        and mean_seg > 8.0
        and len(significant_gaps) < len(positive_gaps) * 0.5
    )

    return VADProfile(
        total_duration_s=total_s,
        speech_ratio=speech_ratio,
        num_segments=len(segments),
        num_gaps=len(positive_gaps),
        gaps_per_minute=round(gaps_per_min, 2),
        mean_gap_ms=round(mean_gap, 1),
        median_gap_ms=round(median_gap, 1),
        gap_std_ms=round(gap_std, 1),
        mean_segment_s=round(mean_seg, 1),
        min_segment_s=round(min_seg, 1),
        max_segment_s=round(max_seg, 1),
        is_continuous=is_continuous,
    )