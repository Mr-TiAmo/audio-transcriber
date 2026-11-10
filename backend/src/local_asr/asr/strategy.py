"""策略决策引擎。

基于音频特征和 VAD 分析，自动决定最优切片策略。

决策树从实测数据中归纳：
- 财经直播 5min: gaps=8.0/min, balanced 60 比 silence 差 +0.60pp, jitter 拉回
- LiPS 朗读 20min: gaps≈1.0/min, balanced 60 仅差 +0.11pp, jitter 无效
- 播客 32min: gaps≈1.0/min (≥40ms), balanced 60 加速 1.98×
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from local_asr.asr.features import VADProfile
from local_asr.asr.quality import AudioProfile


@dataclass
class StrategyDecision:
    """策略决策结果——完整描述应使用的切片方案。"""

    chunk_strategy: str         # "silence" | "balanced"（"dp" 由调用方显式指定）
    target_chunk_s: float       # 目标块长 (秒)
    balance_tolerance: float    # balanced 容差比例
    balance_jitter: float       # jitter 秒数
    use_batch: bool             # 是否开启批量推理
    batch_max: int              # 批大小上限
    reason: str                 # 决策理由（可追溯）


# 决策阈值（来自实测校准）
SHORT_AUDIO_S = 300             # ≤5min 视为短音频
DENSE_GAPS_PER_MIN = 8.0        # ≥8 gaps/min → 对话/直播风格
SPARSE_GAPS_PER_MIN = 3.0       # <3 gaps/min → 朗读/演讲风格
DEFAULT_TARGET_S = 60.0         # 默认目标块长
SILENCE_TARGET_S = 120.0        # silence 策略目标块长
DEFAULT_BATCH_MAX = 4           # 默认批大小 (16GB 机器)


def decide_strategy(
    profile: AudioProfile,
    vad: VADProfile,
) -> StrategyDecision:
    """基于音频特征和 VAD 分析，自动决定最优切片策略。

    决策逻辑（三层决策树）：

    第一层——时长：
    - ≤5min → silence 120（优先质量，块数少不追求 batch）

    第二层——停顿密度（核心特征）：
    - <3 gaps/min → 朗读/演讲：balanced 60，不开 jitter
      （停顿太稀，jitter 吸不到，实测 0-5% 吸附率）
    - ≥8 gaps/min → 对话/直播：balanced 60 + jitter 1.0
      （停顿丰富，jitter 吸到 15-40%，消除 +0.60pp 回声）

    第三层——中间地带（3-8 gaps/min）：
    - balanced 60，不开 jitter

    所有 balanced 路径统一开 batch 4。
    """
    dur = profile.duration_s
    gaps = vad.gaps_per_minute

    # ── 第一层：时长 ──
    if dur <= SHORT_AUDIO_S:
        return StrategyDecision(
            chunk_strategy="silence",
            target_chunk_s=SILENCE_TARGET_S,
            balance_tolerance=0.0,
            balance_jitter=0.0,
            use_batch=False,
            batch_max=1,
            reason=(
                f"短音频({dur:.0f}s≤{SHORT_AUDIO_S}s)，"
                f"用 silence {SILENCE_TARGET_S:.0f}s 优先质量"
            ),
        )

    # ── 第二层：停顿密度 ──
    if gaps < SPARSE_GAPS_PER_MIN:
        return StrategyDecision(
            chunk_strategy="balanced",
            target_chunk_s=DEFAULT_TARGET_S,
            balance_tolerance=0.0,
            balance_jitter=0.0,
            use_batch=True,
            batch_max=DEFAULT_BATCH_MAX,
            reason=(
                f"朗读风格(gaps={gaps:.1f}/min<{SPARSE_GAPS_PER_MIN})，"
                f"停顿稀疏，balanced {DEFAULT_TARGET_S:.0f}s 不加 jitter"
            ),
        )

    if gaps >= DENSE_GAPS_PER_MIN:
        return StrategyDecision(
            chunk_strategy="balanced",
            target_chunk_s=DEFAULT_TARGET_S,
            balance_tolerance=0.0,
            balance_jitter=1.0,
            use_batch=True,
            batch_max=DEFAULT_BATCH_MAX,
            reason=(
                f"口语风格(gaps={gaps:.1f}/min≥{DENSE_GAPS_PER_MIN})，"
                f"停顿密集，balanced {DEFAULT_TARGET_S:.0f}s + jitter 1.0"
            ),
        )

    # ── 第三层：中间地带 ──
    return StrategyDecision(
        chunk_strategy="balanced",
        target_chunk_s=DEFAULT_TARGET_S,
        balance_tolerance=0.0,
        balance_jitter=0.0,
        use_batch=True,
        batch_max=DEFAULT_BATCH_MAX,
        reason=(
            f"普通语音(gaps={gaps:.1f}/min)，"
            f"balanced {DEFAULT_TARGET_S:.0f}s"
        ),
    )


def apply_decision_to_config(
    decision: StrategyDecision,
    config: "SegmentationConfig",
) -> "SegmentationConfig":
    """将策略决策应用到 SegmentationConfig，返回新实例。"""
    from dataclasses import replace
    return replace(
        config,
        chunk_strategy=decision.chunk_strategy,
        target_chunk_seconds=decision.target_chunk_s,
        balance_tolerance=decision.balance_tolerance,
        balance_jitter_seconds=decision.balance_jitter,
    )