"""测试 strategy.py 决策引擎的 6 个决策路径。"""
import pytest
from local_asr.asr.features import VADProfile
from local_asr.asr.quality import AudioProfile
from local_asr.asr.strategy import (
    SPARSE_GAPS_PER_MIN,
    DENSE_GAPS_PER_MIN,
    SHORT_AUDIO_S,
    decide_strategy,
    apply_decision_to_config,
)
from local_asr.asr.segmentation import SegmentationConfig


def _ap(duration_s=600.0, zcr=0.2, is_studio=True) -> AudioProfile:
    return AudioProfile(
        duration_s=duration_s, sample_rate=16000, channels=1,
        rms_energy=1000.0, zero_crossing_rate=zcr, peak_ratio=12.0,
        is_studio=is_studio,
    )


def _vp(gaps_per_minute=5.0, is_continuous=False) -> VADProfile:
    return VADProfile(
        total_duration_s=600, speech_ratio=0.9,
        num_segments=50, num_gaps=49,
        gaps_per_minute=gaps_per_minute,
        mean_gap_ms=200.0, median_gap_ms=150.0, gap_std_ms=100.0,
        mean_segment_s=10.0, min_segment_s=2.0, max_segment_s=30.0,
        is_continuous=is_continuous,
    )


class TestDecideStrategy:
    """6 个决策路径的单元测试。"""

    def test_short_audio_always_silence(self):
        """短音频（≤5min）：无论停顿密度，一律 silence 120。"""
        for gpm in [0.5, 3.0, 10.0, 20.0]:
            dec = decide_strategy(_ap(duration_s=180), _vp(gaps_per_minute=gpm))
            assert dec.chunk_strategy == "silence"
            assert dec.target_chunk_s == 120.0
            assert dec.balance_jitter == 0.0
            assert not dec.use_batch

    def test_short_audio_boundary(self):
        """刚好 300s 仍算短音频。"""
        dec = decide_strategy(_ap(duration_s=300), _vp(gaps_per_minute=10))
        assert dec.chunk_strategy == "silence"

    def test_just_over_short_boundary_not_silence(self):
        """301s 不算短音频，走正常决策。"""
        dec = decide_strategy(_ap(duration_s=301), _vp(gaps_per_minute=5))
        assert dec.chunk_strategy == "balanced"

    def test_sparse_gaps_balanced_no_jitter(self):
        """朗读风格（<3 gaps/min）：balanced 60，不开 jitter。"""
        dec = decide_strategy(_ap(), _vp(gaps_per_minute=1.5))
        assert dec.chunk_strategy == "balanced"
        assert dec.target_chunk_s == 60.0
        assert dec.balance_jitter == 0.0
        assert dec.use_batch
        assert dec.batch_max == 4

    def test_dense_gaps_balanced_with_jitter(self):
        """口语风格（≥8 gaps/min）：balanced 60 + jitter 1.0。"""
        dec = decide_strategy(_ap(), _vp(gaps_per_minute=10.0))
        assert dec.chunk_strategy == "balanced"
        assert dec.balance_jitter == 1.0
        assert dec.use_batch
        assert dec.batch_max == 4

    def test_dense_gaps_boundary(self):
        """刚好 8.0 gaps/min 触发 jitter。"""
        dec = decide_strategy(_ap(), _vp(gaps_per_minute=8.0))
        assert dec.balance_jitter == 1.0

    def test_medium_gaps_balanced_no_jitter(self):
        """中间地带（3-8 gaps/min）：balanced 60，不开 jitter。"""
        dec = decide_strategy(_ap(), _vp(gaps_per_minute=5.0))
        assert dec.chunk_strategy == "balanced"
        assert dec.balance_jitter == 0.0
        assert dec.use_batch

    def test_decision_has_reason(self):
        """每个决策都有可追溯的理由。"""
        dec = decide_strategy(_ap(), _vp(gaps_per_minute=10))
        assert len(dec.reason) > 10

    def test_all_paths_covered(self):
        """覆盖全部 6 条路径。"""
        # 短音频 × 任意停顿 → silence
        assert decide_strategy(_ap(180), _vp(0.5)).chunk_strategy == "silence"
        assert decide_strategy(_ap(180), _vp(10)).chunk_strategy == "silence"

        # 长音频 + 稀疏 → balanced no jitter
        assert decide_strategy(_ap(600), _vp(1.0)).balance_jitter == 0.0

        # 长音频 + 密集 → balanced + jitter
        assert decide_strategy(_ap(600), _vp(12.0)).balance_jitter == 1.0

        # 长音频 + 中间 → balanced no jitter
        assert decide_strategy(_ap(600), _vp(5.0)).balance_jitter == 0.0

        # 边界：刚好不短
        assert decide_strategy(_ap(301), _vp(1.0)).chunk_strategy == "balanced"


class TestApplyDecision:
    """测试决策应用到 SegmentationConfig。"""

    def test_silence_decision_overrides_config(self):
        cfg = SegmentationConfig(
            chunk_strategy="balanced", target_chunk_seconds=60,
            balance_tolerance=0.0, balance_jitter_seconds=1.0,
        )
        from local_asr.asr.strategy import StrategyDecision
        dec = StrategyDecision(
            chunk_strategy="silence", target_chunk_s=120,
            balance_tolerance=0.0, balance_jitter=0.0,
            use_batch=False, batch_max=1, reason="test",
        )
        new = apply_decision_to_config(dec, cfg)
        assert new.chunk_strategy == "silence"
        assert new.target_chunk_seconds == 120
        assert new.balance_jitter_seconds == 0.0

    def test_balanced_jitter_decision_overrides_config(self):
        cfg = SegmentationConfig()
        from local_asr.asr.strategy import StrategyDecision
        dec = StrategyDecision(
            chunk_strategy="balanced", target_chunk_s=60,
            balance_tolerance=0.0, balance_jitter=1.0,
            use_batch=True, batch_max=4, reason="test",
        )
        new = apply_decision_to_config(dec, cfg)
        assert new.chunk_strategy == "balanced"
        assert new.balance_jitter_seconds == 1.0


class TestExtractVadProfileUsesRawSegments:
    """回归测试：extract_vad_profile 必须用原始 VAD 段，
    不能先 merge_short_gaps 再提取特征——否则停顿密度信号被抹掉。"""

    def test_merged_segments_have_fewer_gaps_than_raw(self):
        """merge_short_gaps 合并后间隙数应远小于原始间隙数。"""
        from local_asr.asr.segmentation import VadSegment, merge_short_gaps
        from local_asr.asr.features import extract_vad_profile

        # 模拟对话音频：8 个短段，段间有短间隙（200-600ms）
        raw = [
            VadSegment(0, 2000),     # 2s
            VadSegment(2500, 4000),  # 500ms gap → 合并
            VadSegment(4300, 6000),  # 300ms gap → 合并
            VadSegment(6800, 8500),  # 800ms gap → 边界，合并
            VadSegment(9000, 11000), # 500ms gap → 合并
            VadSegment(12000, 14000),# 1000ms gap → 不合并
            VadSegment(16000, 18000),# 2000ms gap → 不合并
            VadSegment(19000, 21000),# 1000ms gap → 不合并
        ]
        total_ms = 22000

        merged = merge_short_gaps(raw)
        raw_profile = extract_vad_profile(raw, total_ms)
        merged_profile = extract_vad_profile(merged, total_ms)

        # 原始 7 个间隙（8 段）
        assert raw_profile.num_gaps == 7
        # 合并后应显著减少（≤3 段 → ≤2 间隙）
        assert merged_profile.num_gaps < raw_profile.num_gaps
        # 合并后的 gaps/min 不应被用于策略决策
        assert merged_profile.gaps_per_minute < raw_profile.gaps_per_minute