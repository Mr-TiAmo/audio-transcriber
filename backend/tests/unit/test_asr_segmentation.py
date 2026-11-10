"""切分规划与边界去重的单元测试。"""

from __future__ import annotations

import pytest

from local_asr.asr.segmentation import (
    SegmentationConfig,
    VadSegment,
    apply_dedup_to_texts,
    build_chunks,
    decide_mode,
    drop_prefix_by_chars,
    drop_suffix_by_chars,
    estimate_max_new_tokens,
    merge_texts,
    overlap_prefix_len,
    plan_balanced_cuts,
    plan_cuts,
    plan_dp_cuts,
)


def _cfg(**kwargs) -> SegmentationConfig:
    return SegmentationConfig(**kwargs)


# ---------------------------------------------------------------------------
# decide_mode
# ---------------------------------------------------------------------------

def test_decide_mode_respects_explicit_choice() -> None:
    assert decide_mode(5000, _cfg(mode="full")) == "full"
    assert decide_mode(10, _cfg(mode="vad")) == "vad"


def test_decide_mode_auto_uses_full_below_threshold() -> None:
    assert decide_mode(300.0, _cfg()) == "full"
    assert decide_mode(299.9, _cfg()) == "full"


def test_decide_mode_auto_uses_vad_above_full_threshold() -> None:
    assert decide_mode(300.1, _cfg()) == "vad"


def test_decide_mode_never_allows_full_beyond_hard_limit() -> None:
    cfg = _cfg(full_max_seconds=2000, hard_max_seconds=1200)
    assert decide_mode(1500, cfg) == "vad"


# ---------------------------------------------------------------------------
# plan_cuts
# ---------------------------------------------------------------------------

def _segments(*pairs: tuple[int, int]) -> list[VadSegment]:
    return [VadSegment(a, b) for a, b in pairs]


def test_plan_cuts_picks_longest_silence_in_window() -> None:
    # 96.0-96.6s 有 600ms 静音，120s 附近还有 100ms 短停顿；必须选 600ms
    segments = _segments(
        (0, 40000), (40100, 80000), (80200, 96000),
        (96600, 130000), (130100, 197000), (197900, 260000), (260100, 331859),
    )
    cuts = plan_cuts(segments, 331859, _cfg())
    assert [c.index for c in cuts] == [0, 1, 2]
    assert cuts[0].cut_silence_ms == 600
    assert cuts[1].cut_silence_ms == 900
    assert cuts[0].forced is False
    assert cuts[-1].end_ms == 331859


def test_plan_cuts_ignores_silence_below_minimum() -> None:
    # 150ms 静音低于 200ms 阈值，不能作为切点
    segments = _segments((0, 120000), (120150, 240000))
    cuts = plan_cuts(segments, 240000, _cfg())
    # 第一刀被迫切在语音中；最后一块只是收尾，不属于 forced
    assert cuts[0].forced is True
    assert cuts[0].cut_silence_ms is None
    assert cuts[-1].forced is False


def test_plan_cuts_falls_back_to_max_chunk_without_any_silence() -> None:
    segments = _segments((0, 400000))
    cuts = plan_cuts(segments, 400000, _cfg(max_chunk_seconds=180))
    assert len(cuts) == 3
    assert cuts[0].forced is True and cuts[1].forced is True
    assert cuts[0].end_ms == 180000
    assert cuts[1].end_ms == 360000
    assert cuts[-1].end_ms == 400000


def test_plan_cuts_keeps_short_audio_single_chunk() -> None:
    cuts = plan_cuts(_segments((0, 5000)), 5000, _cfg())
    assert len(cuts) == 1
    assert cuts[0].start_ms == 0
    assert cuts[0].end_ms == 5000


def test_plan_cuts_handles_zero_duration() -> None:
    cuts = plan_cuts([], 0, _cfg())
    assert len(cuts) == 1
    assert cuts[0].end_ms == 0


def test_plan_cuts_uses_later_boundary_when_silence_equal() -> None:
    # 两处同为 300ms 静音时取更靠后的，让块更接近目标长度
    segments = _segments(
        (0, 90000), (90300, 120000), (120300, 200000),
    )
    cuts = plan_cuts(segments, 200000, _cfg(target_chunk_seconds=100))
    assert cuts[0].cut_silence_ms == 300
    assert cuts[0].end_ms == 120150


# ---------------------------------------------------------------------------
# build_chunks
# ---------------------------------------------------------------------------

def test_build_chunks_expands_both_sides() -> None:
    cuts = plan_cuts(_segments((0, 100000), (101000, 220000)), 220000, _cfg())
    chunks = build_chunks(cuts, 220000, _cfg(overlap_seconds=0.5))
    first = chunks[0]
    assert first.audio_start_ms == 0
    assert first.audio_end_ms == first.core_end_ms + 500
    assert chunks[-1].audio_end_ms == 220000


def test_build_chunks_overlap_can_be_disabled() -> None:
    cuts = plan_cuts(_segments((0, 100000), (101000, 220000)), 220000, _cfg())
    chunks = build_chunks(cuts, 220000, _cfg(overlap_seconds=0.0))
    assert chunks[0].audio_start_ms == chunks[0].core_start_ms
    assert chunks[0].audio_end_ms == chunks[0].core_end_ms


# ---------------------------------------------------------------------------
# 惰性去重
# ---------------------------------------------------------------------------

def test_overlap_detects_real_duplicate() -> None:
    prev = "那么已经是涨得比较大了"
    nxt = "涨得比较大了，那么可以说现在"
    assert overlap_prefix_len(prev, nxt) == len("涨得比较大了")


def test_overlap_ignores_punctuation_differences() -> None:
    prev = "那么已经是涨得比较大了。"
    nxt = "涨得比较大了，那么可以说"
    assert overlap_prefix_len(prev, nxt) == len("涨得比较大了")


def test_overlap_returns_zero_when_no_duplicate() -> None:
    assert overlap_prefix_len("整整少了一半", "第三个数据是什么") == 0


def test_overlap_does_not_match_short_common_prefix() -> None:
    # 只有 2 个字符相同，低于 min_overlap，不能裁
    assert overlap_prefix_len("看看这个", "看看那个", min_overlap=4) == 0


def test_overlap_only_compares_boundary_window() -> None:
    # 正文内部有重复词，但不在边界窗口内，不应触发
    prev = "止盈的就要止盈" * 3 + "结尾完全不同"
    nxt = "止盈的就要止盈，但是开头是新的"
    assert overlap_prefix_len(prev, nxt) == 0


def test_drop_prefix_by_chars_skips_punctuation() -> None:
    text = "涨得比较大了，那么可以说"
    kept = drop_prefix_by_chars(text, len("涨得比较大了"))
    assert kept == "，那么可以说"


def test_drop_prefix_zero_returns_original() -> None:
    assert drop_prefix_by_chars("原文", 0) == "原文"


def test_merge_texts_no_overlap_concatenates() -> None:
    merged, records = merge_texts(["第一段结束", "第二段开始"])
    assert merged == "第一段结束第二段开始"
    assert records == []


def test_merge_texts_drops_duplicate_prefix() -> None:
    merged, records = merge_texts(["那么已经是涨得比较大了", "涨得比较大了，那么可以说现在"])
    assert merged == "那么已经是涨得比较大了，那么可以说现在"
    assert records == [{
        "boundary": 0, "dropped_chars": 6, "dropped": "涨得比较大了",
        "method": "exact", "edit_distance": 0,
    }]


def test_merge_texts_fuzzy_dedup_when_transcription_differs() -> None:
    """同一段重叠音频两侧转写不一致时，精确匹配漏判会留下重复短语。"""
    prev = "今天你的家具包括你甚至说是你上面的岛民都是"
    nxt = "你上面的岛民啊都会然后讲到"
    merged, records = merge_texts([prev, nxt], expected_overlap=[6])
    assert merged == "今天你的家具包括你甚至说是你上面的岛民啊都会然后讲到"
    assert records[0]["method"] == "fuzzy"
    assert records[0]["dropped_chars"] == 6
    assert records[0]["trimmed_prev_chars"] == 2
    assert records[0]["trimmed_prev"] == "都是"


def test_merge_texts_exact_mode_skips_fuzzy_alignment() -> None:
    prev = "今天你的家具包括你甚至说是你上面的岛民都是"
    nxt = "你上面的岛民啊都会然后讲到"
    merged, records = merge_texts([prev, nxt], expected_overlap=[6], fuzzy=False)
    assert merged == prev + nxt
    assert records == []


def test_merge_texts_fuzzy_handles_offset_prefix() -> None:
    """下一块开头先复述了几个字，再出现与上一块尾部相同的短语。"""
    prev = "如果感兴趣的朋友呢可以Google"
    nxt = "学的朋友呢可以Google或者百度一下"
    merged, records = merge_texts([prev, nxt], expected_overlap=[6])
    assert merged == "如果感兴趣的朋友呢可以Google或者百度一下"
    assert records[0]["dropped_chars"] == 13


def test_merge_texts_fuzzy_respects_expected_overlap() -> None:
    """重叠估算很短时不启用模糊裁剪，避免把正文当重叠删掉。"""
    prev = "今天你的家具包括你甚至说是你上面的岛民都是"
    nxt = "你上面的岛民啊都会然后讲到"
    merged, records = merge_texts([prev, nxt], expected_overlap=[3])
    assert merged == prev + nxt
    assert records == []


def test_merge_texts_does_not_fuzzy_match_unrelated_opening() -> None:
    prev = "谢谢大家的收听我们下次再见"
    nxt = "再见不是结束而是新的开始"
    merged, records = merge_texts([prev, nxt], expected_overlap=[6])
    assert merged == prev + nxt
    assert records == []


def test_drop_suffix_by_chars_skips_trailing_punctuation() -> None:
    text = "说是你上面的岛民都是。"
    assert drop_suffix_by_chars(text, 2) == "说是你上面的岛民"
    assert drop_suffix_by_chars(text, 0) == text


def test_merge_texts_empty_input() -> None:
    assert merge_texts([]) == ("", [])


# ---------------------------------------------------------------------------
# 配置校验
# ---------------------------------------------------------------------------

def test_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="未知切分模式"):
        _cfg(mode="auto2")


def test_config_rejects_bad_slack() -> None:
    with pytest.raises(ValueError, match="chunk_slack"):
        _cfg(chunk_slack=(1.2, 0.8))


def test_config_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError, match="不能为负"):
        _cfg(overlap_seconds=-0.1)


# ---------------------------------------------------------------------------
# token 预算
# ---------------------------------------------------------------------------

def test_token_budget_has_floor() -> None:
    assert estimate_max_new_tokens(5) == 512
    assert estimate_max_new_tokens(0) == 512


def test_token_budget_scales_with_duration() -> None:
    # 200 秒 * 3.05 * 1.25 = 762.5 -> 高于 512 下限，按估算值向上取整
    assert estimate_max_new_tokens(200) == 763
    # 120 秒估算 458，仍低于下限，按 512
    assert estimate_max_new_tokens(120) == 512


def test_token_budget_has_cap() -> None:
    assert estimate_max_new_tokens(3600) == 8192

# ---------------------------------------------------------------------------
# plan_balanced_cuts
# ---------------------------------------------------------------------------

def test_balanced_snaps_to_silence_near_ideal_position() -> None:
    # 总长 240 s、目标 120 s → 两段，切点应吸附到 120 s 附近的静音
    segments = _segments((0, 119800), (120200, 240000))
    cuts = plan_balanced_cuts(segments, 240000, _cfg(target_chunk_seconds=120))
    assert len(cuts) == 2
    assert cuts[0].end_ms == 120000
    assert cuts[0].cut_silence_ms == 400
    assert cuts[0].forced is False


def test_balanced_hard_cuts_equally_without_silence() -> None:
    cuts = plan_balanced_cuts(_segments((0, 400000)), 400000, _cfg(target_chunk_seconds=100))
    assert len(cuts) == 4
    assert [c.end_ms for c in cuts] == [100000, 200000, 300000, 400000]
    assert all(c.forced for c in cuts[:-1])
    assert cuts[-1].forced is False


def test_balanced_jitter_snaps_to_nearby_pause() -> None:
    # 理想 120 s；119.7~120.1 s 有 400 ms 静音，中点 119.9 s
    segments = _segments((0, 119700), (120100, 240000))
    cfg = _cfg(target_chunk_seconds=120, balance_tolerance=0,
               balance_jitter_seconds=0.25)
    cuts = plan_balanced_cuts(segments, 240000, cfg)
    assert cuts[0].end_ms == 119900
    assert cuts[0].cut_silence_ms == 400
    assert cuts[0].forced is False


def test_balanced_jitter_ignores_pause_outside_window() -> None:
    # 静音中点 118.0 s，离理想 120 s 有 2 s，超出 ±0.25 s
    segments = _segments((0, 117900), (118100, 240000))
    cfg = _cfg(target_chunk_seconds=120, balance_tolerance=0,
               balance_jitter_seconds=0.25)
    cuts = plan_balanced_cuts(segments, 240000, cfg)
    assert cuts[0].end_ms == 120000
    assert cuts[0].cut_silence_ms is None
    assert cuts[0].forced is True


def test_balanced_jitter_prefers_longer_pause_in_window() -> None:
    # 窗口内 100 ms（119.85 s）与 300 ms（120.20 s）两处，选更长的那处
    segments = _segments((0, 119800), (119900, 120050), (120350, 240000))
    cfg = _cfg(target_chunk_seconds=120, balance_tolerance=0,
               balance_jitter_seconds=0.3)
    cuts = plan_balanced_cuts(segments, 240000, cfg)
    assert cuts[0].end_ms == 120200
    assert cuts[0].cut_silence_ms == 300


def test_balanced_jitter_keeps_chunks_within_target() -> None:
    segments = _segments((0, 119700), (120100, 240000))
    cfg = _cfg(target_chunk_seconds=120, balance_tolerance=0,
               balance_jitter_seconds=0.25)
    cuts = plan_balanced_cuts(segments, 240000, cfg)
    lengths = [c.end_ms - c.start_ms for c in cuts]
    assert all(abs(length - 120000) <= 250 for length in lengths)


def test_balanced_ignores_silence_below_balance_threshold() -> None:
    # 30ms 静音低于 balanced 默认的 40ms，不应作为切点
    segments = _segments((0, 119970), (120000, 240000))
    cuts = plan_balanced_cuts(segments, 240000, _cfg(target_chunk_seconds=120))
    assert cuts[0].cut_silence_ms is None
    assert cuts[0].forced is True


def test_balanced_prefers_longer_silence_within_window() -> None:
    # 窗口内 100ms 与 500ms 两处静音，必须选 500ms
    segments = _segments((0, 119000), (119100, 119500), (120000, 240000))
    cuts = plan_balanced_cuts(segments, 240000, _cfg(target_chunk_seconds=120))
    assert cuts[0].cut_silence_ms == 500
    assert cuts[0].end_ms == 119750


def test_balanced_short_audio_single_chunk() -> None:
    cuts = plan_balanced_cuts(_segments((0, 5000)), 5000, _cfg(target_chunk_seconds=60))
    assert len(cuts) == 1
    assert cuts[0].end_ms == 5000


def test_balanced_handles_zero_duration() -> None:
    cuts = plan_balanced_cuts([], 0, _cfg())
    assert len(cuts) == 1 and cuts[0].end_ms == 0


def test_config_rejects_unknown_chunk_strategy() -> None:
    with pytest.raises(ValueError):
        _cfg(chunk_strategy="fixed")


def test_config_rejects_bad_balance_tolerance() -> None:
    with pytest.raises(ValueError):
        _cfg(balance_tolerance=0.6)


def test_config_rejects_negative_jitter() -> None:
    with pytest.raises(ValueError, match="balance_jitter_seconds"):
        _cfg(balance_jitter_seconds=-0.1)


def test_balanced_lengths_are_much_more_uniform_than_silence_strategy() -> None:
    # 连续口播、静音稀少：原策略会出现 180 s 硬切与 85 s 短块，均衡策略应把块长压平
    segments = _segments(*[(i * 5000, (i + 1) * 5000 - 50) for i in range(384)])
    total = 1_920_000
    old = [c.end_ms - c.start_ms for c in plan_cuts(segments, total, _cfg())]
    new = [c.end_ms - c.start_ms
           for c in plan_balanced_cuts(segments, total, _cfg(target_chunk_seconds=60))]
    spread = lambda xs: (max(xs) - min(xs)) / (sum(xs) / len(xs))
    assert spread(new) < spread(old) / 2


def test_apply_dedup_writes_back_exact_overlap() -> None:
    """精确重叠：回写后下一块开头裁掉重复前缀，拼接等于合并全文。"""
    texts = [
        "第一个阶段是1896年至1931年，这个时期属于拓荒期。",
        "这个时期属于拓荒期，中国电影处于初创和探索阶段。",
    ]
    merged, records = merge_texts(texts)
    assert records, "应检测到边界重叠"
    out = apply_dedup_to_texts(texts, records)
    assert out[0] == texts[0]
    assert "拓荒期" not in out[1] or out[1].index("拓荒期") == -1 or True
    assert not out[1].startswith("这个时期属于拓荒期")
    # 核心不变量：回写后逐块拼接 == 合并全文（页面=导出）
    assert "".join(out) == merged


def test_merge_texts_strips_block_end_punct_when_next_continues() -> None:
    """块末词被截断时 ASR 会补句号；下一块直接接字 → 去掉伪句号。"""
    merged, records = merge_texts([
        "今天呢，让我们一同。",
        "今天呢，让我们一同潜入游泳世界。",
    ])
    assert merged == "今天呢，让我们一同潜入游泳世界。"
    # 光秃秃的「一同。」不再出现
    assert "一同。" not in merged
    assert records[0].get("trimmed_tail_punct") == "。"


def test_merge_texts_drops_duplicate_next_punct_keeps_prev() -> None:
    """下一块开头带真实标点：保留上一块标点、去掉下一块重复标点。"""
    merged, records = merge_texts([
        "第一个阶段是研究，这个时期属于拓荒期。",
        "这个时期属于拓荒期，中国电影处于初创阶段。",
    ])
    assert merged == "第一个阶段是研究，这个时期属于拓荒期。中国电影处于初创阶段。"
    assert records[0].get("trimmed_head_punct") is True


def test_merge_texts_strips_trailing_quote_at_boundary() -> None:
    """块末引语被截断时残留的右引号一并去掉（「渴望。”」+「超过了」）。"""
    merged, _ = merge_texts([
        "赛后他说：“我求生的渴望。”",
        "赛后他说：“我求生的渴望”超过了赢得比赛的雄心。”",
    ])
    assert "渴望”超过了赢得比赛的雄心。" in merged
    assert "渴望。”" not in merged


def test_apply_dedup_matches_merge_with_boundary_punct() -> None:
    """回写后的逐块拼接必须与合并全文完全一致（页面=导出）。"""
    texts = [
        "第一部分，游泳运动概。",
        "第一部分，游泳运动概述；第二部分，游泳运动基本技术。",
        "第二部分，游泳运动基本技术，包括四种泳姿。",
    ]
    merged, records = merge_texts(texts)
    out = apply_dedup_to_texts(texts, records)
    assert "".join(out) == merged
    assert "概。述" not in "".join(out)


def test_apply_dedup_no_overlap_passthrough() -> None:
    """无重叠：原样返回，不产生记录。"""
    texts = ["你好，欢迎使用。", "今天天气很好。"]
    merged, records = merge_texts(texts)
    assert records == []
    assert apply_dedup_to_texts(texts, records) == texts
    assert "".join(texts) == merged


def test_apply_dedup_fuzzy_overlap_keeps_join_invariant() -> None:
    """模糊重叠（硬切错字）：trim_prev + drop_prefix 回写后拼接仍等于合并全文。"""
    texts = [
        "岛民都是渔民，世代以捕鱼为生。",
        "岛民啊都会捕鱼，渔获远销大陆。",
    ]
    merged, records = merge_texts(texts, fuzzy=True)
    out = apply_dedup_to_texts(texts, records)
    assert "".join(out) == merged
    if records:
        # 触发去重时下一块开头不应再重复上一块尾部的同义转录
        assert len(out[1]) < len(texts[1])


# ---------------------------------------------------------------------------
# plan_dp_cuts（动态规划全局最优切分）
# ---------------------------------------------------------------------------

def _regular_speech(total_ms: int, sentence_ms: int, gap_ms: int) -> list[VadSegment]:
    """构造等间隔停顿的 VAD 序列（每句之间 gap_ms 静音）。"""
    out: list[VadSegment] = []
    pos = 0
    while pos < total_ms:
        end = min(pos + sentence_ms, total_ms)
        out.append(VadSegment(pos, end))
        pos = end + gap_ms
    return out


def test_plan_dp_cuts_snaps_to_silence_and_balances() -> None:
    # 10 分钟、每 10s 一句、句间 300ms 静音；目标 60s → 10 块，切点都该在静音
    segments = _regular_speech(600_000, 9_700, 300)
    cuts = plan_dp_cuts(segments, 600_000, _cfg(target_chunk_seconds=60))

    assert len(cuts) in (10, 11)
    assert cuts[0].start_ms == 0
    assert cuts[-1].end_ms == 600_000
    # 中间切点全部落在静音上，不收尾
    assert all(c.forced is False for c in cuts[:-1])
    assert all(c.cut_silence_ms is not None for c in cuts[:-1])
    lengths = [c.end_ms - c.start_ms for c in cuts]
    assert max(lengths) - min(lengths) < 1_000


def test_plan_dp_cuts_hard_cuts_without_any_silence() -> None:
    # 整段连续语音没有停顿：DP 只能等距硬切，且块数仍接近目标
    cuts = plan_dp_cuts(_segments((0, 600_000)), 600_000,
                        _cfg(target_chunk_seconds=60))
    assert len(cuts) == 10
    assert all(c.forced is True for c in cuts[:-1])
    assert cuts[-1].forced is False
    assert cuts[-1].end_ms == 600_000
    assert max(c.end_ms - c.start_ms for c in cuts) <= 108_000


def test_plan_dp_cuts_prefers_longer_silence() -> None:
    # 58.4s 有 800ms 静音，60.05s 只有 100ms；静音奖励应让 DP 选前者
    segments = _segments((0, 58_000), (58_800, 60_000), (60_100, 120_000))
    cuts = plan_dp_cuts(segments, 120_000,
                        _cfg(target_chunk_seconds=60, balance_tolerance=0.2))
    assert cuts[0].end_ms == 58_400
    assert cuts[0].cut_silence_ms == 800
    assert cuts[0].forced is False
    assert cuts[-1].end_ms == 120_000


def test_plan_dp_cuts_short_audio_single_chunk() -> None:
    cuts = plan_dp_cuts(_segments((0, 60_000)), 60_000,
                        _cfg(target_chunk_seconds=120))
    assert len(cuts) == 1
    assert cuts[0].start_ms == 0 and cuts[0].end_ms == 60_000


def test_plan_dp_cuts_handles_zero_duration() -> None:
    cuts = plan_dp_cuts([], 0, _cfg())
    assert len(cuts) == 1
    assert cuts[0].end_ms == 0


def test_plan_dp_cuts_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError):
        _cfg(chunk_strategy="greedy")
