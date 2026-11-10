"""置信度模块（P1/P2）的单元测试。

覆盖：token→字符映射、文本侧特征、采样一致性、全局预算标亮、
段级风险排序与强制对齐特征提取。全部使用桩对象，不加载模型。
"""

from __future__ import annotations

import pytest

from local_asr.asr.confidence import (
    CROSS_BONUS,
    DEFAULT_MAX_RATIO,
    CharSignal,
    SegmentSignals,
    align_char_features,
    align_char_times,
    align_ops,
    build_char_signals,
    char_feature_series,
    collect_segment_signals,
    cross_model_flags,
    finalize_segment_risk,
    merge_positions,
    normalize_numbers,
    sample_agreement,
    score_signal,
    select_low_conf,
    text_char_features,
    token_char_spans,
)


class CharTokenizer:
    """每个字符一个 token 的桩 tokenizer。"""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return "".join(chr(i) for i in ids)


class PairTokenizer:
    """每两个字符一个 token 的桩 tokenizer（模拟多字 token）。"""

    _BASE = 100000

    def encode(self, text, add_special_tokens=False):
        out = []
        for i in range(0, len(text), 2):
            code = 0
            for ch in text[i:i + 2]:
                code = (code << 16) | ord(ch)
            out.append(self._BASE + code)
        return out

    def decode(self, ids, clean_up_tokenization_spaces=False):
        parts = []
        for code in ids:
            code -= self._BASE
            chars = []
            while code:
                chars.append(chr(code % 65536))
                code //= 65536
            parts.append("".join(reversed(chars)))
        return "".join(parts)


def _feats(entropy: float, **kwargs) -> dict:
    base = {"entropy": entropy, "neg_logprob": entropy * 0.5,
            "neg_margin": entropy * 0.3}
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# merge_positions / align_ops
# ---------------------------------------------------------------------------

def test_merge_positions_merges_adjacent():
    assert merge_positions([1, 2, 5, 6, 7, 10]) == [[1, 3], [5, 8], [10, 11]]


def test_merge_positions_empty():
    assert merge_positions([]) == []


def test_align_ops_marks_substitution_and_indel():
    ops = align_ops("abc", "axc")
    assert ("S", 1, 1) in ops
    # 删除 + 插入各自成对出现
    ops2 = align_ops("abc", "abdc")
    kinds = [op for op, _, _ in ops2]
    assert kinds.count("C") == 3
    assert "I" in kinds


# ---------------------------------------------------------------------------
# token → 字符映射
# ---------------------------------------------------------------------------

def test_token_char_spans_maps_single_char_tokens():
    tokenizer = CharTokenizer()
    text = "你好，世界"
    feats = [_feats(0.1 * i) for i in range(len(text))]
    spans = token_char_spans(feats, text, tokenizer)
    assert len(spans) == len(text)
    assert spans[0][0] == 0 and spans[0][1] == 1
    assert spans[3][0] == 3 and spans[3][1] == 4


class OverlongTokenizer:
    """逐 token 解码比原文更长的桩（复现 BPE 解码越界场景）。"""

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return "ab"  # 每个 token 都解出 2 个字符


def test_token_char_spans_clamps_overlong_decode():
    text = "你好世界"
    feats = [_feats(0.5) for _ in range(len(text))]
    spans = token_char_spans(feats, text, OverlongTokenizer())
    assert spans
    assert all(0 <= s < e <= len(text) for s, e, _ in spans)
    # 不抛异常且不产生越界空字符
    signals = build_char_signals(feats, text, OverlongTokenizer())
    assert len(signals) == len(text)


def test_token_char_spans_returns_empty_without_tokenizer():
    assert token_char_spans([_feats(1.0)], "你好", None) == []
    assert token_char_spans([], "你好", CharTokenizer()) == []
    assert token_char_spans([_feats(1.0)], "   ", CharTokenizer()) == []


def test_char_feature_series_skips_punctuation():
    tokenizer = CharTokenizer()
    text = "你，好"
    feats = [_feats(0.2), _feats(0.5), _feats(0.9)]
    series = char_feature_series(feats, text, tokenizer, "entropy")
    assert series == [(0, 0.2), (2, 0.9)]


def test_build_char_signals_marks_multi_char_token():
    tokenizer = PairTokenizer()
    text = "你好世界"
    feats = [_feats(0.3), _feats(0.8)]
    signals = build_char_signals(feats, text, tokenizer)
    assert all(s is not None for s in signals)
    assert signals[0].single_token is False
    assert signals[0].token_len == 2
    assert signals[2].entropy == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 文本侧特征
# ---------------------------------------------------------------------------

def test_text_char_features_detects_repeat_and_punct_run():
    text = "这是重复重复。！？结尾"
    feats = text_char_features(text)
    # 「重复重复」
    assert feats[2]["repeat"] or feats[3]["repeat"] or feats[4]["repeat"]
    # 「。！？」
    assert feats[text.index("！")]["punct_run"] is True
    # 段尾 10% 标记
    assert feats[-1]["tail"] is True


def test_text_char_features_detects_latin_run():
    feats = text_char_features("请拨打 1234 联系")
    i = "请拨打 1234 联系".index("1")
    assert feats[i]["latin_run"] is True
    assert feats[i + 1]["latin_run"] is True


# ---------------------------------------------------------------------------
# 采样一致性
# ---------------------------------------------------------------------------

def test_sample_agreement_full_and_partial():
    text = "你好世界"
    # 两次采样：一次完全一致，一次只对上后半段（前三字缺失）
    agree = sample_agreement(["你好世界", "好世界"], text)
    assert agree[0] == pytest.approx(0.5)
    assert agree[1] == pytest.approx(1.0)
    assert agree[3] == pytest.approx(1.0)


def test_sample_agreement_none_for_punctuation_and_empty():
    agree = sample_agreement(["你好"], "你，好")
    assert agree[1] is None
    assert agree[0] == pytest.approx(1.0)
    assert sample_agreement([], "你好") == [None, None]


# ---------------------------------------------------------------------------
# 打分
# ---------------------------------------------------------------------------

def test_score_signal_modes():
    sig = CharSignal(index=0, entropy=0.5, sample_agree=0.0,
                     align_abs_z=2.0, align_short=True)
    assert score_signal(sig, "balanced") == pytest.approx(0.5)
    assert score_signal(sig, "quality") == pytest.approx(0.5 + 0.8)
    assert score_signal(sig, "recall") == pytest.approx(0.5 + 0.35 * 2 + 0.3)


def test_score_signal_cross_mismatch_bonus_all_modes():
    """第二模型不一致是与档位无关的独立加分。"""
    sig = CharSignal(index=0, entropy=0.1, cross_mismatch=True)
    for mode in ("balanced", "quality", "recall"):
        assert score_signal(sig, mode) == pytest.approx(0.1 + CROSS_BONUS)


# ---------------------------------------------------------------------------
# 第二模型交叉验证
# ---------------------------------------------------------------------------

def test_cross_model_flags_detects_substitution():
    flags = cross_model_flags("我们讨论游泳", "我们讨论蝶泳")
    assert len(flags) == 6
    assert flags[4] and sum(flags) == 1


def test_cross_model_flags_ignores_punctuation():
    assert not any(cross_model_flags("我们，讨论游泳。", "我们讨论游泳"))


def test_cross_model_flags_ignores_filler():
    assert not any(cross_model_flags("嗯我们开始", "呃我们开始"))


def test_cross_model_flags_unifies_chinese_numbers():
    """中文数字与阿拉伯数字写法不同不算不一致。"""
    assert not any(cross_model_flags("二零零八年举办", "2008年举办"))


def test_cross_model_flags_reports_extra_hyp_chars():
    """hyp 多出的字符（重复/幻觉）也标为可疑。"""
    flags = cross_model_flags("我们开始开始说话", "我们开始说话")
    assert sum(flags) == 2
    assert flags[2] and flags[3]


def test_normalize_numbers_returns_mapping():
    out, mapping = normalize_numbers("二零零八年")
    assert out == "2008年"
    assert mapping == [0, 0, 0, 0, 4]
    # 单字数字不转换（避免「一项」被改写）
    out2, mapping2 = normalize_numbers("一项目")
    assert out2 == "一项目" and mapping2 == [0, 1, 2]


def test_cross_model_flags_positions_after_number_compression():
    """数字串压缩（一百二十三→123）后，不一致仍落在原数字串上，不前移错位。"""
    hyp = "我们有一百二十三个苹果"
    other = "我们有一百二十四个苹果"  # 仅「三/四」不同
    flags = cross_model_flags(hyp, other)
    assert len(flags) == len(hyp)
    # 数字串内部无法定位具体位，整串标记（一百二十三）
    assert flags[hyp.index("三")] is True
    assert flags[3:8] == [True] * 5
    assert not flags[0] and not flags[1] and not flags[2]
    assert not flags[hyp.index("个")] and not flags[hyp.index("苹")]


# ---------------------------------------------------------------------------
# 全局预算与门控
# ---------------------------------------------------------------------------

def _seg(seq: int, text: str, entropies: list[float],
         token_len: int = 1) -> SegmentSignals:
    chars = [CharSignal(index=i, entropy=e, single_token=token_len == 1,
                        token_len=token_len) for i, e in enumerate(entropies)]
    seg = SegmentSignals(seq=seq, text=text, chars=chars)
    singles = [c.entropy for c in chars if c.single_token]
    seg.max_single_entropy = max(singles or [0.0])
    seg.mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    return seg


def test_select_low_conf_respects_global_budget():
    # 两段各 50 字，总信息字符 100，比例 4% → 预算 4
    segs = [_seg(0, "甲" * 50, [1.0 - i * 0.01 for i in range(50)]),
            _seg(1, "乙" * 50, [0.9 - i * 0.01 for i in range(50)])]
    # 显式用 4% 预算验证比例逻辑（默认值已改为 1%，与预算语义无关）
    marked = select_low_conf(segs, max_ratio=0.04)
    assert marked == 4
    assert sum(e - s for s, e in segs[0].spans) + \
        sum(e - s for s, e in segs[1].spans) == 4
    # 每段内位置应为连续区间
    assert all(isinstance(sp, list) and len(sp) == 2 for sp in segs[0].spans)


def test_select_low_conf_gates_flat_segment():
    segs = [_seg(0, "甲" * 20, [0.1] * 20)]
    # 段内单字最高熵 0.1 < 门限 0.7 → 整段不标
    assert select_low_conf(segs, max_ratio=0.5) == 0
    assert segs[0].spans == []


def test_select_low_conf_ignores_multi_char_tokens():
    seg = _seg(0, "甲" * 20, [2.0] * 20, token_len=3)
    seg.max_single_entropy = 2.0  # 手工开门，验证候选过滤
    assert select_low_conf([seg], max_ratio=0.5) == 0


def test_select_low_conf_score_floor():
    seg = _seg(0, "甲" * 20, [0.85] + [0.05] * 19)
    marked = select_low_conf([seg], max_ratio=0.5, score_floor=0.1)
    assert marked == 1
    assert seg.spans == [[0, 1]]


def test_select_low_conf_budget_at_least_one():
    seg = _seg(0, "甲" * 5, [1.0] * 5)
    # ceil(5 * 0.01) = 1，预算至少 1
    assert select_low_conf([seg], max_ratio=0.01) == 1


def test_select_low_conf_cross_mismatch_bypasses_gate():
    """交叉验证是独立第二档：段门控关闭、熵很低也照样标。"""
    seg = _seg(0, "甲乙丙丁", [0.05] * 4)  # 单字最高熵 0.05 < 0.7，门控关
    seg.chars[2].cross_mismatch = True
    assert select_low_conf([seg], max_ratio=0.01, score_floor=0.8) == 1
    assert seg.spans == [[2, 3]]


def test_select_low_conf_cross_mismatch_respects_cap():
    """不一致字符过多时受 CROSS_MAX_RATIO 安全上限约束。"""
    seg = _seg(0, "甲" * 100, [0.1] * 100)
    for ch in seg.chars:
        ch.cross_mismatch = True
    # 常规预算因门控关闭为 0；交叉档 ceil(100 * 0.05) = 5
    assert select_low_conf([seg], max_ratio=0.01) == 5
    assert seg.spans == [[0, 5]]


def test_select_low_conf_cross_mismatch_does_not_eat_budget():
    """交叉字符走第二档，不占常规预算（否则会挤掉纯 entropy 候选）。"""
    seg = _seg(0, "甲" * 50, [1.0] * 50)
    seg.chars[0].cross_mismatch = True
    # 常规预算 1 个 → 从非 cross 候选中选（index 1）；第二档再加 index 0
    assert select_low_conf([seg], max_ratio=0.01) == 2
    assert seg.spans == [[0, 2]]


# ---------------------------------------------------------------------------
# 段级风险
# ---------------------------------------------------------------------------

def test_finalize_segment_risk_flags():
    seg = _seg(0, "你好世界" * 10, [0.9] * 40)
    seg.spans = [[0, 2]]
    seg.tail_max_entropy = 0.9
    seg.text_anomaly_count = 2
    seg.align_anomaly_count = 0
    finalize_segment_risk(seg)
    assert seg.risk_score > 0
    assert "low_conf_review" in seg.risk_flags
    assert "text_anomaly" in seg.risk_flags
    assert "boundary_uncertain" in seg.risk_flags


def test_finalize_segment_risk_clean_segment():
    seg = _seg(0, "你好世界", [0.1] * 4)
    seg.tail_max_entropy = 0.1
    finalize_segment_risk(seg)
    assert seg.risk_flags == []


# ---------------------------------------------------------------------------
# 强制对齐
# ---------------------------------------------------------------------------

class _Item:
    def __init__(self, text, start, end):
        self.text, self.start_time, self.end_time = text, start, end


class _Res:
    def __init__(self, items):
        self.items = items


class _Aligner:
    def __init__(self, items=None, fail=False):
        self.items, self.fail = items or [], fail

    def align(self, path, text, language):
        if self.fail:
            raise RuntimeError("align failed")
        return [_Res(self.items)]


def test_align_char_features_maps_durations():
    aligner = _Aligner([_Item("你", 0.0, 0.2), _Item("好", 0.2, 0.25),
                        _Item("世", 0.25, 1.6), _Item("界", 1.6, 1.8)])
    feats = align_char_features(aligner, "a.wav", "你好，世界")
    assert feats[0] and feats[0]["align_dur"] == pytest.approx(0.2)
    assert feats[2] is None  # 标点不参与对齐
    assert feats[3]["align_long"] is True
    assert feats[1]["align_short"] is True


def test_align_char_features_returns_none_on_failure():
    feats = align_char_features(_Aligner(fail=True), "a.wav", "你好")
    assert feats == [None, None]


def test_align_char_times_maps_text_positions_and_skips_punct():
    """时间戳按字母/数字位置映射，标点占位为 None（前端据此跳过）。"""
    aligner = _Aligner([_Item("你", 0.1, 0.3), _Item("好", 0.3, 0.5),
                        _Item("世", 0.5, 0.8), _Item("界", 0.8, 1.0)])
    times = align_char_times(aligner, "a.wav", "你好，世界")
    assert times[0] == (0.1, 0.3)
    assert times[1] == (0.3, 0.5)
    assert times[2] is None  # 逗号不参与对齐
    assert times[3] == (0.5, 0.8)
    assert times[4] == (0.8, 1.0)


def test_align_char_times_drops_out_of_window_items():
    """V3.0 §5.5：超出音频时长（+0.5s 容差）的时间戳必须丢弃。"""
    aligner = _Aligner([_Item("你", 0.0, 0.4), _Item("好", 9.0, 9.6),
                        _Item("世", 1.0, 1.2)])
    times = align_char_times(aligner, "a.wav", "你好世", max_seconds=3.0)
    assert times[0] == (0.0, 0.4)
    assert times[1] is None   # 9.6s 远超 3.0+0.5
    assert times[2] == (1.0, 1.2)  # 越界项之后仍按原位置继续映射


def test_align_char_times_drops_negative_and_reversed():
    aligner = _Aligner([_Item("你", -0.1, 0.2), _Item("好", 0.6, 0.4),
                        _Item("世", 0.5, 0.7)])
    times = align_char_times(aligner, "a.wav", "你好世")
    assert times[0] is None
    assert times[1] is None
    assert times[2] == (0.5, 0.7)


def test_collect_segment_signals_merges_all_sources():
    tokenizer = CharTokenizer()
    text = "你好世界"
    feats = [_feats(0.1), _feats(0.9), _feats(0.4), _feats(0.2)]
    align = [None, {"align_dur_z": -1.5, "align_short": True,
                    "align_long": False}, None, None]
    seg = collect_segment_signals(
        feats, text, tokenizer, seq=3,
        sample_agreement=[1.0, 0.0, 0.5, 1.0], align_features=align)
    assert seg.seq == 3
    assert seg.max_single_entropy == pytest.approx(0.9)
    assert seg.chars[1].sample_agree == pytest.approx(0.0)
    assert seg.chars[1].align_short is True
    assert seg.chars[1].align_abs_z == pytest.approx(1.5)
    assert seg.align_anomaly_count == 1