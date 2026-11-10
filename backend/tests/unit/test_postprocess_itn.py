"""数字规范化（ITN）规则测试。"""

from __future__ import annotations

import pytest

from local_asr.postprocess.itn import (
    cn_to_int, normalize_numbers, normalize_numbers_segmented,
)


def test_cn_to_int_basic():
    assert cn_to_int("三") == 3
    assert cn_to_int("十五") == 15
    assert cn_to_int("三十六") == 36
    assert cn_to_int("一百零八") == 108
    assert cn_to_int("三千") == 3000
    # 口语省略读法（"三千六"=3600）本身有歧义，这里不做补全，只按字面解析
    assert cn_to_int("三千六") == 3006
    assert cn_to_int("abc") is None


@pytest.mark.parametrize("src, expected", [
    ("跌了百分之三十六点七幺", "跌了36.71%"),
    ("从四三八零点跌到幺三五八点", "从4380点跌到1358点"),
    ("市盈率十五倍", "市盈率15倍"),
    ("亏损负三点五万元", "亏损-3.5万元"),
])
def test_standard_conversions(src, expected):
    assert normalize_numbers(src).text == expected


@pytest.mark.parametrize("src", [
    "一模一样",
    "稍等一下",
    "最近两三个月",
    "第一季度",
    "十几个人",
])
def test_natural_expressions_preserved(src):
    assert normalize_numbers(src).text == src


def test_percent_hundred_is_converted():
    # "百分之百"是明确的百分比读法，转成 100% 属于正常转换
    assert normalize_numbers("百分之百").text == "100%"


def test_wan_yi_scale_preserved():
    """wetext 会给出 30000亿，这里恢复为 3万亿。"""
    result = normalize_numbers("成交额三万亿")
    assert "3万亿" in result.text
    assert "30000亿" not in result.text


def test_percent_point_not_percent():
    result = normalize_numbers("下降了三个百分点")
    assert "3个百分点" in result.text
    assert "3%" not in result.text


def test_phone_digits_no_space():
    result = normalize_numbers("电话零零三幺三五八")
    assert "0031 358" not in result.text
    assert "0031358" in result.text


def test_ambiguous_year_kept_and_flagged():
    result = normalize_numbers("二三年七月份")
    assert "二三年" in result.text
    assert any(n.kind == "ambiguous_year" for n in result.notes)


def test_full_year_converted():
    result = normalize_numbers("二零二三年七月份")
    assert "2023" in result.text


def test_duplicated_year_run_not_converted_to_bigint():
    """识别重复的年份串（二零二零二一）不能被 wetext 拼成 202021 或 "~" 区间。"""
    result = normalize_numbers("我们在二零二一到二零二零二一到二零二四年这几年")
    assert "202021" not in result.text
    assert "~" not in result.text
    assert "2021到二零二零二一到2024年" in result.text
    assert any(n.kind == "ambiguous_year" and n.snippet == "二零二零二一"
               for n in result.notes)


def test_standard_year_range_still_converted():
    result = normalize_numbers("二零二一年到二零二四年")
    assert result.text == "2021年到2024年"
    assert result.notes == ()


def test_repeated_numeral_run_not_split_into_multiple_numbers():
    """识别重复的「十五十十五」不能被 wetext 切成 "15 10 15"。"""
    raw = "六万块钱一门课卖了几百万，然后就讲了十五十十五节课"
    result = normalize_numbers(raw)
    assert "15 10 15" not in result.text
    assert "十五十十五" in result.text
    assert any(n.kind == "ambiguous_number" for n in result.notes)


def test_standard_compound_number_with_one_ten_converted():
    result = normalize_numbers("一共一千一百一十块钱")
    assert "1110" in result.text


def test_arabic_numbers_untouched():
    src = "指数从4380点跌到1358点，跌幅39.78%"
    assert normalize_numbers(src).text == src


def test_glossary_only_on_explicit_hit():
    result = normalize_numbers("科创五零和创业板", glossary={"科创五零": "科创50"})
    assert "科创50" in result.text
    assert "创业板" in result.text


def test_glossary_not_applied_without_match():
    # 词典只对明确命中的写法生效，不做近似匹配
    result = normalize_numbers("科创办和创业板", glossary={"科创五零": "科创50"})
    assert "科创50" not in result.text


def test_mappings_returned():
    result = normalize_numbers("跌了百分之三十六点七幺")
    assert result.mappings
    assert any("36.71" in m["output_text"] for m in result.mappings)


def test_digit_sequence_integer_conversion():
    assert cn_to_int("四三八零") == 4380


def test_longest_glossary_hit_and_repeated_occurrences():
    raw = "科创五零和科创五零"
    result = normalize_numbers(raw, {"科创": "不应命中", "科创五零": "科创50"})
    assert result.text == "科创50和科创50"


def test_existing_placeholder_like_text_is_preserved():
    raw = "原样0，科创五零"
    result = normalize_numbers(raw, {"科创五零": "科创50"})
    assert result.text == "原样0，科创50"


def test_existing_separated_numbers_not_concatenated():
    raw = "电话0031 358，分机另记"
    assert normalize_numbers(raw).text == raw


def test_itn_edits_replay_and_include_rule_sources():
    from local_asr.postprocess.verify import replay_edits
    raw = "成交额三万亿，科创五零，下降三个百分点，电话零零三幺三五八"
    result = normalize_numbers(raw, {"科创五零": "科创50"})
    assert replay_edits(raw, list(result.edits)) == result.text
    rules = " ".join(e.rule for e in result.edits)
    assert all(word in rules for word in ("wetext", "glossary", "wan_yi", "percent_point", "phone"))
    for edit in result.edits:
        assert raw[edit.start:edit.end] == edit.before
        assert result.text[edit.output_start:edit.output_end] == edit.after


def test_segmented_matches_whole_and_preserves_untouched_sentences():
    """全句切块并行，结果必须与整篇逐字符一致。"""
    raw = ("今天天气很好。成交额三万亿。我们去游泳吧。市盈率十五倍。"
           "电话零零三幺三五八。就这样。")
    whole = normalize_numbers(raw)
    segmented = normalize_numbers_segmented(
        raw, workers=1, min_chars=1, chunk_chars=20)
    assert segmented.text == whole.text
    from local_asr.postprocess.verify import replay_edits
    assert replay_edits(raw, list(segmented.edits)) == segmented.text
    # 无数字的句子不会被改写
    assert "今天天气很好。" in segmented.text


def test_segmented_handles_single_numeral_before_noun():
    """预筛漏检回归：wetext 会把「西湖十景」转成「西湖10景」。

    旧实现按「数字候选」跳过未命中句，导致 60min 文本出现 1 处漏改；
    现在全句切块，不会再漏。
    """
    raw = ("西湖拥有独特的自然景观，比如西湖十景和历史遗迹。"
           "这里还有十分丰富的文化资源。")
    whole = normalize_numbers(raw)
    segmented = normalize_numbers_segmented(
        raw, workers=1, min_chars=1, chunk_chars=10)
    assert "西湖10景" in whole.text
    assert segmented.text == whole.text


def test_segmented_notes_and_edits_use_full_text_coordinates():
    raw = ("前面一句。今天讲的是九九年的事情。后面还有市盈率十五倍的内容。"
           "再补一句完全无关的话。")
    whole = normalize_numbers(raw)
    segmented = normalize_numbers_segmented(
        raw, workers=1, min_chars=1, chunk_chars=20)
    assert segmented.text == whole.text
    for edit in segmented.edits:
        assert raw[edit.start:edit.end] == edit.before
        assert segmented.text[edit.output_start:edit.output_end] == edit.after
    for note in segmented.notes:
        if note.start is not None and note.end is not None:
            assert raw[note.start:note.end] == note.snippet


def test_damaged_placeholders_roll_back(monkeypatch):
    from types import SimpleNamespace
    class BrokenNormalizer:
        def normalize_with_mapping(self, text):
            return SimpleNamespace(output_text=text.replace("", ""), mappings=())
    monkeypatch.setattr("local_asr.postprocess.itn._get_normalizer", lambda: BrokenNormalizer())
    raw = "一模一样，科创五零"
    result = normalize_numbers(raw, {"科创五零": "科创50"})
    assert result.text == raw
    assert result.degraded
    assert not result.edits