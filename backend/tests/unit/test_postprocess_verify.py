"""修改校验测试。"""

from __future__ import annotations

import pytest

from local_asr.postprocess.verify import Edit, review_corrections


def test_identical_text_returns_no_edits():
    text = "今天天气不错"
    out, edits = review_corrections(text, text)
    assert out == text
    assert edits == []


def test_safe_correction_accepted():
    raw = "这些智谱大模型的公司"
    corrected = "这些智谱大模型的公司"
    out, edits = review_corrections(raw, corrected)
    assert out == corrected
    assert edits == []


def test_chinese_typo_correction_accepted():
    raw = "这个本105他需求"
    corrected = "这个本他需求"
    out, edits = review_corrections(raw, corrected)
    # 删除内容含数字，应转复核而不是直接改
    assert out == raw
    assert all(e.decision == "review" for e in edits)


def test_number_change_flagged():
    raw = "跌到二七七二"
    corrected = "跌到2772"
    out, edits = review_corrections(raw, corrected)
    assert out == raw
    assert edits and edits[0].decision == "review"
    assert "数字" in edits[0].reason


def test_negation_change_flagged():
    raw = "这个不需要担心"
    corrected = "这个需要担心"
    out, edits = review_corrections(raw, corrected)
    assert out == raw
    assert edits[0].decision == "review"
    assert "否定" in edits[0].reason


def test_whole_segment_ratio_guard():
    raw = "短期震荡不改长期趋势"
    corrected = "市场会继续上下波动但大方向仍然向上"
    out, edits = review_corrections(raw, corrected, max_edit_ratio=0.10)
    assert out == raw
    assert all(e.decision == "review" for e in edits)
    assert "编辑比例" in edits[0].reason


def test_partial_accept_keeps_raw_for_risky_parts():
    raw = "市盈率十五倍是合理的"
    corrected = "市盈率15倍是合理的"
    out, edits = review_corrections(raw, corrected, max_edit_ratio=0.5)
    # 数字变化不放行
    assert out == raw
    assert any(e.decision == "review" for e in edits)


def test_edit_dataclass_fields():
    e = Edit(op="replace", start=0, end=1, before="a", after="b",
             decision="accepted", reason="")
    assert e.op == "replace"


@pytest.mark.parametrize("before,after", [
    ("-3.5万元", "+3.5万元"), ("3.5万元", "-3.5万元"),
    ("−3.5万元", "3.5万元"), ("3.5万元", "35万元"),
    ("0031358", "031358"), ("百分之三", "三个百分点"),
    ("负三万元", "三万元"), ("3公斤", "3公里"),
    ("¥3", "$3"), ("€3", "3"), ("人民币3万元", "美元3万元"),
])
def test_entire_numeric_expression_is_protected(before, after):
    prefix = "这是需要核对的重要报告，其中记录的数据为"
    raw = prefix + before + "，请保持记录的原始含义。" * 5
    candidate = raw.replace(before, after)
    adopted, edits = review_corrections(raw, candidate)
    assert adopted == raw
    assert edits and all(e.decision == "review" for e in edits)


@pytest.fixture
def numeric_checker():
    import importlib.util
    import sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("check_numeric_fidelity", Path(__file__).parents[2] / "scripts/check_numeric_fidelity.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("ref,hyp", [
    ("收益-3.5万元", "收益+3.5万元"),
    ("收益负三点五万元", "收益三点五万元"),
    ("电话0031358", "电话31358"),
    ("甲为3元，乙为5元", "甲为5元，乙为3元"),
    ("下降三个百分点", "下降3%"),
    ("二三年七月", "2023年7月"),
])
def test_numeric_fidelity_rejects_meaning_changes(numeric_checker, ref, hyp):
    assert numeric_checker.compare(ref, hyp)["rate"] < 1


@pytest.mark.parametrize("ref,hyp", [
    ("点位四三八零点", "点位4380点"),
    ("收益负三点五万元", "收益-3.5万元"),
    ("电话零零三幺三五八", "电话0031358"),
    ("金额三万亿元", "金额30000亿元"),
    ("下降百分之三点五", "下降3.5%"),
    ("下降三个百分点", "下降3个百分点"),
    ("日期二零二三年七月五日", "日期2023-07-05"),
])
def test_numeric_fidelity_accepts_equivalent_formats(numeric_checker, ref, hyp):
    assert numeric_checker.compare(ref, hyp)["rate"] == 1


def test_numeric_counts_duplicates(numeric_checker):
    result = numeric_checker.compare("甲3元，乙3元", "甲3元，乙遗漏")
    assert result["ref_values"] == 2
    assert result["rate"] == .5
    assert len(result["missing"]) == 1


def test_percent_mismatch_is_not_triggered_by_unrelated_extra_percent(numeric_checker):
    assert not numeric_checker.compare("甲3元", "甲3元，乙3%")["percent_point_mix"]


def test_raw_cer_does_not_strip_signs_or_whitespace():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("eval_cer", Path(__file__).parents[2] / "scripts/eval_cer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = " -3.50万元，啊！\n"
    assert module.normalize(raw, raw=True) == raw


@pytest.fixture
def comparison():
    import importlib.util
    import sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("compare_postprocess", Path(__file__).parents[2] / "scripts/compare_postprocess.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def variant(tmp_path):
    from local_asr.postprocess.pipeline import PostprocessPipeline
    return PostprocessPipeline().run("甲三元，乙五元。\r\n", tmp_path, "sample")


def test_comparison_checks_completed_artifact_hashes(comparison, variant, tmp_path):
    from pathlib import Path
    assert comparison.load_variant(tmp_path / "sample")["raw_text"] == variant.raw_text
    Path(variant.paths["corrected"]).write_text("篡改")
    with pytest.raises(ValueError, match="指纹"):
        comparison.load_variant(tmp_path / "sample")


@pytest.mark.parametrize("state", [None, "running", "failed"])
def test_comparison_skips_unconfirmed_artifacts(comparison, variant, tmp_path, state):
    import json
    from pathlib import Path
    variant.stats["status"] = state
    Path(variant.paths["stats"]).write_text(json.dumps(variant.stats))
    assert comparison.load_variant(tmp_path / "sample") is None


def test_comparison_reports_four_variants(comparison, variant, tmp_path, monkeypatch, capsys):
    import sys
    reference = tmp_path / "ref.txt"
    reference.write_text("甲3元，乙5元。\r\n")
    monkeypatch.setattr(sys, "argv", ["compare", "--ref", str(reference), "--raw", variant.paths["raw"],
                                    "--candidate", str(tmp_path / "sample")])
    assert comparison.main() == 0
    output = capsys.readouterr().out
    assert all(label in output for label in ("raw（ASR 原始）", "仅 ITN（同一 raw）", "sample 仅纠错", "sample 最终版"))


@pytest.mark.parametrize("field", ["raw", "glossary_sha256", "rules_sha256", "wetext_version"])
def test_comparison_rejects_mixed_baselines(comparison, variant, tmp_path, monkeypatch, field):
    import json
    import sys
    from pathlib import Path
    raw = tmp_path / "source.txt"
    raw.write_text("不同原文" if field == "raw" else variant.raw_text, newline="")
    if field in ("glossary_sha256", "rules_sha256"):
        variant.stats[field] = "different"
    if field == "wetext_version":
        variant.stats["runtime_versions"]["wetext"] = "different"
    Path(variant.paths["stats"]).write_text(json.dumps(variant.stats))
    monkeypatch.setattr(sys, "argv", ["compare", "--ref", str(raw), "--raw", str(raw),
                                    "--candidate", str(tmp_path / "sample")])
    with pytest.raises(ValueError, match="不同|不一致"):
        comparison.main()


def test_comparison_handles_empty_reference_and_position_metric(comparison):
    import math
    assert math.isnan(comparison.evaluate_text("empty", "", "") ["literal_cer"])
    row = comparison.evaluate_text("swap", "甲3元，乙5元", "甲5元，乙3元")
    assert row["numeric"] == 0 and row["numeric_bag"] == 100