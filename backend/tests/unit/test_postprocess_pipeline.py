"""两阶段后处理编排测试。"""

from __future__ import annotations

import json

import pytest

from local_asr.asr.generation import GenerationResult
from local_asr.postprocess import (
    MLXCorrector,
    PassthroughCorrector,
    PostprocessPipeline,
    build_corrector,
)
from local_asr.postprocess.corrector import BaseCorrector, CorrectionResult


class FakeCorrector(BaseCorrector):
    name = "fake"

    def __init__(self, output: str) -> None:
        self.output = output

    def correct(self, text: str) -> CorrectionResult:
        return CorrectionResult(text=self.output, model=self.name, segments=1)


def test_passthrough_only_normalizes_numbers(tmp_path):
    raw = "成交额三万亿，跌了百分之三十六点七幺"
    pipe = PostprocessPipeline(corrector=PassthroughCorrector())
    result = pipe.run(raw, output_dir=tmp_path, stem="demo")

    assert "3万亿" in result.final_text
    assert "36.71%" in result.final_text
    assert result.edits == []
    assert result.stats["corrector"] == "none"
    for key in ("raw", "corrected", "final", "edits", "stats"):
        assert key in result.paths


def test_output_files_written(tmp_path):
    pipe = PostprocessPipeline()
    result = pipe.run("市盈率十五倍", output_dir=tmp_path, stem="demo")

    assert (tmp_path / "demo_raw.txt").read_text(encoding="utf-8") == "市盈率十五倍"
    assert (tmp_path / "demo_corrected.txt").read_text(encoding="utf-8") == "市盈率十五倍"
    assert "15倍" in (tmp_path / "demo_final.txt").read_text(encoding="utf-8")

    stats = json.loads((tmp_path / "demo_postprocess.json").read_text(encoding="utf-8"))
    assert stats["raw_chars"] == len("市盈率十五倍")
    assert stats["elapsed_scope"] == "postprocess_before_stats_publish"
    assert 0 <= stats["derived_publish_seconds"] <= stats["elapsed_seconds"]
    assert stats["corrector_successful_segments"] + stats["corrector_failed_segments"] == stats["corrector_segments"]


def test_corrector_changes_reviewed_before_itn(tmp_path):
    raw = "这个本105他需求"
    pipe = PostprocessPipeline(corrector=FakeCorrector("这个本他需求"))
    result = pipe.run(raw, output_dir=tmp_path, stem="demo")

    # 删除含数字，需复核，最终保留原文
    assert result.corrected_text == raw
    assert result.review_count == 1
    assert result.accepted_count == 0


def test_safe_correction_applied(tmp_path):
    raw = "这些智普公司"
    # 短文本的 1 字修改已超过默认 10% 阈值，这里放宽阈值以验证"安全改动可应用"
    pipe = PostprocessPipeline(corrector=FakeCorrector("这些智谱公司"), max_edit_ratio=0.5)
    result = pipe.run(raw, output_dir=tmp_path, stem="demo")

    assert result.corrected_text == "这些智谱公司"
    assert result.accepted_count == 1


def test_itn_notes_written_to_edits_jsonl(tmp_path):
    pipe = PostprocessPipeline()
    pipe.run("二三年七月份", output_dir=tmp_path, stem="demo")

    lines = (tmp_path / "demo_edits.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert any(r.get("kind") == "ambiguous_year" for r in records)


def test_build_corrector_none():
    assert isinstance(build_corrector(None), PassthroughCorrector)
    assert isinstance(build_corrector("none"), PassthroughCorrector)


def test_build_corrector_mlx_requires_path():
    try:
        build_corrector("mlx")
    except ValueError as exc:
        assert "corrector-model" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应当抛出 ValueError")


def test_mlx_missing_model_path_raises(tmp_path):
    corrector = MLXCorrector(tmp_path / "not-exist")
    try:
        corrector.correct("测试文本")
    except RuntimeError as exc:
        assert "不存在" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应当抛出 RuntimeError")


def test_segments_respect_target_chars():
    corrector = MLXCorrector("unused", target_chars=10)
    text = "第一句话。第二句话。第三句话。"
    segments = corrector._segments(text)
    assert len(segments) >= 2
    assert "".join(target for _, target in segments) == text

def test_itn_can_be_disabled_for_correction_only_variant():
    raw = "成交额三万亿"
    pipe = PostprocessPipeline(corrector=PassthroughCorrector(), enable_itn=False)
    result = pipe.run(raw, write_files=False)

    assert result.final_text == raw
    assert result.itn_notes == []
    assert result.stats["itn_enabled"] is False


def test_raw_preserves_whitespace_and_line_endings(tmp_path):
    raw = "  原始文本\r\n第二行\n"
    result = PostprocessPipeline(enable_itn=False).run(raw, tmp_path, "exact")
    assert result.raw_text == raw
    assert (tmp_path / "exact_raw.txt").read_bytes() == raw.encode()


def test_raw_checkpoint_survives_corrector_failure(tmp_path):
    class BrokenCorrector(BaseCorrector):
        def correct(self, text):
            raise RuntimeError("模型加载失败")
    with pytest.raises(RuntimeError):
        PostprocessPipeline(BrokenCorrector()).run("原文", tmp_path, "failed")
    assert (tmp_path / "failed_raw.txt").read_text() == "原文"
    assert not (tmp_path / "failed_postprocess.json").exists()


def test_existing_outputs_rejected_before_generation(tmp_path):
    class MustNotRun(BaseCorrector):
        def correct(self, text):
            pytest.fail("应在推理前拒绝已有产物")
    (tmp_path / "demo_final.txt").write_text("旧结果")
    with pytest.raises(FileExistsError):
        PostprocessPipeline(MustNotRun()).run("原文", tmp_path, "demo")
    assert (tmp_path / "demo_final.txt").read_text() == "旧结果"


def test_different_raw_cannot_overwrite_checkpoint(tmp_path):
    (tmp_path / "demo_raw.txt").write_text("既有原文")
    with pytest.raises(FileExistsError):
        PostprocessPipeline(enable_itn=False).run("新原文", tmp_path, "demo")
    assert (tmp_path / "demo_raw.txt").read_text() == "既有原文"


def test_identical_raw_checkpoint_can_be_reused(tmp_path):
    path = tmp_path / "demo_raw.txt"
    path.write_bytes(b"raw\r\n")
    before = path.stat().st_mtime_ns
    PostprocessPipeline(enable_itn=False).run("raw\r\n", tmp_path, "demo")
    assert path.stat().st_mtime_ns == before


def test_itn_runtime_failure_degrades_without_losing_correction(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("规则引擎失败")
    monkeypatch.setattr(
        "local_asr.postprocess.pipeline.normalize_numbers_segmented", fail)
    result = PostprocessPipeline(FakeCorrector("这些智谱公司"), max_edit_ratio=.5).run(
        "这些智普公司", tmp_path, "demo")
    assert result.final_text == result.corrected_text == "这些智谱公司"
    assert result.stats["status"] == "degraded"
    assert result.stats["degradations"]


def test_long_unpunctuated_input_has_bounded_chunks():
    text = "没有标点Hello\n" * 100
    chunks = MLXCorrector("unused", target_chars=300)._segments(text)
    assert all(len(target) <= 300 for _, target in chunks)
    assert "".join(target for _, target in chunks) == text


def test_segment_ratio_not_diluted_by_unchanged_text():
    from local_asr.postprocess.corrector import CorrectionSegment
    first, second = "甲" * 300, "乙" * 690 + "这是智普公司的描述。"
    class Segmented(BaseCorrector):
        def correct(self, text):
            a = "丙" * 40 + first[40:]
            b = second.replace("智普", "智谱")
            return CorrectionResult(a + b, "fake", segments=2, segment_results=[
                CorrectionSegment(0, len(first), first, a),
                CorrectionSegment(len(first), len(text), second, b),
            ])
    result = PostprocessPipeline(Segmented(), enable_itn=False).run(first + second, write_files=False)
    assert result.corrected_text == first + second.replace("智普", "智谱")
    assert result.review_count and result.accepted_count
    assert result.stats["review_fallback_segments"] == 1
    from local_asr.postprocess.verify import replay_edits
    assert replay_edits(result.raw_text, result.edits) == result.corrected_text


@pytest.mark.parametrize("output,expected", [
    ("<think>分析</think>正文", "正文"),
    ("分析</think>正文", "正文"),
    ("正文", "正文"),
    ("<think>未结束的分析", ""),
    ("<\\think>分析<\\think>正文", ""),
    ("正文\nuser\n指令", ""),
    ("正文<|im_end|>", ""),
    ("待校对文本：复述", ""),
    ("<think>一</think><think>二</think>正文", ""),
])
def test_output_parser_is_fail_closed(output, expected):
    assert MLXCorrector._parse_output(output) == expected


def _end_think() -> str:
    """拆开书写，避免编辑器/传输层把它当成标签。"""
    return "</" + "think" + ">"


def test_parser_strips_leading_role_line_from_body():
    """4-bit 模型偶尔复刻 chat template，把 user 行写进正文。"""
    body = "分析" + _end_think() + "\n\nuser\n纠正句"
    assert MLXCorrector._parse_output(body) == "纠正句"
    assert MLXCorrector._parse_output(body + "\n") == "纠正句"


def test_parser_treats_explicit_no_error_as_unchanged():
    """模型判「无误」且无正文时保持原文，而不是判为格式失败。"""
    out = "分析\n错误类型：无误  \n修改原因：原句表达准确。" + _end_think() + "\n\n"
    assert MLXCorrector._parse_output(out, "原文") == "原文"
    assert MLXCorrector._parse_output(out) == ""


def test_parser_discards_echoed_body_after_no_error():
    """判「无误」后复述原文是已知行为，不能把复述当纠错结果。"""
    out = "分析\n错误类型：无误\n修改原因：原句无误。" + _end_think() + "\n原文原文原文"
    assert MLXCorrector._parse_output(out, "原文") == "原文"


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "提示" + messages[-1]["content"]

    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))


def test_input_budget_counts_prompt_tokens_without_losing_characters():
    c = MLXCorrector("unused", target_chars=300, max_input_tokens=100)
    c._tokenizer = FakeTokenizer()
    raw = "中文与English组合\n" * 70
    segments = c._segments(raw)
    assert "".join(s for _, s in segments) == raw
    assert all(len(c._prompt_tokens(ctx, s)) <= 100 for ctx, s in segments)


def test_generation_passes_repetition_penalty(monkeypatch):
    """复读根因：官方 README 要求 repetition_penalty=1.1，必须透传给生成器。"""
    import sys
    from types import SimpleNamespace
    seen = {}

    def fake_processors(**kwargs):
        seen["params"] = kwargs
        return ["proc"]

    def fake_stream(*args, **kwargs):
        seen["kwargs"] = kwargs
        return iter([SimpleNamespace(text="正文", finish_reason="stop",
                                     prompt_tokens=1, generation_tokens=1, peak_memory=0)])

    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(stream_generate=fake_stream))
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils",
                        SimpleNamespace(make_logits_processors=fake_processors))
    c = MLXCorrector("unused")
    c._model, c._tokenizer = object(), FakeTokenizer()
    assert c.correct("原文").text == "正文"
    assert seen["params"] == {"repetition_penalty": 1.1, "repetition_context_size": 128}
    assert seen["kwargs"]["logits_processors"] == ["proc"]
    assert seen["kwargs"]["max_tokens"] == 512


def test_user_message_has_no_label_prefix():
    """user 消息带标签会让模型复述标签，正文因此被判为泄漏。"""
    class RecordingTokenizer(FakeTokenizer):
        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return "提示"

    c = MLXCorrector("unused")
    c._tokenizer = RecordingTokenizer()
    c._prompt_tokens("", "原文")
    assert c._tokenizer.messages[-1] == {"role": "user", "content": "原文"}


def test_repetition_penalty_below_one_is_rejected():
    with pytest.raises(ValueError):
        MLXCorrector("unused", repetition_penalty=0.9)


def test_parser_rejects_abnormally_long_body():
    """去掉 think 块后仍超长，说明残留分析块，整段回退。"""
    assert MLXCorrector._parse_output("正文" * 10, "甲") == ""


def test_generation_uses_real_stop_reason_and_token_count(monkeypatch):
    from types import SimpleNamespace
    import sys
    events = [
        SimpleNamespace(text="<think>分析</think>", finish_reason=None, prompt_tokens=40,
                        generation_tokens=10, peak_memory=.1),
        SimpleNamespace(text="正文", finish_reason="length", prompt_tokens=40,
                        generation_tokens=2048, peak_memory=.2),
    ]
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(stream_generate=lambda *a, **kw: iter(events)))
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", SimpleNamespace(
        make_logits_processors=lambda **kw: []))
    c = MLXCorrector("unused")
    c._model, c._tokenizer = object(), FakeTokenizer()
    result = c.correct("原文")
    assert result.text == "原文"
    assert result.failed_segments == 1 and result.truncated
    assert result.segment_results[0].generation_tokens == 2048
    assert result.mlx_peak_bytes == 200_000_000


def test_generation_exception_returns_original_segment(monkeypatch):
    c = MLXCorrector("unused")
    c._model = object()
    def fail(*args):
        raise RuntimeError("模拟推理失败")
    monkeypatch.setattr(c, "_generate", fail)
    result = c.correct("原文。")
    assert result.text == "原文。"
    assert result.failed_segments == 1
    assert result.segment_results[0].status == "failed"


def test_input_output_alias_and_symlink_are_rejected(tmp_path):
    raw = tmp_path / "source.txt"
    raw.write_text("原文")
    pipe = PostprocessPipeline(enable_itn=False)
    with pytest.raises(ValueError):
        pipe.run("原文", tmp_path, "source", input_path=tmp_path / "source_final.txt")
    link = tmp_path / "alias_final.txt"
    try:
        link.symlink_to(raw)
    except OSError:
        # Windows 非管理员 shell 无创建符号链接特权（WinError 1314）：
        # 前半别名拒绝断言已覆盖，符号链接部分跳过而非误报失败
        pytest.skip("当前环境无创建符号链接特权")
    with pytest.raises(ValueError):
        pipe.run("原文", tmp_path, "alias", input_path=raw)
    assert raw.read_text() == "原文"


def test_hardlink_alias_rejected(tmp_path):
    import os
    source = tmp_path / "source.txt"
    source.write_text("原文")
    os.link(source, tmp_path / "demo_final.txt")
    with pytest.raises(ValueError):
        PostprocessPipeline(enable_itn=False).run("原文", tmp_path, "demo", input_path=source)


def test_write_failure_has_no_complete_stats(tmp_path, monkeypatch):
    from local_asr.postprocess import pipeline as module
    publish = module.publish_text
    def fail(path, text, **kwargs):
        if path.name.endswith("_final.txt"):
            raise OSError("模拟磁盘错误")
        return publish(path, text, **kwargs)
    monkeypatch.setattr(module, "publish_text", fail)
    with pytest.raises(OSError):
        PostprocessPipeline(enable_itn=False).run("原文", tmp_path, "demo")
    assert (tmp_path / "demo_raw.txt").read_text() == "原文"
    assert not (tmp_path / "demo_postprocess.json").exists()


def test_publish_does_not_clobber_a_racing_writer(tmp_path, monkeypatch):
    from local_asr.postprocess import artifacts
    real_link = artifacts.os.link
    destination = tmp_path / "final.txt"
    def race(src, dst):
        destination.write_text("另一个写入者")
        real_link(src, dst)
    monkeypatch.setattr(artifacts.os, "link", race)
    with pytest.raises(FileExistsError):
        artifacts.publish_text(destination, "新结果")
    assert destination.read_text() == "另一个写入者"


def _script(name):
    import importlib.util
    import sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[2] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _asr_args(tmp_path):
    import argparse
    import wave
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 100)
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "processor_config.json"):
        (model / name).write_text("{}")
    (model / "model.safetensors").write_bytes(b"stub")
    return argparse.Namespace(
        audio=str(audio), output=str(tmp_path / "demo.txt"), raw_output=None,
        model=str(model), window=0, max_new_tokens=4096, device="cpu",
        language="Chinese", prompt=None,
        segmentation="auto", full_max_seconds=300.0, target_chunk=120.0,
        max_chunk=180.0, min_silence_ms=200, overlap=0.5, vad_model=None,
        batch_size=1, batch_auto=False, batch_tolerance=0.2, batch_max=3,
        batch_max_driver_mb=8000,
    )


def test_asr_output_is_itself_the_raw_checkpoint(tmp_path, monkeypatch):
    """ASR 只产出原始文本：--output 就是 raw，不产生第二份同一内容。"""
    script = _script("qwen3_asr_transcribe")
    args = _asr_args(tmp_path)
    monkeypatch.setattr(script, "require_module", lambda *a: None)
    monkeypatch.setattr(script, "check_device", lambda *a: None)
    monkeypatch.setattr(script, "load_model", lambda *a: (object(), object()))
    monkeypatch.setattr(script, "decode_window", lambda *a: GenerationResult("原文\r\n", 5, True, False))
    assert script.run(args) == 0
    assert (tmp_path / "demo.txt").read_bytes() == "原文\r\n".encode()
    assert not (tmp_path / "demo_raw.txt").exists()


def test_asr_separate_raw_checkpoint_is_written(tmp_path, monkeypatch):
    """显式指定 --raw-output 时，raw 与 output 是两份独立文件。"""
    script = _script("qwen3_asr_transcribe")
    args = _asr_args(tmp_path)
    args.raw_output = str(tmp_path / "checkpoint.txt")
    monkeypatch.setattr(script, "require_module", lambda *a: None)
    monkeypatch.setattr(script, "check_device", lambda *a: None)
    monkeypatch.setattr(script, "load_model", lambda *a: (object(), object()))
    monkeypatch.setattr(script, "decode_window", lambda *a: GenerationResult("原文\r\n", 5, True, False))
    assert script.run(args) == 0
    assert (tmp_path / "checkpoint.txt").read_bytes() == "原文\r\n".encode()
    assert (tmp_path / "demo.txt").read_bytes() == "原文\r\n".encode()


def test_asr_raw_survives_stats_publish_failure(tmp_path, monkeypatch):
    """统计写入失败不应回滚已发布的原始文本。"""
    script = _script("qwen3_asr_transcribe")
    args = _asr_args(tmp_path)
    monkeypatch.setattr(script, "require_module", lambda *a: None)
    monkeypatch.setattr(script, "check_device", lambda *a: None)
    monkeypatch.setattr(script, "load_model", lambda *a: (object(), object()))
    monkeypatch.setattr(script, "decode_window", lambda *a: GenerationResult("原文\r\n", 5, True, False))
    real_publish = script.publish_text
    def publish(path, text, **kwargs):
        if str(path).endswith(".stats.json"):
            raise RuntimeError("统计写入失败")
        return real_publish(path, text, **kwargs)
    monkeypatch.setattr(script, "publish_text", publish)
    with pytest.raises(RuntimeError, match="统计写入失败"):
        script.run(args)
    assert (tmp_path / "demo.txt").read_bytes() == "原文\r\n".encode()
    assert not (tmp_path / "demo.stats.json").exists()


def test_non_16k_input_preflights_converter_before_asr_load(tmp_path, monkeypatch):
    """非 16k 输入且没有 ffmpeg / PyAV 时，必须在加载模型前失败。"""
    import wave

    script = _script("qwen3_asr_transcribe")
    args = _asr_args(tmp_path)
    audio = tmp_path / "audio_8k.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(b"\0\0" * 100)
    args.audio = str(audio)
    monkeypatch.setattr(script, "require_module", lambda *a: None)
    monkeypatch.setattr(script, "can_convert_audio", lambda: False)
    monkeypatch.setattr(script, "load_model", lambda *a: pytest.fail("不应加载模型"))
    with pytest.raises(RuntimeError, match="PyAV"):
        script.run(args)


def test_asr_script_has_no_postprocess_duty():
    """识别脚本只出 raw：数字规范化与纠错一律由后处理入口承担。"""
    script = _script("qwen3_asr_transcribe")
    assert not hasattr(script, "apply_itn")
    assert not hasattr(script, "PostprocessPipeline")
    assert not hasattr(script, "build_corrector")


def test_offline_model_preflight_rejects_missing_shards(tmp_path):
    from local_asr.postprocess.corrector import validate_local_model
    args = _asr_args(tmp_path)
    from pathlib import Path
    model = Path(args.model)
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "missing.safetensors"}}))
    with pytest.raises(RuntimeError, match="分片"):
        validate_local_model(model)


@pytest.mark.parametrize("left,right,changed", [
    ("金额为-", "3万元", "+"),
    ("重量3公", "斤", "里"),
])
def test_numeric_protection_crosses_segment_boundaries(left, right, changed):
    from local_asr.postprocess.corrector import CorrectionSegment
    first, second = "相关说明" * 30 + left, right + "相关说明" * 30
    a, b = (first[:-1] + changed, second) if left.endswith("-") else (first, changed + second[1:])
    class Segmented(BaseCorrector):
        def correct(self, text):
            return CorrectionResult(a + b, "fake", segments=2, segment_results=[
                CorrectionSegment(0, len(first), first, a),
                CorrectionSegment(len(first), len(text), second, b),
            ])
    result = PostprocessPipeline(Segmented(), enable_itn=False).run(first + second, write_files=False)
    assert result.corrected_text == first + second
    assert result.review_count > 0


def test_pipeline_never_adopts_a_failed_segment_candidate():
    from local_asr.postprocess.corrector import CorrectionSegment
    raw = "这些智普公司在此提供相关服务。"
    class Failed(BaseCorrector):
        def correct(self, text):
            candidate = text.replace("智普", "智谱")
            return CorrectionResult(candidate, "fake", segments=1, segment_results=[
                CorrectionSegment(0, len(text), text, candidate, "failed", "length")])
    result = PostprocessPipeline(Failed(), enable_itn=False).run(raw, write_files=False)
    assert result.corrected_text == raw
    assert result.stats["status"] == "degraded"
    assert result.stats["corrector_failed_segments"] == 1


def test_model_manifest_identifies_files_without_claiming_weight_hash(tmp_path):
    from pathlib import Path
    from local_asr.postprocess.artifacts import model_manifest
    args = _asr_args(tmp_path)
    model = Path(args.model)
    manifest = model_manifest(model)
    assert manifest["metadata_sha256"]["config.json"]
    assert manifest["weights"][0]["name"] == "model.safetensors"
    assert manifest["weights"][0]["size_bytes"] == 4
    assert manifest["weight_identity_scope"] == "file_metadata_not_content_hash"
    (model / "model.safetensors").write_bytes(b"changed")
    assert model_manifest(model)["manifest_sha256"] != manifest["manifest_sha256"]


def test_mlx_preflight_rejects_legacy_stream_metadata(tmp_path, monkeypatch):
    from pathlib import Path
    args = _asr_args(tmp_path)
    monkeypatch.setattr("local_asr.postprocess.corrector.require_module", lambda *a: None)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.20.0")
    with pytest.raises(RuntimeError, match="0.32.2"):
        MLXCorrector(Path(args.model)).preflight()


def test_asr_decoder_preserves_raw_whitespace(monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    script = _script("qwen3_asr_transcribe")
    class Inputs(dict):
        def to(self, *args):
            return self
    class Tokens:
        shape = (1, 10)

        def __getitem__(self, key):
            return self

        def __int__(self):
            return 151645

    raw = "  原文\r\n"

    # qwen-asr 0.0.6 processor API：apply_chat_template 构建 prompt、
    # __call__ 编码、batch_decode 解码（新 API 无 apply_transcription_request）
    class FakeProcessor:
        def apply_chat_template(self, msgs, **kw):
            return "PROMPT"

        def __call__(self, text=None, audio=None, **kw):
            return Inputs(input_ids=SimpleNamespace(shape=(1, 3)))

        def batch_decode(self, ids, **kw):
            return [raw]

    model = SimpleNamespace(
        dtype=None,
        device="cpu",
        generate=lambda **kw: Tokens(),
        generation_config=SimpleNamespace(eos_token_id=[151643, 151645]),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    # 音频加载旁路：伪造 qwen_asr 模块树（避免真实导入拉起 transformers，
    # 其 find_spec("torch") 会因 fake torch 无 __spec__ 而报错）
    import types
    fake_utils = types.ModuleType("qwen_asr.inference.utils")
    fake_utils.normalize_audios = lambda paths: [object()] * len(paths)
    fake_pkg = types.ModuleType("qwen_asr")
    fake_pkg.__path__ = []
    fake_inf = types.ModuleType("qwen_asr.inference")
    fake_inf.__path__ = []
    fake_pkg.inference, fake_inf.utils = fake_inf, fake_utils
    for name, mod in (("qwen_asr", fake_pkg), ("qwen_asr.inference", fake_inf),
                      ("qwen_asr.inference.utils", fake_utils)):
        monkeypatch.setitem(sys.modules, name, mod)
    result = script.decode_window(FakeProcessor(), model, "unused", None, "cpu", 10)
    assert result.text == raw
    assert result.generated_tokens == 10
    assert result.hit_eos is True
    # 长度恰好触顶但收到 EOS：按正常结束处理，不误报截断
    assert result.status == "completed"


def test_itn_initialization_fails_before_corrector_loading(tmp_path, monkeypatch):
    def broken():
        raise RuntimeError("ITN 规则文件损坏")
    class MustNotLoad(BaseCorrector):
        def prepare(self):
            pytest.fail("应在加载纠错模型前检查 ITN 依赖及规则")
    monkeypatch.setattr("local_asr.postprocess.itn._get_normalizer", broken)
    with pytest.raises(RuntimeError, match="规则"):
        PostprocessPipeline(MustNotLoad()).run("原文", tmp_path, "demo")
    assert not (tmp_path / "demo_postprocess.json").exists()


def test_interrupted_stream_retains_observed_token_counts(monkeypatch):
    import sys
    from types import SimpleNamespace
    def interrupted(*args, **kwargs):
        yield SimpleNamespace(text="未结束分析", prompt_tokens=40, generation_tokens=10,
                              finish_reason=None, peak_memory=.1)
        raise RuntimeError("异常正文不得进入审计")
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(stream_generate=interrupted))
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", SimpleNamespace(
        make_logits_processors=lambda **kw: []))
    c = MLXCorrector("unused")
    c._model, c._tokenizer = object(), FakeTokenizer()
    result = c.correct("原文")
    assert result.text == "原文" and result.failed_segments == 1
    assert result.segment_results[0].generation_tokens == 10
    assert "异常正文" not in result.segment_results[0].reason


# 音频函数（_pcm_bytes / prepare_pcm16k 等）已下沉到 local_asr.audio，
# 对应测试迁移至 tests/unit/test_audio.py，不再经由脚本命名空间间接测试。


def _pieces(script, count: int = 2, seconds: float = 100.0):
    return [
        script.PiecePlan(i, i * seconds, (i + 1) * seconds,
                         int(i * seconds * 1000), int((i + 1) * seconds * 1000))
        for i in range(count)
    ]


def test_plan_batch_keeps_single_when_disabled(tmp_path, monkeypatch):
    script = _script("qwen3_asr_transcribe")
    args = _asr_args(tmp_path)
    monkeypatch.setattr(script, "mps_driver_mb", lambda: 4000.0)
    _, size, groups, _ = script.plan_batch(args, {2: 120.0, 0: 100.0, 1: 110.0})
    assert size == 1
    assert groups == [[0], [1], [2]]  # 按块号还原，不受输入 dict 顺序影响


def test_plan_batch_downgrades_when_driver_limit_is_low(tmp_path, monkeypatch):
    script = _script("qwen3_asr_transcribe")
    args = _asr_args(tmp_path)
    args.batch_auto = True
    args.batch_max_driver_mb = 5000  # 5500 基线 + 800 已超限
    monkeypatch.setattr(script, "mps_driver_mb", lambda: 4392.0)
    _, size, groups, _ = script.plan_batch(args, {0: 96.0, 1: 102.0, 2: 134.0})
    assert size == 1
    assert [len(g) for g in groups] == [1, 1, 1]


def test_decode_pieces_falls_back_to_single_on_batch_error(tmp_path, monkeypatch):
    """批推理失败时整组回退逐条重跑，产物仍完整。"""
    script = _script("qwen3_asr_transcribe")
    args = _asr_args(tmp_path)
    args.batch_auto = True
    monkeypatch.setattr(script, "mps_driver_mb", lambda: 4000.0)
    monkeypatch.setattr(script, "decode_group", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("MPS OOM")))
    monkeypatch.setattr(script, "decode_window",
                        lambda *a, **k: GenerationResult("单条", 5, True, False))

    segments, text, batching, _, _ = script.decode_pieces(
        object(), object(), args, "Chinese", tmp_path, script.Path(args.audio),
        _pieces(script, seconds=0.002), merge=True)

    assert len(segments) == 2
    assert text == "单条单条"
    assert batching["applied_size"] == 2
    assert batching["fallbacks"][0]["group"] == [0, 1]
    assert "MPS OOM" in batching["fallbacks"][0]["error"]


def test_decode_pieces_marks_degraded_when_retry_also_fails(tmp_path, monkeypatch):
    """批与单条都失败时只标记该块 degraded，不丢整份结果。"""
    script = _script("qwen3_asr_transcribe")
    args = _asr_args(tmp_path)
    args.batch_auto = True
    monkeypatch.setattr(script, "mps_driver_mb", lambda: 4000.0)
    monkeypatch.setattr(script, "decode_group", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("batch failed")))
    monkeypatch.setattr(script, "decode_window", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("single failed")))

    segments, text, batching, _, _ = script.decode_pieces(
        object(), object(), args, "Chinese", tmp_path, script.Path(args.audio),
        _pieces(script, count=1, seconds=0.002), merge=True)

    assert segments[0].result.status == "degraded"
    assert segments[0].result.text == ""
    assert text == ""
    assert batching["fallbacks"]
