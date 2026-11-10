"""生成完整性判定的单元测试。"""

from __future__ import annotations

import pytest

from local_asr.asr.generation import (
    COMPLETED,
    DEGRADED,
    TRUNCATED,
    UNEXPECTED_STOP,
    GenerationResult,
    aggregate_status,
    classify_generation,
    eos_ids_from,
    flag_empty_transcript,
    segment_payload,
)


def test_status_completed_when_eos_hit() -> None:
    r = GenerationResult(text="你好", generated_tokens=120, hit_eos=True, hit_limit=False)
    assert r.status == COMPLETED


def test_status_truncated_when_limit_hit_without_eos() -> None:
    r = GenerationResult(text="你好", generated_tokens=4096, hit_eos=False, hit_limit=True)
    assert r.status == TRUNCATED


def test_status_unexpected_stop_when_neither() -> None:
    r = GenerationResult(text="你好", generated_tokens=10, hit_eos=False, hit_limit=False)
    assert r.status == UNEXPECTED_STOP


def test_eos_wins_when_both_conditions_hold() -> None:
    # 恰好在上限处收尾：有 EOS 就视为正常结束，不能误报截断。
    r = GenerationResult(text="x", generated_tokens=512, hit_eos=True, hit_limit=True)
    assert r.status == COMPLETED


class _Cfg:
    def __init__(self, eos: object) -> None:
        self.eos_token_id = eos


def test_eos_ids_from_none() -> None:
    assert eos_ids_from(_Cfg(None)) == set()


def test_eos_ids_from_int() -> None:
    assert eos_ids_from(_Cfg(151645)) == {151645}


def test_eos_ids_from_list() -> None:
    assert eos_ids_from(_Cfg([151643, 151645])) == {151643, 151645}


def test_classify_detects_eos() -> None:
    hit_eos, hit_limit = classify_generation(
        last_token_id=151645, generated_tokens=100, max_new_tokens=4096, eos_ids={151643, 151645})
    assert hit_eos is True
    assert hit_limit is False


def test_classify_detects_limit() -> None:
    hit_eos, hit_limit = classify_generation(
        last_token_id=88, generated_tokens=4096, max_new_tokens=4096, eos_ids={151645})
    assert hit_eos is False
    assert hit_limit is True


def test_classify_zero_tokens_is_not_success() -> None:
    hit_eos, hit_limit = classify_generation(
        last_token_id=-1, generated_tokens=0, max_new_tokens=4096, eos_ids={151645})
    assert hit_eos is False
    assert hit_limit is False


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([], UNEXPECTED_STOP),
        ([COMPLETED], COMPLETED),
        ([COMPLETED, TRUNCATED], TRUNCATED),
        ([COMPLETED, UNEXPECTED_STOP], UNEXPECTED_STOP),
        ([TRUNCATED, UNEXPECTED_STOP], TRUNCATED),
        ([COMPLETED, DEGRADED], DEGRADED),
        ([DEGRADED, TRUNCATED], DEGRADED),
    ],
)
def test_aggregate_status(statuses: list[str], expected: str) -> None:
    results = [
        GenerationResult(text="", generated_tokens=1,
                         hit_eos=s == COMPLETED,
                         hit_limit=s == TRUNCATED,
                         degraded=s == DEGRADED)
        for s in statuses
    ]
    assert aggregate_status(results) == expected


def test_degraded_overrides_other_status() -> None:
    """降级块即使有 EOS 也按 degraded 上报，提醒调用方该块文本不可信。"""
    r = GenerationResult(text="", generated_tokens=0, hit_eos=True, hit_limit=False,
                         degraded=True)
    assert r.status == DEGRADED


def test_segment_payload_fields() -> None:
    r = GenerationResult(text="你好世界", generated_tokens=42, hit_eos=True, hit_limit=False)
    payload = segment_payload(r, index=2, start_s=96.06, end_s=197.955, cut_silence_ms=1080)
    assert payload == {
        "index": 2,
        "start_s": 96.06,
        "end_s": 197.96,
        "cut_silence_ms": 1080,
        "generated_tokens": 42,
        "hit_eos": True,
        "hit_limit": False,
        "status": COMPLETED,
        "degraded": False,
        "chars": 4,
    }


def test_empty_transcript_on_long_audio_is_degraded() -> None:
    """32 分钟音频只生成 1 个 EOS token、文本为空时不能算 completed。"""
    r = GenerationResult(text="", generated_tokens=1, hit_eos=True, hit_limit=False)
    flagged = flag_empty_transcript(r, 1920.34)
    assert flagged.status == DEGRADED
    assert flagged.text == ""


def test_empty_transcript_on_short_audio_stays_completed() -> None:
    r = GenerationResult(text="  ", generated_tokens=1, hit_eos=True, hit_limit=False)
    assert flag_empty_transcript(r, 5.0).status == COMPLETED


def test_non_empty_transcript_is_unchanged() -> None:
    r = GenerationResult(text="你好", generated_tokens=3, hit_eos=True, hit_limit=False)
    assert flag_empty_transcript(r, 1920.34) is r
