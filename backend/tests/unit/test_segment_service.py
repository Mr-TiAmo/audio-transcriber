"""片段构建服务单测：stats.json -> DB 片段文本的去重应用（回归）。

背景：去重记录（merge_texts 的 dedup）只能在入库时应用一次。
上一版曾在 ASR 脚本里对 segments 回写一次、服务层又按记录裁一次，
导致真实链路中 chunk 边界被裁两遍（句首误删，如「设计这么重」→「么」）。
本文件锁定"入库文本 = 各 chunk 原文按去重记录裁一次"。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.services.segment_service import (
    BuiltSegment,
    build_segments_from_stats,
    repack_display_segments,
)
from local_asr.asr.segmentation import merge_texts


def _write_stats(tmp_path: Path, chunks: list[dict], dedup: list[dict]) -> Path:
    stats = {
        "segments": chunks,
        "segmentation": {"dedup": dedup},
    }
    path = tmp_path / "raw.stats.json"
    path.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
    return path


def _chunk(start_s: float, end_s: float, text: str) -> dict:
    return {"start_s": start_s, "end_s": end_s, "text": text}


def test_dedup_applied_exactly_once(tmp_path: Path) -> None:
    """stats 中存的是 chunk 原文：入库只裁一次，且拼接等于合并全文。"""
    texts = ["我们今天讨论旅游线路设计", "旅游线路设计需要了解目标市场"]
    merged, dedup = merge_texts(texts)
    assert dedup and dedup[0]["boundary"] == 0
    assert dedup[0]["dropped_chars"] == 6  # "旅游线路设计"

    path = _write_stats(
        tmp_path,
        [_chunk(0, 10, texts[0]), _chunk(10, 20, texts[1])],
        dedup,
    )
    segs = build_segments_from_stats(path, repack=False)

    assert segs[0].text == texts[0]
    # 只裁一次：结果是 "需要了解目标市场"；若再裁一遍会变成 "解目标市场"
    assert segs[1].text == "需要了解目标市场"
    assert "".join(s.text for s in segs) == merged


def test_dedup_trim_prev_suffix(tmp_path: Path) -> None:
    """trimmed_prev_chars 裁前一段尾部（与 dropped_chars 可同时出现）。"""
    texts = ["我们今天讨论旅游线路设计的", "旅游线路设计需要了解目标市场"]
    path = _write_stats(
        tmp_path,
        [_chunk(0, 10, texts[0]), _chunk(10, 20, texts[1])],
        [{
            "boundary": 0,
            "dropped_chars": 6,
            "dropped": "旅游线路设计",
            "trimmed_prev_chars": 1,
            "trimmed_prev": "的",
            "method": "fuzzy",
            "edit_distance": 1,
        }],
    )
    segs = build_segments_from_stats(path, repack=False)

    assert segs[0].text == "我们今天讨论旅游线路设计"
    assert segs[1].text == "需要了解目标市场"


def test_low_conf_spans_shift_after_dedup_and_strip(tmp_path: Path) -> None:
    """去重裁首 + strip 空白都要平移 low_conf_spans，避免标亮错位。"""
    texts = ["  前一段结尾", "  重复开头需要了解目标市场"]
    # 第 1 段开头裁掉「重复开头」（4 字）；第 0 段只受 strip 影响
    path = _write_stats(
        tmp_path,
        [
            {**_chunk(0, 10, texts[0]),
             # strip 掉开头 2 个空格后，原文 [2,4]/[4,5] → [0,2]/[2,3]
             "low_conf_spans": [[2, 4], [4, 5]]},
            {**_chunk(10, 20, texts[1]),
             # 原文下标：0/1 空格，2..5「重复开头」，6 起「需要了解…」
             "low_conf_spans": [[2, 6], [8, 9]]},
        ],
        [{
            "boundary": 0,
            "dropped_chars": 4,
            "dropped": "重复开头",
            "method": "exact",
            "edit_distance": 0,
        }],
    )
    segs = build_segments_from_stats(path, repack=False)

    assert segs[0].text == "前一段结尾"
    assert segs[0].low_conf_spans == [[0, 2], [2, 3]]
    assert segs[1].text == "需要了解目标市场"
    # 裁掉 4 字（重复开头）+ 其前 2 个空格 = 平移 6
    # [2,6] 完全落在被裁区 → 丢弃；[8,9] → [2,3]
    assert segs[1].low_conf_spans == [[2, 3]]


def test_low_conf_spans_clamped_to_trimmed_text(tmp_path: Path) -> None:
    """尾裁后超出新文本末尾的 span 被截断，不越界。"""
    texts = ["前一段的结尾尾巴", "后一段内容"]
    path = _write_stats(
        tmp_path,
        [
            {**_chunk(0, 10, texts[0]), "low_conf_spans": [[4, 9]]},
            _chunk(10, 20, texts[1]),
        ],
        [{
            "boundary": 0,
            "dropped_chars": 0,
            "trimmed_prev_chars": 2,
            "trimmed_prev": "尾巴",
            "method": "exact",
            "edit_distance": 0,
        }],
    )
    segs = build_segments_from_stats(path, repack=False)

    assert segs[0].text == "前一段的结尾"
    # [4,9] 超出新长度 6 → 截断为 [4,6]
    assert segs[0].low_conf_spans == [[4, 6]]


def test_save_and_load_review_fields_roundtrip(tmp_path: Path) -> None:
    """risk_flags / low_conf_spans / review_priority 入库后可完整读回。"""
    from app.core.database import Database, new_id, now_iso
    from app.services.segment_service import (
        BuiltSegment,
        load_segments,
        save_segments,
    )

    db = Database(tmp_path / "app.db")
    project_id, run_id = new_id("proj"), new_id("run")
    db.execute(
        """INSERT INTO projects (id, original_name, audio_path, pcm_path, file_size,
           duration_ms, status, created_at, updated_at)
           VALUES (?, 'a.wav', 'a.wav', 'b.wav', 1, 1000, 'READY', ?, ?)""",
        (project_id, now_iso(), now_iso()),
    )
    db.execute(
        "INSERT INTO transcription_runs (id, project_id, status) VALUES (?, ?, 'COMPLETED')",
        (run_id, project_id),
    )
    save_segments(db, run_id, [
        BuiltSegment(sequence_no=0, start_ms=0, end_ms=1000, text="你好",
                     risk_flags=["low_conf_review"], low_conf_spans=[[0, 1]],
                     review_priority=2.5,
                     aligned_tokens=[[0, 380], [380, 760]]),
    ])

    seg = load_segments(db, run_id)[0]
    assert seg["riskFlags"] == ["low_conf_review"]
    assert seg["lowConfSpans"] == [[0, 1]]
    assert seg["reviewPriority"] == 2.5
    # aligned_tokens 以 JSON 存库，读回要还原成数组（含 null 占位）
    assert seg["alignedTokens"] == [[0, 380], [380, 760]]


def test_aligned_tokens_sliced_with_dedup_and_strip(tmp_path: Path) -> None:
    """字符时间戳与文本同步做去重裁剪 + strip，并与入库文本等长。"""
    texts = ["  前一段结尾", "  重复开头需要了解目标市场"]
    times0 = [[i * 10, i * 10 + 9] for i in range(len(texts[0]))]
    times1 = [[1000 + i, 1009 + i] for i in range(len(texts[1]))]
    path = _write_stats(
        tmp_path,
        [
            {**_chunk(0, 10, texts[0]), "char_times": times0},
            {**_chunk(10, 20, texts[1]), "char_times": times1},
        ],
        [{
            "boundary": 0,
            "dropped_chars": 4,
            "dropped": "重复开头",
            "method": "exact",
            "edit_distance": 0,
        }],
    )
    segs = build_segments_from_stats(path, repack=False)

    assert segs[0].text == "前一段结尾"
    # strip 掉 2 个空格后取 times0[2:7]
    assert segs[0].aligned_tokens == times0[2:]
    assert segs[1].text == "需要了解目标市场"
    # 去重裁掉前 4 字 + 前 2 个空格 = 起点 6
    assert segs[1].aligned_tokens == times1[6:]
    assert len(segs[1].aligned_tokens) == len(segs[1].text)


def test_aligned_tokens_length_mismatch_degrades_to_none(tmp_path: Path) -> None:
    """时间戳长度与文本不符（过期/损坏）时整段丢弃，前端回退插值。"""
    path = _write_stats(
        tmp_path,
        [{"start_s": 0, "end_s": 10, "text": "一二三四五",
          "char_times": [[0, 100], [100, 200]]}],
        [],
    )
    assert build_segments_from_stats(path, repack=False)[0].aligned_tokens is None


def test_boundary_punct_cleanup_keeps_index_alignment(tmp_path: Path) -> None:
    """块末伪标点被删后，aligned_tokens 与 low_conf_spans 仍与文本对齐。"""
    texts = ["第一部分，游泳运动概。", "第一部分，游泳运动概述；第二部分。"]
    times0 = [[i * 10, i * 10 + 9] for i in range(len(texts[0]))]
    times1 = [[1000 + i * 10, 1009 + i * 10] for i in range(len(texts[1]))]
    merged, records = merge_texts(texts)
    assert merged == "第一部分，游泳运动概述；第二部分。"

    path = _write_stats(
        tmp_path,
        [
            {**_chunk(0, 10, texts[0]), "char_times": times0,
             "low_conf_spans": [[8, 9]]},
            {**_chunk(10, 20, texts[1]), "char_times": times1,
             "low_conf_spans": [[10, 11]]},
        ],
        records,
    )
    segs = build_segments_from_stats(path, repack=False)

    # 段 0 尾部伪句号删除，段 1 从「述」开始
    assert segs[0].text == "第一部分，游泳运动概"
    assert segs[1].text == "述；第二部分。"
    assert segs[0].aligned_tokens == times0[:-1]
    assert segs[1].aligned_tokens == times1[10:]
    assert segs[0].low_conf_spans == [[8, 9]]
    assert segs[1].low_conf_spans == [[0, 1]]  # 原下标 10 → 平移 10


def test_aligned_tokens_all_null_degrades_to_none(tmp_path: Path) -> None:
    """全部字符都没对齐上时视为无数据，不写入 DB。"""
    path = _write_stats(
        tmp_path,
        [{"start_s": 0, "end_s": 10, "text": "一二三",
          "char_times": [None, None, None]}],
        [],
    )
    assert build_segments_from_stats(path, repack=False)[0].aligned_tokens is None


def test_empty_chunks_skipped_and_sequence_continuous(tmp_path: Path) -> None:
    """空文本 chunk 不建段；结果按 start 排序且序号连续。"""
    path = _write_stats(
        tmp_path,
        [
            _chunk(10, 20, "第二段内容"),
            _chunk(0, 10, "第一段内容"),
            _chunk(20, 30, "   "),
        ],
        [],
    )
    segs = build_segments_from_stats(path, repack=False)

    assert [s.text for s in segs] == ["第一段内容", "第二段内容"]
    assert [s.sequence_no for s in segs] == [0, 1]
    assert [s.start_ms for s in segs] == [0, 10000]

# ---------------------------------------------------------------------------
# 展示段重切分：识别块边界落在句中时，入库按句子边界重新打包
# ---------------------------------------------------------------------------
def _built(text: str, start_ms: int, end_ms: int, times=None, spans=None) -> BuiltSegment:
    return BuiltSegment(
        sequence_no=0, start_ms=start_ms, end_ms=end_ms, text=text,
        risk_flags=[], low_conf_spans=spans, aligned_tokens=times,
    )


def test_repack_keeps_cross_chunk_sentence_together() -> None:
    """块边界处句子未完（句号后无停顿）时，两块合成一个展示段。"""
    t1 = "第一部分，游泳运动概。"
    t2 = "述；第二部分。"
    # 每字 100ms；标点不参与对齐（None）。块 1 末尾「。」到块 2 首字间隔 20ms。
    times1 = [[i * 100, i * 100 + 90] if t1[i] != "。" else None for i in range(len(t1))]
    times2 = [[i * 100 + 2000, i * 100 + 2090] if t2[i] != "。" else None
              for i in range(len(t2))]
    out = repack_display_segments(
        [_built(t1, 0, 1000, times1), _built(t2, 1000, 2000, times2)],
        target_chars=1000, target_seconds=999,
    )

    assert len(out) == 1
    assert out[0].text == t1 + t2
    assert out[0].aligned_tokens is not None
    assert len(out[0].aligned_tokens) == len(out[0].text)


def test_repack_breaks_at_sentence_end_with_real_pause() -> None:
    """句末标点后有明显停顿，且已达目标长度 -> 在完整句子处断开。"""
    text = "第一句话。第二句话。"
    times = [
        [0, 90], [100, 190], [200, 290], [300, 390], None,   # 第一句话。
        [900, 990], [1000, 1090], [1100, 1190], [1200, 1290], [1300, 1390], None,
    ]
    out = repack_display_segments(
        [_built(text, 0, 2000, times)], target_chars=4, target_seconds=999,
    )

    assert len(out) == 2
    assert out[0].text == "第一句话。"
    assert out[1].text == "第二句话。"


def test_repack_slices_times_and_low_conf_spans() -> None:
    """合并后 aligned_tokens 等长、低置信区间平移到新段坐标。"""
    t1 = "第一段内容。"
    t2 = "第二段内容。"
    times1 = [[i * 100, i * 100 + 90] if t1[i] != "。" else None for i in range(len(t1))]
    times2 = [[i * 100 + 2000, i * 100 + 2090] if t2[i] != "。" else None
              for i in range(len(t2))]
    seg1 = _built(t1, 0, 1000, times1, spans=[[2, 4]])
    seg2 = _built(t2, 1000, 2000, times2, spans=[[1, 3]])

    out = repack_display_segments([seg1, seg2], target_chars=1000, target_seconds=999)

    assert len(out) == 1
    # 段 2 的 [1,3] 平移 len(t1) 后与段 1 的 [2,4] 分离
    assert out[0].low_conf_spans == [[2, 4], [len(t1) + 1, len(t1) + 3]]
    assert len(out[0].aligned_tokens) == len(out[0].text)


def test_repack_without_alignment_falls_back_to_punctuation() -> None:
    """没有字符时间时按标点切，不因缺失对齐数据合并成一段。"""
    text1 = "第一句话。第二句话。"
    text2 = "第三句话。"
    out = repack_display_segments(
        [_built(text1, 0, 1000), _built(text2, 1000, 1500)],
        target_chars=4, target_seconds=999,
    )

    assert [s.text for s in out] == ["第一句话。", "第二句话。", "第三句话。"]
    assert all(s.aligned_tokens is None for s in out)


def test_repack_preserves_full_text_and_sequence() -> None:
    """重切分不丢字、不改顺序，序号连续。"""
    texts = ["大家好，今天呢，", "我们讨论游泳。它是一项运动。", "谢谢大家。"]
    segs = [_built(t, i * 1000, (i + 1) * 1000) for i, t in enumerate(texts)]

    out = repack_display_segments(segs, target_chars=12, target_seconds=999)

    assert "".join(s.text for s in out) == "".join(texts)
    assert [s.sequence_no for s in out] == list(range(len(out)))
    assert all(s.end_ms > s.start_ms for s in out)
