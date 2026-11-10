# 片段构建服务（V3.0 §5.6 / §5.7：片段边界即 chunk 边界，经过去重后合并）
# 数据源：编排层产出的 stats.json（chunk 时间 + 文本 + dedup 记录）
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from local_asr.asr.segmentation import (
    apply_dedup_to_texts,
    drop_prefix_by_chars,
    drop_suffix_by_chars,
    strip_head_punct,
    strip_tail_punct,
)

from ..core.database import Database, new_id, now_iso


@dataclass
class BuiltSegment:
    sequence_no: int
    start_ms: int
    end_ms: int
    text: str
    risk_flags: list[str]
    low_conf_spans: list | None = None  # [[start, end], ...] 低置信字符区间
    review_priority: float | None = None  # 段级风险分（高=优先复核）
    # 字符级对齐时间：[[start_ms, end_ms] | None, ...]，与入库文本等长
    aligned_tokens: list | None = None


# ---------------------------------------------------------------------------
# 展示段重切分（识别块 ≠ 展示段）
# ---------------------------------------------------------------------------
# 识别块按音频时长切（60 s），边界常落在句子中间；页面直接按块渲染会把
# 同一句话拆成两行。入库前把各块去重后的文本串成字符流，按「句子边界」
# 重新打包成展示段：句子完整、编辑粒度仍是展示段，前端无需感知技术块。
DISPLAY_TARGET_SECONDS = 240.0  # 展示段目标时长
DISPLAY_TARGET_CHARS = 1200     # 展示段目标字数（与时长共同决定切点）
_END_PUNCT = frozenset("。！？；…!?;")
_SUB_PUNCT = frozenset("，、：,:")
# 句末标点后至少停顿这么久才算句子边界（过滤 ASR 对截断音频补的伪句号）
_END_MIN_GAP_MS = 150
# 没有句末标点时，逗号/顿号后至少停顿这么久才允许作为次级边界
_SUB_MIN_GAP_MS = 300


def _nearest_time(times: list, i: int, *, forward: bool) -> tuple[int, int] | None:
    """找 i 处（含）向前/向后最近的有效对齐时间。"""
    step = 1 if forward else -1
    j = i
    while 0 <= j < len(times):
        t = times[j]
        if t is not None:
            return (int(t[0]), int(t[1]))
        j += step
    return None


def _boundary_gap(times: list, i: int) -> int | None:
    """边界字符 i 之后到下一字符之前的停顿（毫秒）；缺对齐数据返回 None。"""
    prev = _nearest_time(times, i, forward=False)
    nxt = _nearest_time(times, i + 1, forward=True)
    if prev is None or nxt is None:
        return None
    return max(0, nxt[0] - prev[1])


def repack_display_segments(
    segments: list[BuiltSegment],
    *,
    target_seconds: float = DISPLAY_TARGET_SECONDS,
    target_chars: int = DISPLAY_TARGET_CHARS,
) -> list[BuiltSegment]:
    """按句子边界把识别块重新打包成展示段。

    规则：
    - 句末标点 + 字符间隔 >= 150 ms 视为句子边界；间隔未知（无对齐数据）
      时按标点切；
    - 没有句末标点时，逗号/顿号 + 间隔 >= 300 ms 可作次级边界；
    - 累积到目标字数/时长后在最近的句子边界断开，保留完整句子；
    - 文本、aligned_tokens、low_conf_spans 都是字符级切片，天然对齐。
    """
    if not segments:
        return segments

    chars: list[str] = []
    times: list = []
    src: list[int] = []
    for idx, seg in enumerate(segments):
        chars.extend(seg.text)
        at = seg.aligned_tokens
        if not isinstance(at, list) or len(at) != len(seg.text):
            at = [None] * len(seg.text)
        times.extend(at)
        src.extend([idx] * len(seg.text))
    n = len(chars)
    if n == 0:
        return segments

    # 全片码点级标亮区间（各段平移后合并）
    spans: list[tuple[int, int]] = []
    off = 0
    for seg in segments:
        for a, b in seg.low_conf_spans or []:
            spans.append((off + int(a), off + int(b)))
        off += len(seg.text)

    # 边界分两级：主边界=句末标点，次边界=逗号/顿号。达到目标后优先等主边界，
    # 只有超过 1.5 倍目标才退而求其次，避免「一句话被逗号切开」。
    boundaries: list[tuple[int, bool]] = []
    for i, ch in enumerate(chars[:-1]):
        if ch in _END_PUNCT:
            gap = _boundary_gap(times, i)
            if gap is None or gap >= _END_MIN_GAP_MS:
                boundaries.append((i, True))
        elif ch in _SUB_PUNCT:
            gap = _boundary_gap(times, i)
            if gap is not None and gap >= _SUB_MIN_GAP_MS:
                boundaries.append((i, False))
    force_chars = max(target_chars + 1, int(target_chars * 1.5))
    force_ms = int(target_seconds * 1.5 * 1000)

    def _duration(start: int, end: int) -> int | None:
        head = _nearest_time(times, start, forward=True)
        tail = _nearest_time(times, end, forward=False)
        if head is None or tail is None:
            return None
        return max(0, tail[1] - head[0])

    cuts: list[int] = []
    start = 0
    for b, primary in boundaries + [(n - 1, True)]:
        if b < start:
            continue
        length = b - start + 1
        dur = _duration(start, b)
        forced = length >= force_chars or (dur is not None and dur >= force_ms)
        reached = length >= target_chars or (dur is not None
                                             and dur >= int(target_seconds * 1000))
        if forced or (reached and primary):
            cuts.append(b)
            start = b + 1
    if not cuts or cuts[-1] != n - 1:
        cuts.append(n - 1)

    out: list[BuiltSegment] = []
    start = 0
    for cut in cuts:
        if cut < start:
            continue
        text = "".join(chars[start:cut + 1])
        if not text.strip():
            start = cut + 1
            continue
        valid = [t for t in times[start:cut + 1] if t is not None]
        if valid:
            start_ms, end_ms = int(valid[0][0]), int(valid[-1][1])
        else:
            start_ms = segments[src[start]].start_ms
            end_ms = segments[src[cut]].end_ms
        if end_ms <= start_ms:
            end_ms = max(start_ms + 1, segments[src[cut]].end_ms)
        local_spans: list[list[int]] = []
        for a, b in spans:
            s2, e2 = max(a, start) - start, min(b, cut + 1) - start
            if e2 > s2:
                local_spans.append([s2, e2])
        flags: list[str] = []
        for sid in sorted(set(src[start:cut + 1])):
            for flag in segments[sid].risk_flags:
                if flag not in flags:
                    flags.append(flag)
        prios = [segments[sid].review_priority for sid in set(src[start:cut + 1])
                 if segments[sid].review_priority is not None]
        aligned = ([list(t) if t is not None else None for t in times[start:cut + 1]]
                   if valid else None)
        out.append(BuiltSegment(
            sequence_no=len(out), start_ms=start_ms, end_ms=end_ms, text=text,
            risk_flags=flags, low_conf_spans=local_spans or None,
            review_priority=max(prios) if prios else None,
            aligned_tokens=aligned,
        ))
        start = cut + 1
    return out


def _apply_dedup(chunk_texts: list[str], dedup_records: list[dict]) -> list[str]:
    """把 merge_texts 的去重记录回放到 chunk 级文本，得到去重后的分段文本。

    去重回写唯一实现见 ``local_asr.asr.segmentation.apply_dedup_to_texts``：
    ASR 脚本只对合并全文去重、chunk 原文原样写入 stats，入库时在此裁一次。
    注意不要在脚本层再回写一次，否则同一记录会被裁两遍（句首误裁）。
    """
    return apply_dedup_to_texts(chunk_texts, dedup_records)


def _dedup_offsets(chunk_texts: list[str], records: list[dict]) -> list[tuple[int, int]]:
    """计算每段去重裁剪后的 ``[起点, 终点)``（原文本码点下标）。

    去重按「有效字符数」裁，但 ``low_conf_spans`` 用的是码点下标；
    必须记录实际裁掉了多少个码点，才能把标亮位置平移到入库文本上。
    计算顺序与 ``apply_dedup_to_texts`` 完全一致（尾裁先、首裁后）。
    """
    bounds = [[0, len(t)] for t in chunk_texts]
    for rec in records:
        i = int(rec["boundary"])
        trimmed_prev = int(rec.get("trimmed_prev_chars") or 0)
        dropped = int(rec.get("dropped_chars") or 0)
        if trimmed_prev and 0 <= i < len(bounds):
            lo, hi = bounds[i]
            kept = drop_suffix_by_chars(chunk_texts[i][lo:hi], trimmed_prev)
            bounds[i][1] = lo + len(kept)
        if dropped and 0 <= i + 1 < len(bounds):
            lo, hi = bounds[i + 1]
            kept = drop_prefix_by_chars(chunk_texts[i + 1][lo:hi], dropped)
            bounds[i + 1][0] = hi - len(kept)
            # 与 apply_dedup_to_texts 完全同步：下一块开头带标点 → 保留上一块
            # 标点、去掉下一块重复标点；否则去掉上一块的块末伪标点。
            lo_p, hi_p = bounds[i]
            stripped_prev, tail_cut = strip_tail_punct(chunk_texts[i][lo_p:hi_p])
            lo_n, hi_n = bounds[i + 1]
            _, head_cut = strip_head_punct(chunk_texts[i + 1][lo_n:hi_n])
            if tail_cut and not head_cut:
                bounds[i][1] = lo_p + len(stripped_prev)
            elif tail_cut and head_cut:
                bounds[i + 1][0] = lo_n + head_cut
    return [(lo, hi) for lo, hi in bounds]


def _shift_spans(spans: list, start: int, text_len: int) -> list | None:
    """把码点区间平移到裁剪后的文本；越界部分丢弃，全部越界返回 None。"""
    out = []
    for span in spans:
        s, e = int(span[0]) - start, int(span[1]) - start
        s, e = max(0, s), min(text_len, e)
        if e > s:
            out.append([s, e])
    return out or None


def _slice_aligned(times: object, start: int, end: int, lead: int,
                   text_len: int) -> list | None:
    """把 chunk 级字符时间数组裁到入库文本的码点范围。

    与文本同步执行「去重裁剪 [start, end)」+「strip 掉前导空白 lead」；
    长度对不上时返回 None，前端自动回退按字数插值（宁可不给也不给错）。
    """
    if not isinstance(times, list) or not times:
        return None
    kept = times[start:end]
    kept = kept[lead:lead + text_len]
    if len(kept) != text_len:
        return None
    return kept if any(t is not None for t in kept) else None


def build_segments_from_stats(
    stats_path: Path, *, repack: bool = True,
) -> list[BuiltSegment]:
    """从 ASR stats.json 构建片段列表（VAD 级时间戳 + 去重后 chunk 文本）。

    ``repack``：默认把识别块按句子边界重新打包成展示段（见
    :func:`repack_display_segments`）；单元测试与离线审计可传 False
    保留「一个识别块一个段」的旧口径。
    """
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    chunks = stats.get("segments", [])
    dedup = stats.get("segmentation", {}).get("dedup", [])
    # 兼容：dedup 记录可能位于顶层
    if not dedup:
        dedup = stats.get("dedup", [])

    texts = [c.get("text", "") for c in chunks]
    deduped = _apply_dedup(texts, dedup) if dedup else texts
    # 去重裁剪会改变码点下标，low_conf_spans 必须同步平移（否则标亮错位）
    offsets = _dedup_offsets(texts, dedup) if dedup else [(0, len(t)) for t in texts]

    out: list[BuiltSegment] = []
    seq = 0
    for chunk, text, (start, end) in zip(chunks, deduped, offsets):
        raw = text or ""
        text = raw.strip()
        if not text:
            continue
        lead = len(raw) - len(raw.lstrip())  # strip 掉的空白也算偏移
        flags: list[str] = []
        if chunk.get("degraded"):
            flags.append("empty_transcript_degraded")
        if chunk.get("cut_silence_ms") is not None and chunk.get("cut_silence_ms", 0) == 0:
            flags.append("boundary_cut")  # 硬切边界（未吸附静音）
        # P1：ASR 脚本产出的段级风险（文本异常 / 对齐异常 / 低置信复核）
        for flag in chunk.get("risk_flags") or []:
            if isinstance(flag, str) and flag not in flags:
                flags.append(flag)
        low_conf = chunk.get("low_conf_spans")
        if low_conf and isinstance(low_conf, list) and len(low_conf) > 0:
            low_conf = _shift_spans(
                [[int(span[0]), int(span[1])] for span in low_conf],
                start + lead, len(text))
        else:
            low_conf = None
        out.append(
            BuiltSegment(
                sequence_no=seq,
                start_ms=int(round(float(chunk["start_s"]) * 1000)),
                end_ms=int(round(float(chunk["end_s"]) * 1000)),
                text=text,
                risk_flags=flags,
                low_conf_spans=low_conf,
                review_priority=(float(chunk["review_priority"])
                                 if chunk.get("review_priority") is not None else None),
                aligned_tokens=_slice_aligned(
                    chunk.get("char_times"), start, end, lead, len(text)),
            )
        )
        seq += 1
    # §5.7 校验：按 start 排序 + start < end
    out.sort(key=lambda s: s.start_ms)
    out = [s for s in out if s.end_ms > s.start_ms]
    for i, s in enumerate(out):
        s.sequence_no = i
    if repack:
        out = repack_display_segments(out)
    return out


def save_segments(db: Database, run_id: str, segments: list[BuiltSegment]) -> int:
    """写入 transcript_segments 表。"""
    for s in segments:
        low_conf_json = json.dumps(s.low_conf_spans, ensure_ascii=False) if s.low_conf_spans else None
        aligned_json = json.dumps(s.aligned_tokens, ensure_ascii=False) if s.aligned_tokens else None
        db.execute(
            """INSERT INTO transcript_segments
               (id, run_id, sequence_no, start_ms, end_ms, original_text,
                confidence_source, risk_flags, low_conf_spans, review_priority,
                aligned_tokens, review_status, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 'none', ?, ?, ?, ?, 'pending', ?)""",
            (
                new_id("seg"), run_id, s.sequence_no, s.start_ms, s.end_ms,
                s.text, json.dumps(s.risk_flags, ensure_ascii=False),
                low_conf_json, s.review_priority, aligned_json, now_iso(),
            ),
        )
    return len(segments)


def load_segments(db: Database, run_id: str) -> list[dict]:
    """查询片段（§10.1 数据结构；final_text = edited ?? original）。

    响应键统一驼峰（与 §11 其它端点一致，前端 TS 类型对齐）。
    """
    rows = db.query(
        "SELECT * FROM transcript_segments WHERE run_id = ? ORDER BY sequence_no",
        (run_id,),
    )
    out = []
    for r in rows:
        d = dict(r)
        d["riskFlags"] = json.loads(d.pop("risk_flags") or "[]")
        d["finalText"] = d.get("edited_text") if d.get("edited_text") is not None else d["original_text"]
        # 蛇形列名 -> 驼峰响应键
        for snake, camel in (
            ("run_id", "runId"), ("sequence_no", "sequenceNo"),
            ("start_ms", "startMs"), ("end_ms", "endMs"),
            ("original_text", "originalText"), ("edited_text", "editedText"),
            ("confidence_value", "confidenceValue"),
            ("confidence_source", "confidenceSource"),
            ("review_status", "reviewStatus"),
            ("aligned_tokens", "alignedTokens"), ("updated_at", "updatedAt"),
            ("low_conf_spans", "lowConfSpans"),
            ("review_priority", "reviewPriority"),
        ):
            if snake in d:
                d[camel] = d.pop(snake)
        if "lowConfSpans" in d and isinstance(d.get("lowConfSpans"), str):
            try:
                d["lowConfSpans"] = json.loads(d["lowConfSpans"])
            except (json.JSONDecodeError, TypeError):
                d["lowConfSpans"] = None
        if "alignedTokens" in d and isinstance(d.get("alignedTokens"), str):
            try:
                d["alignedTokens"] = json.loads(d["alignedTokens"])
            except (json.JSONDecodeError, TypeError):
                d["alignedTokens"] = None
        out.append(d)
    return out
