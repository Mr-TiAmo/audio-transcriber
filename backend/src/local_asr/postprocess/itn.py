"""中文数字规范化（ITN）：wetext 规则引擎 + 定向修正。

设计原则（来自方案）：
1. wetext 是规则引擎，不是通用语义模型，不能整篇无条件采信。
2. 数字按类型处理：金额、日期、百分比、点位、编号；保留成语、概数、自然表达。
3. 多义表达（口语年份等）不猜，保留原文并记入复核项。
4. 已有阿拉伯数字不重新计算，精度用字符串保留。

用法::

    from local_asr.postprocess.itn import normalize_numbers

    result = normalize_numbers("成交额三万亿，跌了百分之三十六点七幺")
    result.text    # "成交额3万亿，跌了36.71%"
    result.notes   # 需要人工复核的片段
"""

from __future__ import annotations

import difflib
import multiprocessing as mp
import os
import re
from dataclasses import dataclass, field, replace

from local_asr.postprocess.verify import Edit

__all__ = [
    "ItnNote", "ItnResult", "normalize_numbers", "normalize_numbers_segmented",
    "needs_itn", "cn_to_int",
]


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ItnNote:
    """需要人工复核的片段。"""

    kind: str
    snippet: str
    reason: str
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class ItnResult:
    """数字规范化结果。"""

    text: str
    notes: tuple[ItnNote, ...] = field(default_factory=tuple)
    mappings: tuple[dict, ...] = field(default_factory=tuple)
    edits: tuple[Edit, ...] = field(default_factory=tuple)
    degraded: bool = False


# ---------------------------------------------------------------------------
# 中文数字 -> 整数（仅用于补规则，覆盖 0-9999 及"万"）
# ---------------------------------------------------------------------------

_CN_DIGIT = {
    "零": 0, "〇": 0, "一": 1, "幺": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_SMALL_UNIT = {"十": 10, "百": 100, "千": 1000}


def cn_to_int(text: str) -> int | None:
    """中文数字转整数。无法解析时返回 None。"""
    if not text:
        return None
    if all(ch in _CN_DIGIT for ch in text):
        return int("".join(str(_CN_DIGIT[ch]) for ch in text))
    total = section = num = 0
    for ch in text:
        if ch in _CN_DIGIT:
            num = _CN_DIGIT[ch]
        elif ch in _CN_SMALL_UNIT:
            section += (num if num else 1) * _CN_SMALL_UNIT[ch]
            num = 0
        elif ch == "万":
            section = (section + num) * 10000
            total += section
            section = num = 0
        else:
            return None
    return total + section + num


# ---------------------------------------------------------------------------
# 定向规则
# ---------------------------------------------------------------------------

# 1) "三万亿" 经 wetext 会变成 "30000亿"；恢复万/亿的阅读尺度。
_WAN_YI_RE = re.compile(r"(?<![\d.])(\d{5,})亿")

# 2) "三个百分点" wetext 不转换，补转数字（不能变成 %）。
_PERCENT_POINT_RE = re.compile(r"([零〇一二两三四五六七八九十百千]+)个百分点")

# 3) 口语两位年份（"二三年""九九年"）wetext 会误转成 "23/07" 这类日期；
#    这类表达本身有歧义，保留原文并标记复核。
_AMBIG_YEAR_RE = re.compile(r"(?<![零〇一二三四五六七八九])([零〇一二三四五六七八九]{2})年")

# 3b) 年份数字串：标准年份是 4 位（"二零二一"）。长度不是 4 的整数倍的年份串
#     多是识别重复（"二零二零二一"），wetext 会当成大整数输出 "202 021"，
#     与相邻年份拼成 "2021~202021"。这类不猜，保留原文并标记复核。
_YEAR_RUN_RE = re.compile(r"(?:一九|二零|二〇)[零〇一二三四五六七八九]+")

# 3c) 连续数词串：标准中文数字里「十」最多出现一次（如「一千一百一十」）。
#     同一段连续数词出现两个以上「十」，多是识别重复（"十五十十五"），
#     wetext 会切成 "15 10 15"。这类不猜，保留原文并标记复核。
_NUM_RUN_RE = re.compile(r"[零〇一二两三四五六七八九十百千万亿]{3,}")

# 4) 电话/账号/编号：wetext 会插入空格（"0031 358"），需要去掉。
_PHONE_CTX_RE = re.compile(r"((?:电话|手机|号码|账号|工号|编号|代码)[：:\s]*)([0-9][0-9\s]{2,})")

# 5) 明确保护、绝不参与替换的表达（wetext 通常已处理好，这里是双保险）
_PROTECTED_PHRASES = (
    "一模一样", "三心二意", "五花八门", "七上八下", "乱七八糟", "九牛一毛",
    "十全十美", "一心一意", "一五一十", "一清二楚", "三分之一", "一分为二",
    "一会儿", "一下", "一些", "一起", "一定", "一般", "一直", "一边", "一共",
    "一旦", "一味", "一番", "一致", "一向", "一律", "唯一", "统一", "万一",
    "十分", "百分百", "两下", "三番五次", "四面八方", "两三个月", "十几", "几十", "几百",
    "几千", "几万", "上百", "上千", "上万",
)

_LITERAL_SLOT_RE = re.compile("+[^]*")
_ARABIC_RE = re.compile(r"[+＋\-−－¥￥$]?\d+(?:[.,，:/\-]\d+)*(?:[ \t]+\d+)*")
_PHONE_SPOKEN_RE = re.compile(r"(?:电话|手机|号码|账号|工号|编号|代码)[：:\s]*([零〇一幺二三四五六七八九]+)")


class _TrackedText:
    """字符来源跨度用于组合各阶段规则，最终编辑仍以原文坐标存储。"""

    def __init__(self, text: str):
        self.text = text
        self.origins = [(i, i + 1) for i in range(len(text))]
        self.events: list[tuple[int, int, str]] = []

    def span(self, start: int, end: int) -> tuple[int, int]:
        if start == end:
            pos = self.origins[start][0] if start < len(self.origins) else (self.origins[-1][1] if self.origins else 0)
            return pos, pos
        selected = self.origins[start:end]
        return min(a for a, _ in selected), max(b for _, b in selected)

    def apply(self, output: str, rule: str) -> None:
        origins = []
        for op, a, b, c, d in difflib.SequenceMatcher(None, self.text, output, autojunk=False).get_opcodes():
            if op == "equal":
                origins.extend(self.origins[a:b])
            else:
                start, end = self.span(a, b)
                origins.extend([(start, end)] * (d - c))
                self.events.append((start, end, rule))
        self.text, self.origins = output, origins


def _ambiguous_year_runs(text: str):
    """长度不是 4 的整数倍、且不在电话/编号上下文里的年份数字串。"""
    phone_spans = [m.span(1) for m in _PHONE_SPOKEN_RE.finditer(text)]
    runs = []
    for match in _YEAR_RUN_RE.finditer(text):
        if len(match.group()) % 4 == 0:
            continue
        if any(start <= match.start() and match.end() <= end for start, end in phone_spans):
            continue
        runs.append(match)
    return runs


def _ambiguous_number_runs(text: str):
    """连续数词里出现两个以上「十」的串（正常中文数字最多一个「十」）。"""
    phone_spans = [m.span(1) for m in _PHONE_SPOKEN_RE.finditer(text)]
    runs = []
    for match in _NUM_RUN_RE.finditer(text):
        if match.group().count("十") < 2:
            continue
        if any(start <= match.start() and match.end() <= end for start, end in phone_spans):
            continue
        runs.append(match)
    return runs


def _protected_spans(text: str, glossary: dict[str, str]):
    candidates = []
    for phrase in sorted(set(glossary) | set(_PROTECTED_PHRASES), key=lambda s: (-len(s), s)):
        for match in re.finditer(re.escape(phrase), text):
            candidates.append((match.start(), match.end(), glossary.get(phrase, phrase),
                               "glossary" if phrase in glossary else "protect"))
    for pattern in (_AMBIG_YEAR_RE, _LITERAL_SLOT_RE, _ARABIC_RE):
        candidates.extend((m.start(), m.end(), m.group(), "protect") for m in pattern.finditer(text))
    candidates.extend((m.start(), m.end(), m.group(), "protect")
                      for m in _ambiguous_year_runs(text))
    candidates.extend((m.start(), m.end(), m.group(), "ambiguous_number")
                      for m in _ambiguous_number_runs(text))
    cursor = 0
    selected = []
    for start, end, replacement, rule in sorted(candidates, key=lambda item: (item[0], -(item[1] - item[0]))):
        if start >= cursor:
            selected.append((start, end, replacement, rule))
            cursor = end
    return selected


def _fix_wan_yi(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        value = int(m.group(1))
        if value >= 10000 and value % 10000 == 0:
            return f"{value // 10000}万亿"
        return m.group(0)

    return _WAN_YI_RE.sub(repl, text)


def _fix_percent_point(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        value = cn_to_int(m.group(1))
        return f"{value}个百分点" if value is not None else m.group(0)

    return _PERCENT_POINT_RE.sub(repl, text)


def _fix_phone_spacing(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        return m.group(1) + re.sub(r"\s+", "", m.group(2))

    return _PHONE_CTX_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

_NORMALIZER = None


def _get_normalizer():
    global _NORMALIZER
    if _NORMALIZER is None:
        try:
            from wetext import Normalizer
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise RuntimeError(
                "数字规范化需要 wetext。请安装可选依赖：pip install -e '.[postprocess]'"
            ) from exc

        _NORMALIZER = Normalizer(lang="zh", operator="itn")
    return _NORMALIZER


def preflight_itn() -> None:
    """预热本地 FST 及映射接口，依赖或规则加载错误在昂贵推理前明确失败。"""
    result = _get_normalizer().normalize_with_mapping("零")
    if not isinstance(result.output_text, str) or not hasattr(result, "mappings"):
        raise RuntimeError("wetext 规则或映射接口不可用，请使用已准备的 wetext>=0.2.1 环境")


def normalize_numbers(text: str, glossary: dict[str, str] | None = None) -> ItnResult:
    """中文数字规范化。

    Args:
        text: 待规范化的中文文本（通常是 ASR 原始输出）。
        glossary: 实体词典，如 ``{"科创五零": "科创50"}``；只在明确命中时替换。

    Returns:
        ItnResult，包含规范化文本、复核项与 wetext 映射。
    """
    if not text:
        return ItnResult(text="")

    glossary = glossary if glossary is not None else {}
    if not isinstance(glossary, dict) or any(
        not isinstance(k, str) or not k or not isinstance(v, str) or not v
        for k, v in glossary.items()
    ):
        raise ValueError("实体词典必须包含非空字符串键和值")
    notes = [ItnNote("ambiguous_year", m.group(), "两位口语年份有歧义，保留原文待确认", *m.span())
             for m in _AMBIG_YEAR_RE.finditer(text)]
    notes.extend(ItnNote("ambiguous_year", m.group(),
                         "年份数字串不是标准 4 位年份，疑似识别重复，保留原文待确认", *m.span())
                 for m in _ambiguous_year_runs(text))
    notes.extend(ItnNote("ambiguous_number", m.group(),
                         "连续数词中出现多个「十」，不是单个数字，疑似识别重复，保留原文待确认",
                         *m.span())
                 for m in _ambiguous_number_runs(text))
    tracked = _TrackedText(text)
    slots = []
    parts = []
    cursor = 0
    prefix = ""
    while prefix in text:
        prefix += ""
    for index, (start, end, replacement, rule) in enumerate(_protected_spans(text, glossary)):
        token = f"{prefix}{index}"
        parts.extend((text[cursor:start], token))
        slots.append((token, start, end, replacement, rule))
        cursor = end
    parts.append(text[cursor:])
    tracked.apply("".join(parts), "protect")

    result = _get_normalizer().normalize_with_mapping(tracked.text)
    output = result.output_text
    if not isinstance(output, str):
        raise ValueError("wetext 未返回有效正文")
    # 不把占位符文本的字符坐标直接输出；映射来源转换到原始正文范围。
    for mapping in result.mappings:
        a, b = mapping.input_start, mapping.input_end
        if not 0 <= a <= b <= len(tracked.text) or tracked.text[a:b] != mapping.input_text:
            raise ValueError("wetext 输入映射不一致")
        if output[mapping.output_start:mapping.output_end] != mapping.output_text:
            raise ValueError("wetext 输出映射不一致")
        start, end = tracked.span(a, b)
        tracked.events.append((start, end, f"wetext:{mapping.token_type}"))
    tracked.apply(output, "wetext")
    tracked.apply(_fix_wan_yi(tracked.text), "wan_yi")
    tracked.apply(_fix_percent_point(tracked.text), "percent_point")

    phone_spans = [m.span(1) for m in _PHONE_SPOKEN_RE.finditer(text)]
    def phone_replacement(match):
        a, b = tracked.span(*match.span(2))
        if any(start <= a and b <= end for start, end in phone_spans):
            return match.group(1) + re.sub(r"[ \t]+", "", match.group(2))
        return match.group()
    tracked.apply(_PHONE_CTX_RE.sub(phone_replacement, tracked.text), "phone")

    positions = [tracked.text.find(token) for token, *_ in slots]
    if (any(tracked.text.count(token) != 1 for token, *_ in slots) or positions != sorted(positions)):
        notes.append(ItnNote("placeholder_failure", "", "保护范围损坏，已回退 ITN 输入"))
        return ItnResult(text=text, notes=tuple(notes), degraded=True)
    if slots:
        lookup = {token: replacement for token, _, _, replacement, _ in slots}
        pattern = re.compile("|".join(re.escape(token) for token in lookup))
        restored = pattern.sub(lambda m: lookup[m.group()], tracked.text)
        tracked.apply(restored, "restore")
        tracked.events.extend((start, end, rule) for _, start, end, replacement, rule in slots
                              if replacement != text[start:end])
    edits = []
    mappings = []
    for op, a, b, c, d in difflib.SequenceMatcher(None, text, tracked.text, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        rules = sorted({rule for start, end, rule in tracked.events
                        if (start < b and end > a or a == b and start <= a <= end)
                        and rule not in ("protect", "restore")})
        rule = "+".join(rules) or "wetext"
        edits.append(Edit(op, a, b, text[a:b], tracked.text[c:d], "accepted",
                          stage="itn", rule=rule, output_start=c, output_end=d))
        mappings.append({"kind": op, "token_type": rule, "input_start": a, "input_end": b,
                         "output_start": c, "output_end": d,
                         "input_text": text[a:b], "output_text": tracked.text[c:d]})
    return ItnResult(tracked.text, tuple(notes), tuple(mappings), tuple(edits))


# ---------------------------------------------------------------------------
# 分段并行 / 预筛（长文本 ITN 优化）
#
# wetext ITN 的成本与字符量近似线性（实测约 4.5–5 ms/字），因此长文本
# 按句末标点切块后多进程并行。
#
# 早期版本用「含数字候选的句子」做预筛，只处理命中区间（10min 12.2s→1.4s），
# 但 60min 文本实测发现预筛会漏改：wetext 对「单个数词 + 任意名词」也会
# 转换（「西湖十景」→「西湖10景」），而「十分」「一起」「一季度」保留——
# 这条边界无法用正则枚举。现在改为**全句切块**（不跳过任何句子）：
# 60min 文本 100s→29.5s（4 进程，3.4×），与整篇逐字符一致；
# 10min 文本约 3s，仍远快于整篇。
# ---------------------------------------------------------------------------

# 句子切分：保留句末标点与换行，拼接后必须等于原文
_ITN_SENT_RE = re.compile(r"[^。！？!?\n]*(?:[。！？!?]+|\n|$)")

# 宽口径「存在数字/数词」判断：用于快速路径（既无数词也无词典 → 必无改写）。
# 不用于切分：wetext 的转换边界比这里宽，基于候选词的切分会漏改。
_ANY_NUMBER_RE = re.compile(r"[0-9０-９零〇一二两三四五六七八九十百千万亿]")


def needs_itn(text: str) -> bool:
    """文本是否包含数字/中文数词（宽口径，宁可多跑不可漏改）。"""
    return bool(_ANY_NUMBER_RE.search(text))


def _itn_spans(text: str, chunk_chars: int) -> list[tuple[int, int]]:
    """按句末标点切块（原文坐标），覆盖全文，块长接近 chunk_chars。"""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _ITN_SENT_RE.finditer(text):
        if not match.group(0):
            continue
        end = match.end()
        if end - start >= chunk_chars:
            spans.append((start, end))
            start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _normalize_block(payload: tuple[str, dict[str, str]]) -> ItnResult:
    """进程池任务：规范化一个连续区间（模块级函数，Windows spawn 需要）。"""
    text, glossary = payload
    return normalize_numbers(text, glossary=glossary)


def _shift_notes(notes: tuple[ItnNote, ...], offset: int) -> list[ItnNote]:
    return [replace(
        n,
        start=(n.start + offset if n.start is not None else None),
        end=(n.end + offset if n.end is not None else None),
    ) for n in notes]


def normalize_numbers_segmented(
    text: str,
    glossary: dict[str, str] | None = None,
    *,
    workers: int | None = None,
    min_chars: int = 1200,
    chunk_chars: int = 800,
) -> ItnResult:
    """按句切块 + 多进程并行的数字规范化（与整篇结果逐字符一致）。

    每块沿用 :func:`normalize_numbers`，块间在句末标点处切开，因此
    跨块的编辑/映射坐标可安全平移回全文。

    快速路径：文本短于 ``min_chars``，或既无数词/数字且无词典时，
    直接走整篇（进程开销不划算，且不可能产生改写）。

    Args:
        text: 待规范化全文。
        glossary: 实体词典。
        workers: 并行进程数；None = min(8, CPU 核数)，1 = 串行。
        min_chars: 文本长度下限（低于则整篇处理）。
        chunk_chars: 单个切块的目标字符数。
    """
    if not text:
        return ItnResult(text="")
    glossary = glossary if glossary is not None else {}
    if len(text) < min_chars or (not glossary and not needs_itn(text)):
        return normalize_numbers(text, glossary=glossary)
    spans = _itn_spans(text, chunk_chars)
    if len(spans) < 2:
        return normalize_numbers(text, glossary=glossary)

    blocks = [text[s:e] for s, e in spans]
    worker_count = workers
    if worker_count is None:
        # 24 核机器实测：4 进程 20.8s、8 进程 13.6s（60min 文本）；
        # 再往上收益递减且每进程各持一份 wetext 运行时。
        worker_count = min(8, os.cpu_count() or 1)
    if worker_count > 1 and len(blocks) > 1:
        with mp.Pool(worker_count) as pool:
            results = pool.map(_normalize_block, [(b, glossary) for b in blocks])
    else:
        results = [_normalize_block((b, glossary)) for b in blocks]

    parts: list[str] = []
    notes: list[ItnNote] = []
    mappings: list[dict] = []
    edits: list[Edit] = []
    degraded = False
    cursor = 0
    for (start, end), result in zip(spans, results):
        parts.append(text[cursor:start])
        out_offset = sum(len(p) for p in parts)
        parts.append(result.text)
        notes.extend(_shift_notes(result.notes, start))
        mappings.extend({
            **m,
            "input_start": m["input_start"] + start,
            "input_end": m["input_end"] + start,
            "output_start": m["output_start"] + out_offset,
            "output_end": m["output_end"] + out_offset,
        } for m in result.mappings)
        edits.extend(replace(
            e,
            start=e.start + start,
            end=e.end + start,
            output_start=e.output_start + out_offset,
            output_end=e.output_end + out_offset,
            candidate_start=e.candidate_start + start,
            candidate_end=e.candidate_end + start,
        ) for e in result.edits)
        degraded = degraded or result.degraded
        cursor = end
    parts.append(text[cursor:])
    return ItnResult("".join(parts), tuple(notes), tuple(mappings), tuple(edits), degraded)