"""语义纠错结果的修改校验。

原则（来自方案）：
- 每项改动必须能准确定位原文，无法对齐的输出不自动应用。
- 语义纠错阶段不自动放行数值、编号、日期、单位、正负号及否定关系变化。
- 单段编辑比例超过阈值（默认 10%）的整体转为待复核。
- 校验能减少误改，但不能证明语义正确。
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

__all__ = ["Edit", "review_corrections", "MAX_EDIT_RATIO"]

MAX_EDIT_RATIO = 0.10

_NUM_RE = re.compile(r"[0-9零〇一幺二三四五六七八九十百千万亿两]")
_UNIT_RE = re.compile(r"[元块万亿%％倍点年月日号个只次天周岁分秒]")
_NEG_RE = re.compile(r"[不没无未别非否]")

# 连同符号、精度、日期分隔符和单位保护，不能只检查 diff 中变化的字符。
_NUMERIC_EXPRESSION = re.compile(
    r"[+＋\-−－正负]?(?:人民币|美元|港元|欧元|[¥￥$€£])?[+＋\-−－正负]?(?:百分之)?"
    r"[0-9零〇一幺二两三四五六七八九十百千万亿]+"
    r"(?:[.．点,，:/年月日\-][0-9零〇一幺二两三四五六七八九十百千万亿]+)*"
    r"(?:个百分点|百分比|万亿元|亿元|万元|万亿|人民币|美元|港元|欧元|"
    r"公斤|千克|公里|千米|厘米|毫米|毫升|平方米|立方米|kg|km|mL|"
    r"元|块|%|％|倍|点|年|月|日|号|个|只|次|天|周|岁|分|秒|吨|米|升)?"
)


def numeric_spans(text: str) -> list[tuple[int, int]]:
    """完整正文中的数值范围，可映射到段内坐标以保护跨段表达式。"""
    return [m.span() for m in _NUMERIC_EXPRESSION.finditer(text)]


def _touches(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= start <= b if start == end else start < b and end > a
               for a, b in spans)


@dataclass(frozen=True)
class Edit:
    """一条待校验的修改。"""

    op: str          # replace / insert / delete
    start: int       # 原文起始位置
    end: int         # 原文结束位置
    before: str
    after: str
    decision: str    # accepted / review
    reason: str = ""
    stage: str = "correction"
    rule: str = "semantic_review"
    output_start: int = 0
    output_end: int = 0
    candidate_start: int = 0
    candidate_end: int = 0


def _risk_reason(before: str, after: str) -> str:
    """判断一处改动是否涉及需要人工复核的内容。"""
    combo = before + after
    if _NUM_RE.search(combo):
        return "涉及数字或数量表达"
    if _UNIT_RE.search(combo):
        return "涉及单位或时间量词"
    if _NEG_RE.search(combo):
        return "涉及否定关系"
    return ""


def review_corrections(
    raw: str,
    corrected: str,
    max_edit_ratio: float = MAX_EDIT_RATIO,
    *,
    protected_raw_spans: list[tuple[int, int]] | None = None,
    protected_candidate_spans: list[tuple[int, int]] | None = None,
) -> tuple[str, list[Edit]]:
    """校验并应用纠错结果。

    Args:
        raw: ASR 原始文本（事实底稿）。
        corrected: 纠错模型输出。
        max_edit_ratio: 单段允许的最大编辑字符占比，超过则整段回退。

    Returns:
        (最终采用文本, 编辑记录)。编辑记录包含 accepted 与 review 两类；
        review 类改动不会被应用到最终文本。
    """
    if not 0 <= max_edit_ratio <= 1:
        raise ValueError("max_edit_ratio 必须在 0 到 1 之间")
    if corrected == raw:
        return raw, []

    sm = difflib.SequenceMatcher(None, raw, corrected, autojunk=False)
    opcodes = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if not opcodes:
        return raw, []

    changed = sum(
        max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in opcodes
    )
    ratio = changed / len(raw) if raw else 1.0
    whole_review = ratio > max_edit_ratio

    edits: list[Edit] = []
    pieces: list[str] = []
    raw_spans = numeric_spans(raw) + (protected_raw_spans or [])
    candidate_spans = numeric_spans(corrected) + (protected_candidate_spans or [])
    output_pos = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pieces.append(raw[i1:i2])
            output_pos += i2 - i1
            continue

        before, after = raw[i1:i2], corrected[j1:j2]
        risk = _risk_reason(before, after)
        if _touches(i1, i2, raw_spans) or _touches(j1, j2, candidate_spans):
            risk = "涉及完整数字表达式（符号、精度、编号或单位）"
        if whole_review:
            decision = "review"
            ratio_msg = f"整段编辑比例 {ratio:.1%} 超过阈值 {max_edit_ratio:.0%}"
            reason = f"{risk}；{ratio_msg}" if risk else ratio_msg
        elif risk:
            decision = "review"
            reason = risk
        else:
            decision = "accepted"
            reason = ""

        adopted = after if decision == "accepted" else before
        edits.append(Edit(
            op=tag,
            start=i1,
            end=i2,
            before=before,
            after=after,
            decision=decision,
            reason=reason,
            output_start=output_pos,
            output_end=output_pos + len(adopted),
            candidate_start=j1,
            candidate_end=j2,
        ))
        pieces.append(adopted)
        output_pos += len(adopted)

    return "".join(pieces), edits


def replay_edits(source: str, edits: list[Edit]) -> str:
    """按某一阶段输入坐标重放接受的修改，拒绝重叠或错位记录。"""
    cursor = 0
    pieces = []
    for edit in edits:
        if edit.decision != "accepted":
            continue
        if not cursor <= edit.start <= edit.end <= len(source):
            raise ValueError("编辑范围重叠或越界")
        if source[edit.start:edit.end] != edit.before:
            raise ValueError("编辑记录与输入不一致")
        pieces.extend((source[cursor:edit.start], edit.after))
        cursor = edit.end
    pieces.append(source[cursor:])
    return "".join(pieces)