#!/usr/bin/env python3
"""数值保真检查：独立于 CER 的数字/单位/编号核对。

CER 会把数字、标点和虚词归一化后比较，无法回答"这个数字有没有被改错"。
本脚本单独比较参考与识别文本中的数值表达式。

用法:
    ./.venv/bin/python scripts/check_numeric_fidelity.py \
        output/测试音频_实际字幕版.txt output/bench/*.txt

输出每个文本的：数值命中率、缺失数值、单位不一致、百分比/百分点混用。
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# 独立评测解析器：不调用待测 ITN 规则，不把规则输出当作参考答案。
_DIGITS = dict(zip("零〇一幺二两三四五六七八九", "0011223456789"))
_SMALL = {"十": 10, "百": 100, "千": 1000}
CN_NUM = "零〇一幺二两三四五六七八九十百千万亿"
UNITS = ("个百分点", "百分比", "%", "％", "点", "倍", "元", "块", "美元", "港元", "欧元",
         "年", "月", "日", "号", "个", "只", "次", "天", "周", "岁", "分", "秒",
         "公斤", "公里", "千克", "千米", "米", "吨")
_BODY = rf"(?:[0-9]+(?:[,，][0-9]{{3}})*(?:\.[0-9]+)?|[{CN_NUM}]+(?:点[零〇一幺二两三四五六七八九0-9]+)?)"
_VALUE_RE = re.compile(
    rf"(?P<currency>[¥￥$])?(?P<sign>[+＋\-−－正负])?(?P<percent>百分之)?"
    rf"(?P<number>{_BODY})(?P<scale>万亿|亿|万)?(?P<unit>{'|'.join(sorted(UNITS, key=len, reverse=True))})?"
)
_DATE_PART = r"[0-9零〇一幺二两三四五六七八九十]+"
_DATE_RE = re.compile(
    rf"(?P<y>{_DATE_PART})年(?P<m>{_DATE_PART})月(?:(?P<d>{_DATE_PART})[日号])?"
    r"|(?P<ay>\d{2,4})[-/](?P<am>\d{1,2})(?:[-/](?P<ad>\d{1,2}))?"
)
_IDENTIFIER_CONTEXT = re.compile(r"(?:电话|手机|号码|账号|工号|编号|代码)[：:\s]*(?:是|为)?$")
_NATURAL = re.compile(r"一模一样|三心二意|五花八门|一心一意|一下|一些|一起|一直|统一|万一|唯一|两三个月|十几|几十")


def _integer(raw: str) -> int:
    if not raw:
        return 0
    if raw.isascii() and raw.isdigit():
        return int(raw)
    if all(c in _DIGITS for c in raw):
        return int("".join(_DIGITS[c] for c in raw))
    for unit, scale in (("亿", 100000000), ("万", 10000)):
        if unit in raw:
            left, right = raw.split(unit, 1)
            if not left or unit in right:
                raise ValueError("缺失数量或重复尺度")
            return _integer(left) * scale + _integer(right)
    total = number = 0
    previous = 10000
    for char in raw:
        if char in _DIGITS:
            number = int(_DIGITS[char])
        elif char in _SMALL and _SMALL[char] < previous:
            total += (number or 1) * _SMALL[char]
            previous, number = _SMALL[char], 0
        else:
            raise ValueError("无法解析数量")
    return total + number


def _format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def normalize_value(raw: str) -> str | None:
    """精确转换数值；无尺度的逐位数字保留前导零。"""
    if not raw:
        return None
    raw = raw.replace(",", "").replace("，", "")
    sign = ""
    if raw[0] in "+＋-−－正负":
        sign = "-" if raw[0] in "-−－负" else ""
        raw = raw[1:]
    try:
        if not raw:
            return None
        if all(c in _DIGITS or c in "0123456789" for c in raw):
            return sign + "".join(_DIGITS.get(c, c) for c in raw)
        if re.fullmatch(r"\d+(?:\.\d+)?", raw):
            return sign + _format_decimal(Decimal(raw))
        if "点" in raw:
            integer, fraction = raw.split("点", 1)
            if not fraction or not all(c in _DIGITS or c.isascii() and c.isdigit() for c in fraction):
                return None
            return sign + _format_decimal(Decimal(str(_integer(integer)) + "." + "".join(_DIGITS.get(c, c) for c in fraction)))
        return sign + str(_integer(raw))
    except (ValueError, InvalidOperation):
        return None


@dataclass(frozen=True)
class NumericValue:
    start: int
    end: int
    raw: str
    value: str
    unit: str
    kind: str

    @property
    def key(self):
        return self.value, self.unit


def extract_values(text: str) -> list[NumericValue]:
    values = []
    excluded = [m.span() for m in _NATURAL.finditer(text)]
    for m in _DATE_RE.finditer(text):
        year, month, day = m.group("y") or m.group("ay"), m.group("m") or m.group("am"), m.group("d") or m.group("ad")
        try:
            y, mo = _integer(year), _integer(month)
            d = _integer(day) if day else None
            if not 1 <= mo <= 12 or d is not None and not 1 <= d <= 31:
                continue
        except ValueError:
            continue
        value = f"{y}-{mo}" + (f"-{d}" if d is not None else "")
        values.append(NumericValue(*m.span(), m.group(), value, "日期", "date"))
        excluded.append(m.span())
    for m in _VALUE_RE.finditer(text):
        if any(m.start() < end and m.end() > start for start, end in excluded):
            continue
        raw = (m.group("sign") or "") + m.group("number")
        value = normalize_value(raw)
        if value is None:
            continue
        unit = m.group("unit") or "(无)"
        unit = {"％": "%", "百分比": "%", "块": "元"}.get(unit, unit)
        if m.group("percent"):
            unit = "%"
        if m.group("currency"):
            unit = "美元" if m.group("currency") == "$" else "元"
        identifier = bool(_IDENTIFIER_CONTEXT.search(text[max(0, m.start() - 12):m.start()]))
        identifier |= unit == "(无)" and len(value) > 1 and value.startswith("0") and "." not in value
        scale = {None: 1, "万": 10000, "亿": 100000000, "万亿": 1000000000000}[m.group("scale")]
        if identifier and scale == 1 and unit == "(无)":
            kind, unit = "identifier", "编号"
        else:
            with localcontext() as ctx:
                ctx.prec = max(80, len(value) + 20)
                value = _format_decimal(Decimal(value) * scale)
            kind = "percent" if unit == "%" else "percent_point" if unit == "个百分点" else "quantity"
        values.append(NumericValue(*m.span(), m.group(), value, unit, kind))
    return sorted(values, key=lambda v: v.start)


def extract(text: str) -> dict[tuple[str, str], int]:
    return dict(Counter(v.key for v in extract_values(text)))


def _masked(text: str, values: list[NumericValue], marker: str):
    pieces, positions = [], {}
    cursor = size = 0
    for index, value in enumerate(values):
        prefix = text[cursor:value.start]
        pieces.extend((prefix, marker))
        size += len(prefix)
        positions[size] = index
        size += 1
        cursor = value.end
    pieces.append(text[cursor:])
    return "".join(pieces), positions


def compare(ref_text: str, hyp_text: str) -> dict:
    ref_values, hyp_values = extract_values(ref_text), extract_values(hyp_text)
    ref, hyp = Counter(v.key for v in ref_values), Counter(v.key for v in hyp_values)
    marker = next(chr(c) for c in range(0xE000, 0xF8FF) if chr(c) not in ref_text + hyp_text)
    a, ref_positions = _masked(ref_text, ref_values, marker)
    b, hyp_positions = _masked(hyp_text, hyp_values, marker)
    pairs = []
    for block in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for pos, index in ref_positions.items():
            if block.a <= pos < block.a + block.size:
                other = hyp_positions.get(block.b + pos - block.a)
                if other is not None:
                    pairs.append((ref_values[index], hyp_values[other]))
    hit = sum(r.key == h.key for r, h in pairs)
    bag_hit = sum((ref & hyp).values())
    mismatches = [{"ref": r.raw, "hyp": h.raw, "ref_start": r.start, "hyp_start": h.start}
                  for r, h in pairs if r.key != h.key]
    return {
        "metric_version": 2,
        "ref_values": len(ref_values), "hyp_values": len(hyp_values), "hit": hit,
        "rate": hit / len(ref_values) if ref_values else (1.0 if not hyp_values else 0.0),
        "multiset_hit": bag_hit,
        "multiset_rate": bag_hit / len(ref_values) if ref_values else (1.0 if not hyp_values else 0.0),
        "format_exact_rate": sum(r.raw == h.raw for r, h in pairs) / len(ref_values) if ref_values else None,
        "missing": list((ref - hyp).elements()), "extra": list((hyp - ref).elements()),
        "position_mismatches": mismatches,
        "percent_point_mix": any({r.unit, h.unit} == {"%", "个百分点"} for r, h in pairs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("hyps", nargs="+")
    args = ap.parse_args()

    ref_text = Path(args.ref).read_text(encoding="utf-8")
    print(f"参考: {args.ref}")
    print("-" * 70)

    for hyp_arg in args.hyps:
        p = Path(hyp_arg)
        if not p.exists():
            continue
        r = compare(ref_text, p.read_text(encoding="utf-8"))
        print(f"{p.name}")
        print(f"  数值命中 {r['hit']}/{r['ref_values']} = {r['rate'] * 100:.1f}%")
        if r["missing"]:
            print(f"  缺失: {r['missing']}")
        if r["extra"]:
            print(f"  多余: {r['extra']}")
        if r["position_mismatches"]:
            print(f"  位置/含义不一致: {r['position_mismatches']}")
        if r["percent_point_mix"]:
            print("  警告: 百分比/百分点写法与参考不一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())