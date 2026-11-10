#!/usr/bin/env python3
"""中文 ASR 字错率（CER）评估脚本。

用法:
    python scripts/eval_cer.py 参考文本.txt 识别文本.txt [更多识别文本.txt ...]
    python scripts/eval_cer.py --raw 参考文本.txt 识别文本.txt   # 关闭数字/虚词归一化

默认归一化（公平对比不同模型的输出风格）：
1. 中文数字转阿拉伯数字（"四三八零" -> "4380"，"一千三百五十八" -> "1358"）
2. 去掉 "百分之"、"个百分点"
3. 去掉空白与全部中英文标点（保留数字间的小数点）
4. 去掉口语填充词（啊/嘛/哈/呢/吧/哦/呀/嗯/呃）

CER = 编辑距离 / 参考归一化后字符数。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from decimal import Decimal, localcontext

CN_DIGIT = {
    "零": 0, "〇": 0, "一": 1, "幺": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}

CN_NUM_RE = re.compile(r"[零〇一幺二两三四五六七八九十百千万亿]+(?:点[零〇一幺二两三四五六七八九]+)?")
# "万亿" 是复合尺度（"三万亿"=3 万 亿=30000 亿），需先于普通中文数字转换折叠
_CN_SCALE_RE = re.compile(r"([零〇一幺二两三四五六七八九十百千]+)万亿")
# 阿拉伯数字 + 中文尺度单位（"3万亿" 与 "30000亿" 数值等价，需折叠到同一写法）
_SCALE_RE = re.compile(r"(\d+(?:\.\d+)?)万亿")
_KEEP = re.compile(r"[^一-鿿㐀-䶿a-zA-Z0-9.]")
_FILL = re.compile(r"[啊嘛哈呢吧哦呀嗯呃]")


def cn2num(s: str) -> str:
    """中文数字串转阿拉伯数字串。"""
    if not s:
        return ""
    if "点" in s:
        a, b = s.split("点", 1)
        ip = cn2num(a) if a else "0"
        dp = "".join(str(CN_DIGIT[c]) for c in b if c in CN_DIGIT)
        return ip + "." + dp if dp else ip
    if any(c in CN_UNIT for c in s):
        total, section, num = 0, 0, 0
        for c in s:
            if c in CN_DIGIT:
                num = CN_DIGIT[c]
            elif c in ("十", "百", "千"):
                section += (num if num else 1) * CN_UNIT[c]
                num = 0
            else:  # 万 / 亿
                section = (section + num) * CN_UNIT[c]
                total += section
                section, num = 0, 0
        return str(total + section + num)
    return "".join(str(CN_DIGIT[c]) for c in s if c in CN_DIGIT)


def normalize(text: str, raw: bool = False, legacy_raw: bool = False) -> str:
    if raw:
        return text
    if legacy_raw:
        return _KEEP.sub("", text).lower()
    t = text.replace("百分之", "").replace("个百分点", "")
    t = _CN_SCALE_RE.sub(lambda m: f"{cn2num(m.group(1))}0000亿", t)

    def _cn_repl(m: re.Match[str]) -> str:
        s = m.group()
        # 纯单位串（"亿""万亿"）不参与转换，避免把 "30000亿" 变成 "300000"
        if not any(c in CN_DIGIT for c in s):
            return s
        return cn2num(s)

    t = CN_NUM_RE.sub(_cn_repl, t)
    # "3万亿" -> "30000亿"，与参考的写法对齐
    def _scale_repl(m):
        with localcontext() as ctx:
            ctx.prec = max(80, len(m.group(1)) + 10)
            number = format(Decimal(m.group(1)) * 10000, "f")
            if "." in number:
                number = number.rstrip("0").rstrip(".")
            return number + "亿"
    t = _SCALE_RE.sub(_scale_repl, t)
    t = _KEEP.sub("", t).lower()
    t = re.sub(r"(?<![0-9])\.|\.(?![0-9])", "", t)
    return _FILL.sub("", t)


def edit_stats(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """返回 (替换, 删除, 插入, 编辑距离)。"""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0, 0, m, m
    if m == 0:
        return 0, n, 0, n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
    sub = dele = ins = 0
    i, j = n, m
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            sub += 1
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    dele += i
    ins += j
    return sub, dele, ins, dp[n][m]


def main() -> int:
    ap = argparse.ArgumentParser()
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--raw", action="store_true", help="逐字符比较，包括符号、大小写和空白")
    modes.add_argument("--legacy-raw", action="store_true", help="兼容旧 raw 口径：删除标点空白并转小写")
    ap.add_argument("ref")
    ap.add_argument("hyps", nargs="+")
    args = ap.parse_args()

    ref = normalize(Path(args.ref).read_bytes().decode("utf-8"), args.raw, args.legacy_raw)
    mode = "逐字符原文" if args.raw else "旧 raw 口径" if args.legacy_raw else "数字/虚词归一化（非数字保真）"
    print(f"参考: {args.ref}  ({len(ref)} 字, {mode})")
    print("-" * 60)

    for hyp_arg in args.hyps:
        hyp_path = Path(hyp_arg)
        if not hyp_path.exists():
            print(f"跳过（不存在）: {hyp_path}")
            continue
        hyp = normalize(hyp_path.read_bytes().decode("utf-8"), args.raw, args.legacy_raw)
        sub, dele, ins, dist = edit_stats(ref, hyp)
        cer = dist / len(ref) if ref else float("nan")
        print(hyp_path.name)
        print(f"  字符数={len(hyp)}  S(替换)={sub}  D(删除)={dele}  I(插入)={ins}")
        print(f"  编辑距离={dist}  CER={cer * 100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())