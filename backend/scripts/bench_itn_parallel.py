"""ITN（wetext 数字规范化）分段并行 / 预筛评估实验。

回答三个问题：
1. 按句切分后，有多少句子真的需要 ITN（含数字候选）？
2. 「整篇 normalize」与「按句 normalize 后拼接」结果是否一致？
3. 多进程分段并行能把 60min 文本的 ITN 从多久降到多久？

用法：
    python -I backend/scripts/bench_itn_parallel.py data/temp/<run>/postprocess/result_raw.txt \
        [--workers 4] [--chunk-chars 800]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local_asr.postprocess.itn import normalize_numbers  # noqa: E402

# 句末切分：每个句子（含标点）保持原文顺序，拼接后必须等于原文
_SENT_RE = re.compile(r"[^。！？!?\n]*(?:[。！？!?]+|\n|$)")
# 需要 ITN 的候选：连续中文数字串、阿拉伯数字、百分号/单位数字
_NUM_CANDIDATE_RE = re.compile(
    r"[零〇一二两三四五六七八九十百千万亿]{2,}"  # 连续数词（「一下」不命中）
    r"|\d+"                                     # 阿拉伯数字
    r"|百分之"                                  # 百分比
    r"|(?<![一二三四五六七八九十])([两三四五六七八九十]+)(?=个|年|月|日|元|块|万|亿|千)"
)


def split_sentences(text: str) -> list[str]:
    return [m.group(0) for m in _SENT_RE.finditer(text) if m.group(0)]


def needs_itn(sentence: str) -> bool:
    return bool(_NUM_CANDIDATE_RE.search(sentence))


def pack(sentences: list[str], limit: int) -> list[str]:
    """只把原文中连续的命中句合并成块（跳过未命中句会改变上下文，禁止跨句拼接）。"""
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for s in sentences:
        if not needs_itn(s):
            if buf:
                out.append("".join(buf))
                buf, size = [], 0
            continue
        if size + len(s) > limit and buf:
            out.append("".join(buf))
            buf, size = [], 0
        buf.append(s)
        size += len(s)
    if buf:
        out.append("".join(buf))
    return out


def _normalize_one(text: str) -> str:
    return normalize_numbers(text).text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk-chars", type=int, default=800)
    args = ap.parse_args()

    raw = args.text.read_text(encoding="utf-8")
    sentences = split_sentences(raw)
    hit = [s for s in sentences if needs_itn(s)]
    print(f"原文 {len(raw)} 字 / {len(sentences)} 句；含数字候选 {len(hit)} 句 "
          f"（{sum(len(s) for s in hit)} 字，"
          f"{sum(len(s) for s in hit) / max(len(raw), 1):.1%} 字符量）")

    t = time.perf_counter()
    whole = normalize_numbers(raw).text
    whole_s = time.perf_counter() - t
    print(f"整篇 normalize: {whole_s:.2f}s")

    # 只对命中句跑 ITN，其余原样保留；同一连续区间内按原文顺序拼接
    chunks = pack(sentences, args.chunk_chars)
    t = time.perf_counter()
    serial = [_normalize_one(c) for c in chunks]
    serial_s = time.perf_counter() - t
    print(f"命中句串行（{len(chunks)} 块）: {serial_s:.2f}s")

    t = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        parallel = pool.map(_normalize_one, chunks)
    parallel_s = time.perf_counter() - t
    print(f"{args.workers} 进程并行: {parallel_s:.2f}s（含进程启动）")

    # 文本一致性：分段结果只覆盖命中句，未命中句保持原文。
    # 用「按块替换」重建全文，与整篇结果对比差异率。
    rebuilt_parts, cursor = [], 0
    for i, c in enumerate(chunks):
        pos = raw.find(c, cursor)
        if pos < 0:
            print("  警告: 块无法在原文中定位，跳过一致性检查", file=sys.stderr)
            rebuilt_parts = []
            break
        rebuilt_parts.append(raw[cursor:pos])
        rebuilt_parts.append(serial[i])
        cursor = pos + len(c)
    if rebuilt_parts:
        rebuilt = "".join(rebuilt_parts) + raw[cursor:]
        import difflib
        same = sum(i2 - i1 for op, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, whole, rebuilt, autojunk=False).get_opcodes() if op == "equal")
        print(f"整篇 vs 分段（仅命中块）字符级一致率: {same / max(len(whole), 1):.2%}；"
              f"长度 {len(whole)} vs {len(rebuilt)}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())