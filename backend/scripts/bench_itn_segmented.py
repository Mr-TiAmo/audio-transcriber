#!/usr/bin/env python
"""基准：生产版 ITN「预筛 + 分段并行」的正确性与耗时。

对比 ``normalize_numbers``（整篇）与 ``normalize_numbers_segmented``
（生产 pipeline 使用），要求输出逐字符一致。

用法（backend 目录）::

    .venv/Scripts/python.exe scripts/bench_itn_segmented.py \
        ../data/temp/<run>/postprocess/result_raw.txt --workers 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local_asr.postprocess.itn import (  # noqa: E402
    normalize_numbers, normalize_numbers_segmented,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="ITN 分段并行基准")
    parser.add_argument("text")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=1200)
    parser.add_argument("--chunk-chars", type=int, default=800)
    args = parser.parse_args()

    text = Path(args.text).read_text(encoding="utf-8")
    started = time.perf_counter()
    whole = normalize_numbers(text)
    whole_s = time.perf_counter() - started

    started = time.perf_counter()
    segmented = normalize_numbers_segmented(
        text, workers=args.workers, min_chars=args.min_chars,
        chunk_chars=args.chunk_chars)
    segmented_s = time.perf_counter() - started

    print(f"字符数: {len(text)}")
    print(f"整篇:   {whole_s:.2f}s（编辑 {len(whole.edits)} 条）")
    print(f"分段:   {segmented_s:.2f}s（编辑 {len(segmented.edits)} 条，"
          f"加速 {whole_s / max(segmented_s, 1e-9):.1f}×）")
    print(f"文本一致: {whole.text == segmented.text}")
    return 0 if whole.text == segmented.text else 1


if __name__ == "__main__":
    sys.exit(main())