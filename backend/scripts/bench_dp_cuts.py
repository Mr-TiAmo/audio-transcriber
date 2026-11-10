#!/usr/bin/env python3
"""对比 silence / balanced / dp 三种切分策略的块长与静音吸附质量。

只跑 VAD + 切分，不加载 ASR 模型，用于快速验证长音频分段方案：

- 块数、块长 min/max/均值/标准差：块长越均匀，batch 内 padding 越少；
- 静音吸附率：中间切点落在 VAD 停顿上的比例（越高，边界回声越少）；
- 强制切分数：找不到可用静音、只能硬切的刀数。

用法::

    python scripts/bench_dp_cuts.py \
        --audio ../data/bench/random5m/random1.wav \
        --vad-model ../models/funasr-fsmn-vad --target 60
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND / "scripts"))
sys.path.insert(0, str(BACKEND / "src"))

from local_asr.asr.segmentation import (  # noqa: E402
    SegmentationConfig,
    VadSegment,
    plan_balanced_cuts,
    plan_cuts,
    plan_dp_cuts,
)
from local_asr.audio import audio_duration  # noqa: E402
from qwen3_asr_transcribe import detect_speech  # noqa: E402


def summarize(name: str, cuts, total_ms: int) -> dict:
    """汇总一种切分策略的块长与吸附指标。"""
    lengths = [c.end_ms - c.start_ms for c in cuts]
    inner = cuts[:-1]  # 最后一块是收尾，没有切点
    snapped = sum(1 for c in inner if not c.forced)
    return {
        "strategy": name,
        "chunks": len(cuts),
        "min_s": min(lengths) / 1000,
        "max_s": max(lengths) / 1000,
        "mean_s": statistics.fmean(lengths) / 1000,
        "std_s": statistics.pstdev(lengths) / 1000 if len(lengths) > 1 else 0.0,
        "snap_rate": snapped / len(inner) if inner else 1.0,
        "forced": len(inner) - snapped,
        "covered": cuts[-1].end_ms == total_ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--vad-model", type=Path,
                    default=BACKEND.parent / "models/funasr-fsmn-vad")
    ap.add_argument("--target", type=float, default=60.0, help="目标块长（秒）")
    ap.add_argument("--tolerance", type=float, default=0.15)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--dp-forced", default="0.3,0.6,1.2,2.0",
                    help="dp 硬切代价扫描值（逗号分隔）；越大越偏向块长均匀")
    ap.add_argument("--dp-gap", default="0.15", help="dp 静音奖励扫描值")
    ap.add_argument("--refresh-vad", action="store_true",
                    help="忽略 <audio>.vad.json 缓存，重新跑 VAD")
    args = ap.parse_args()

    cache = args.audio.with_suffix(".vad.json")
    if cache.is_file() and not args.refresh_vad:
        vad = [VadSegment(s, e) for s, e in json.loads(cache.read_text())]
        print(f"[dp-bench] 复用 VAD 缓存 {cache.name}（{len(vad)} 段）")
    else:
        vad = detect_speech(args.audio, args.vad_model)
        cache.write_text(json.dumps([[v.start_ms, v.end_ms] for v in vad]))
        print(f"[dp-bench] VAD {len(vad)} 段，已缓存 {cache.name}")
    total_ms = int(round(audio_duration(args.audio) * 1000))
    print(f"[dp-bench] {args.audio.name}: {total_ms / 1000:.1f}s，"
          f"VAD {len(vad)} 段，目标 {args.target:.0f}s")

    cfg = SegmentationConfig(
        target_chunk_seconds=args.target,
        chunk_strategy="dp",
        balance_tolerance=args.tolerance,
        balance_jitter_seconds=args.jitter,
    )
    rows = [
        summarize("silence", plan_cuts(vad, total_ms, cfg), total_ms),
        summarize("balanced", plan_balanced_cuts(vad, total_ms, cfg), total_ms),
    ]
    for forced in [float(x) for x in args.dp_forced.split(",") if x.strip()]:
        for bonus in [float(x) for x in args.dp_gap.split(",") if x.strip()]:
            dp_cfg = replace(cfg, dp_forced_penalty=forced, dp_gap_bonus=bonus)
            rows.append(summarize(
                f"dp(f={forced:g},b={bonus:g})",
                plan_dp_cuts(vad, total_ms, dp_cfg), total_ms))

    print("\n策略                块数  最短   最长   均值   标准差  吸附率  硬切  覆盖")
    for r in rows:
        print(f"{r['strategy']:<18} {r['chunks']:>4}  {r['min_s']:>5.1f}  "
              f"{r['max_s']:>5.1f}  {r['mean_s']:>5.1f}  {r['std_s']:>6.2f}  "
              f"{r['snap_rate']:>5.1%}  {r['forced']:>4}  {'是' if r['covered'] else '否'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())