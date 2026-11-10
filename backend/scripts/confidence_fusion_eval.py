#!/usr/bin/env python3
"""多信号融合评估：entropy × 采样一致性 × 强制对齐。

输入 stats 需由 ``qwen3_asr_transcribe.py --dump-token-features`` 产出，
且带 ``--sample-check N --forced-aligner PATH`` 时 ``char_signals`` 才含
采样/对齐列。标签口径与 ``confidence_audit_prod.py`` 相同（LiPS 参考全局对齐）。

输出：单特征 AUROC/AP + 各融合策略在不同覆盖率下的 P/R。

用法::

    python scripts/confidence_fusion_eval.py \
        --stats ../data/bench/random5m/run_f1/raw.stats.json \
                ../data/bench/random5m/run_f2/raw.stats.json \
        --reference ../data/bench/random5m/random1.txt \
                    ../data/bench/random5m/random2.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND / "scripts"))
sys.path.insert(0, str(BACKEND / "src"))

from confidence_audit_prod import (  # noqa: E402
    _signals_from_stats,
    auroc,
    average_precision,
    label_texts,
)

FLOOR = 0.1
GATE = 0.7


def _score_fns() -> dict[str, callable]:
    def align_boost(c) -> float:
        return (0.35 * min(c.align_abs_z, 4.0)
                + 0.3 * int(c.align_short) + 0.3 * int(c.align_long))

    def agree_boost(c) -> float:
        return 0.8 * (1.0 - c.sample_agree) if c.sample_agree is not None else 0.0

    return {
        "entropy": lambda c: c.entropy,
        # 生产 recall 档
        "entropy+align": lambda c: c.entropy + align_boost(c),
        # 生产 quality 档
        "entropy+agree": lambda c: c.entropy + agree_boost(c),
        # 融合：对齐权重降低，避免过度依赖对齐噪声
        "entropy+0.5align+0.5agree": lambda c: (
            c.entropy + 0.5 * align_boost(c) + 0.5 * agree_boost(c)),
        # 只用附加信号（诊断用）
        "align_only": lambda c: align_boost(c),
    }


def _pick(segs, labels, score_fn, ratio: float, *, gate: bool = True,
          require_single: bool = True) -> tuple[int, int, int, int]:
    """返回 (marked, tp, total_err, total_info)。"""
    total_info = sum(1 for sg, labs in zip(segs, labels)
                     for v in labs if v is not None)
    budget = max(1, int(math.ceil(total_info * ratio)))
    cands = []
    for sg, labs in zip(segs, labels):
        if gate:
            singles = [c.entropy for c in sg.chars if c is not None and c.single_token]
            if max(singles or [0.0]) < GATE and sg.align_anomaly_count < 2:
                continue
        for c in sg.chars:
            if c is None or (require_single and not c.single_token):
                continue
            score = score_fn(c)
            if score < FLOOR:
                continue
            cands.append((score, sg.seq, c.index))
    cands.sort(key=lambda x: x[0], reverse=True)
    picked = cands[:budget]
    tp = 0
    for _score, seq, idx in picked:
        labs = labels[seq]
        if 0 <= idx < len(labs) and labs[idx] == 1:
            tp += 1
    total_err = sum(1 for labs in labels for v in labs if v == 1)
    return len(picked), tp, total_err, total_info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, nargs="+", required=True)
    ap.add_argument("--reference", type=Path, nargs="+", required=True)
    ap.add_argument("--ratios", default="0.01,0.02,0.04,0.08")
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    all_segs, all_labels = [], []
    for stats_path, ref_path in zip(args.stats, args.reference):
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        texts = [s.get("text", "") for s in stats.get("segments", [])]
        labels = label_texts(texts, ref_path.read_text(encoding="utf-8"))
        segs = _signals_from_stats(stats)
        for sg in segs:
            sg.seq = len(all_segs)
            all_segs.append(sg)
        all_labels.extend(labels)

    # 单特征 AUROC/AP（仅单字 token 候选）
    scores = {
        "entropy": lambda c: c.entropy,
        "align_abs_z": lambda c: c.align_abs_z,
        "align_short": lambda c: float(c.align_short),
        "sample_disagree": lambda c: (1.0 - c.sample_agree
                                      if c.sample_agree is not None else 0.0),
        "token_len": lambda c: float(c.token_len),
    }
    single_rows = []
    for name, fn in scores.items():
        scored = []
        for sg, labs in zip(all_segs, all_labels):
            for c in sg.chars:
                if c is None or not c.single_token:
                    continue
                if 0 <= c.index < len(labs) and labs[c.index] is not None:
                    scored.append((fn(c), labs[c.index]))
        pos = [s for s, y in scored if y == 1]
        neg = [s for s, y in scored if y == 0]
        single_rows.append({
            "feature": name,
            "n": len(scored),
            "AUROC": round(auroc(pos, neg), 4) if pos and neg else None,
            "AP": round(average_precision(scored), 4) if scored else None,
        })

    fns = _score_fns()
    table = []
    for ratio in ratios:
        for name, fn in fns.items():
            marked, tp, err, info = _pick(all_segs, all_labels, fn, ratio)
            table.append({
                "ratio": ratio, "strategy": name, "marked": marked,
                "tp": tp, "err": err,
                "coverage": round(marked / max(info, 1), 4),
                "precision": round(tp / marked, 4) if marked else None,
                "recall": round(tp / err, 4) if err else None,
            })

    print("单特征（单字 token 候选）：")
    print(" feature            n    AUROC    AP")
    for r in single_rows:
        print(f" {r['feature']:<16} {r['n']:4d}  {r['AUROC']}  {r['AP']}")
    print("\n融合策略：")
    print(" strategy                     ratio  marked  TP   P       R")
    for r in table:
        print(f" {r['strategy']:<26} {r['ratio']:.0%}   {r['marked']:4d}  {r['tp']:3d}  "
              f"{r['precision'] if r['precision'] is not None else '-':>6}  "
              f"{r['recall'] if r['recall'] is not None else '-':>6}")

    out = {"single_features": single_rows, "strategies": table}
    out_path = args.stats[0].with_name("confidence_fusion.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[fusion] 写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())