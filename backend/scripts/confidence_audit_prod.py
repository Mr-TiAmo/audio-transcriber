#!/usr/bin/env python3
"""对账生产 ASR stats 的低置信标亮：字符级 P/R/AUROC/AP 与段级复核排序。

输入是 ``qwen3_asr_transcribe.py`` 产出的 ``*.stats.json``（``low_conf_spans``
已由全局预算截断）。参考文本按 ``confidence_build_dataset.py`` 的保守归一化
+ 语气词规则打标签，口径与 ``confidence_p1p2_bench.py`` 一致。

支持一次传入多个 (stats, reference) 做汇总；若 stats 由 ``--dump-token-features``
产出（含 ``char_signals``），额外按 1%/2%/4%/8%/10% 覆盖率重放生产策略。

用法::

    python scripts/confidence_audit_prod.py \
        --stats ../data/bench/random5m/run_r1/raw.stats.json \
                ../data/bench/random5m/run_r2c/raw.stats.json \
        --reference ../data/bench/random5m/random1.txt \
                    ../data/bench/random5m/random2.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND / "scripts"))
sys.path.insert(0, str(BACKEND / "src"))

from confidence_build_dataset import (  # noqa: E402
    align_ops,
    apply_filler_rules,
    conservative_norm,
)
from local_asr.asr.confidence import (  # noqa: E402
    CharSignal,
    SegmentSignals,
    select_low_conf,
)
from local_asr.asr.segmentation import keep_char  # noqa: E402
from app.services.segment_service import build_segments_from_stats  # noqa: E402

DEFAULT_RATIOS = "0.01,0.02,0.04,0.08,0.10"


def label_texts(texts: list[str], ref_text: str) -> list[list[int | None]]:
    """全局拼接对齐，返回每段每个原文码点的标签：1=错误，0=正确，None=非信息。"""
    parts, mapping = [], []
    for si, text in enumerate(texts):
        norm, idx_map = conservative_norm(text)
        mapping.extend((si, orig) for orig in idx_map)
        parts.append(norm)
    hyp = "".join(parts)
    ref_norm, _ = conservative_norm(ref_text)
    ops, _ = apply_filler_rules(align_ops(ref_norm, hyp), ref_norm, hyp)
    labels: dict[tuple[int, int], int] = {}
    for op, _ri, hi in ops:
        if hi is not None:
            # 与 bench 的 label 口径相反：这里直接用 1 表示错误，便于统计
            labels[mapping[hi]] = 0 if op == "C" else 1
    return [[labels.get((si, i)) for i in range(len(t))] for si, t in enumerate(texts)]


def auroc(pos: list[float], neg: list[float]) -> float:
    """错误字符（正类）分数高于正确字符的概率；并列记 0.5。"""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def average_precision(scored: list[tuple[float, int]]) -> float:
    """按分数降序的 PR 曲线面积（sklearn 口径，阶梯插值）。"""
    if not scored:
        return float("nan")
    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    total_pos = sum(1 for _, y in scored if y == 1)
    if not total_pos:
        return float("nan")
    tp = fp = 0
    prev_recall = ap = 0.0
    for _score, y in scored:
        if y == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def _signals_from_stats(stats: dict) -> list[SegmentSignals]:
    """用 stats 中 dump 的 char_signals 还原每段信号（原文本索引）。"""
    segs: list[SegmentSignals] = []
    for i, seg in enumerate(stats.get("segments", [])):
        text = seg.get("text", "")
        chars: list[CharSignal | None] = [None] * len(text)
        for row in seg.get("char_signals") or []:
            idx, ent, nlp, nmg, tlen, single = row[:6]
            extra = row[6:10] + [None] * (4 - len(row[6:10]))
            agree, abs_z, short, long_ = extra
            if 0 <= idx < len(text):
                chars[idx] = CharSignal(
                    index=int(idx), entropy=float(ent), neg_logprob=float(nlp),
                    neg_margin=float(nmg), token_len=int(tlen),
                    single_token=bool(single),
                    sample_agree=(float(agree) if agree is not None else None),
                    align_abs_z=float(abs_z or 0.0),
                    align_short=bool(short), align_long=bool(long_))
        sg = SegmentSignals(seq=i, text=text, chars=chars)
        singles = [c.entropy for c in chars if c is not None and c.single_token]
        sg.max_single_entropy = max(singles or [0.0])
        sg.align_anomaly_count = sum(
            1 for c in chars if c is not None and (c.align_short or c.align_long))
        segs.append(sg)
    return segs


def _sweep(stats: dict, labels: list[list[int | None]], ratios: list[float]) -> list[dict]:
    """在原始 stats 文本空间重放生产策略，输出各覆盖率下的 P/R。"""
    if not any(seg.get("char_signals") for seg in stats.get("segments", [])):
        return []
    segs = _signals_from_stats(stats)
    rows = []
    for ratio in ratios:
        marked = select_low_conf(segs, max_ratio=ratio)
        tp = total_err = total_info = 0
        for sg, labs in zip(segs, labels):
            pos = {i for s, e in sg.spans for i in range(s, e)}
            err = {i for i, v in enumerate(labs) if v == 1}
            tp += len(pos & err)
            total_err += len(err)
            total_info += sum(1 for v in labs if v is not None)
        rows.append({
            "ratio": ratio,
            "marked": marked,
            "coverage": round(marked / max(total_info, 1), 4),
            "tp": tp,
            "err": total_err,
            "precision": round(tp / marked, 4) if marked else None,
            "recall": round(tp / total_err, 4) if total_err else None,
        })
    return rows


def audit_one(stats_path: Path, ref_path: Path) -> dict:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    ref_text = ref_path.read_text(encoding="utf-8")

    # 默认档：用入库文本（去重 + strip 后）对账；按识别块口径统计，
    # 不合并展示段（合并只影响页面展示粒度，不改变字符级标亮）。
    segs = build_segments_from_stats(stats_path, repack=False)
    texts = [s.text for s in segs]
    labels = label_texts(texts, ref_text)

    marked = tp = total_err = total_info = 0
    per_seg = []
    for seg, labs in zip(segs, labels):
        pos = set()
        for a, b in seg.low_conf_spans or []:
            pos.update(range(a, min(b, len(labs))))
        err = {i for i, v in enumerate(labs) if v == 1}
        info = sum(1 for v in labs if v is not None)
        seg_tp = len(pos & err)
        marked += len(pos)
        tp += seg_tp
        total_err += len(err)
        total_info += info
        per_seg.append({"seq": seg.sequence_no, "info": info, "err": len(err),
                        "marked": len(pos), "tp": seg_tp, "flags": seg.risk_flags})

    # entropy 排序质量（全体信息字符，不依赖预算）
    ent_scored = []
    for chunk, labs in zip(stats.get("segments", []), labels):
        ent = dict(chunk.get("char_entropies") or [])
        for i, lab in enumerate(labs):
            if lab is not None and i in ent:
                ent_scored.append((float(ent[i]), lab))
    pos_scores = [s for s, y in ent_scored if y == 1]
    neg_scores = [s for s, y in ent_scored if y == 0]

    err_segs = {s["seq"] for s in per_seg if s["err"]}
    order = [int(x) for x in stats.get("review_order") or []]
    hits, seen = [], 0
    for k, seq in enumerate(order, 1):
        seen += 1 if seq in err_segs else 0
        hits.append((k, seen))

    return {
        "stats": str(stats_path),
        "reference": str(ref_path),
        "segments": len(segs),
        "info_chars": total_info,
        "error_chars": total_err,
        "marked_chars": marked,
        "coverage": round(marked / max(total_info, 1), 4),
        "tp": tp,
        "precision": round(tp / marked, 4) if marked else None,
        "recall": round(tp / total_err, 4) if total_err else None,
        "entropy_auroc": (round(auroc(pos_scores, neg_scores), 4)
                          if pos_scores and neg_scores else None),
        "entropy_ap": (round(average_precision(ent_scored), 4)
                       if ent_scored else None),
        "segments_with_error": len(err_segs),
        "review_hits_at": hits,
        "per_segment": per_seg,
        # 内部字段：供汇总，不写盘
        "_ent_scored": ent_scored,
    }


def aggregate(reports: list[dict]) -> dict:
    info = sum(r["info_chars"] for r in reports)
    err = sum(r["error_chars"] for r in reports)
    marked = sum(r["marked_chars"] for r in reports)
    tp = sum(r["tp"] for r in reports)
    scored = [x for r in reports for x in r["_ent_scored"]]
    pos = [s for s, y in scored if y == 1]
    neg = [s for s, y in scored if y == 0]
    return {
        "files": len(reports),
        "segments": sum(r["segments"] for r in reports),
        "info_chars": info,
        "error_chars": err,
        "marked_chars": marked,
        "coverage": round(marked / max(info, 1), 4),
        "tp": tp,
        "precision": round(tp / marked, 4) if marked else None,
        "recall": round(tp / err, 4) if err else None,
        "entropy_auroc": round(auroc(pos, neg), 4) if pos and neg else None,
        "entropy_ap": round(average_precision(scored), 4) if scored else None,
        "segments_with_error": sum(r["segments_with_error"] for r in reports),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, nargs="+", required=True)
    ap.add_argument("--reference", type=Path, nargs="+", required=True)
    ap.add_argument("--ratios", default=DEFAULT_RATIOS,
                    help="覆盖率扫描（逗号分隔），需 stats 含 char_signals")
    args = ap.parse_args()
    if len(args.stats) != len(args.reference):
        ap.error("--stats 与 --reference 数量必须一致")

    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    reports, sweeps = [], []
    for stats_path, ref_path in zip(args.stats, args.reference):
        report = audit_one(stats_path, ref_path)
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        sweep_labels = label_texts([s.get("text", "") for s in stats.get("segments", [])],
                                   ref_path.read_text(encoding="utf-8"))
        sweep = _sweep(stats, sweep_labels, ratios)
        report["sweep"] = sweep
        reports.append(report)
        sweeps.append(sweep)

    summary = aggregate(reports)
    for r in reports:
        r.pop("_ent_scored", None)

    out = {"per_file": reports, "aggregate": summary}
    # 覆盖率曲线的跨文件合并（按 ratio 汇总 marked/tp/err/info）
    if any(sweeps):
        merged = []
        for k, ratio in enumerate(ratios):
            rows = [sw[k] for sw in sweeps if len(sw) > k]
            if not rows:
                continue
            m = sum(x["marked"] for x in rows)
            t = sum(x["tp"] for x in rows)
            e = sum(x["err"] for x in rows) if "err" in rows[0] else None
            merged.append({"ratio": ratio, "marked": m, "tp": t,
                           "precision": round(t / m, 4) if m else None,
                           "recall": round(t / e, 4) if e else None})
        out["sweep_merged"] = merged

    print(json.dumps(out["aggregate"], ensure_ascii=False, indent=2))
    if out.get("sweep_merged"):
        print("\n覆盖率扫描（跨文件合并）：")
        print(" ratio  marked   TP   P       R")
        for row in out["sweep_merged"]:
            print(f" {row['ratio']:.0%}   {row['marked']:4d}  {row['tp']:3d}  "
                  f"{row['precision'] if row['precision'] is not None else '-':>6}  "
                  f"{row['recall'] if row['recall'] is not None else '-':>6}")

    out_path = args.stats[0].with_name("confidence_audit.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[audit] 写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())