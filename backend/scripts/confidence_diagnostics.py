# 置信度信号诊断：比较 entropy / neg_logprob / neg_margin，分析错误构成、
# 位置偏差与校准曲线，回答「为什么标亮的字大多是正确字」。
#
# 只读脚本，依赖 confidence_threshold_sweep 的语料构建/打分函数。
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from confidence_build_dataset import conservative_norm, align_ops, apply_filler_rules  # noqa: E402
from confidence_threshold_sweep import auc_ap, score, top_ratio_selection  # noqa: E402

REF_PATH = Path(r"D:\work\workspace\audio-transcriber\data\bench\bench_10min_reference.txt")
OUT_PATH = PROJECT_ROOT / "output" / "confidence" / "diagnostics.json"


def load_stats(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for i, s in enumerate(data.get("segments") or []):
        feats: dict[int, tuple[float, float, float]] = {}
        for row in s.get("char_features") or []:
            feats[int(row[0])] = (float(row[1]), float(row[2]), float(row[3]))
        if not feats:  # 兼容只有 char_entropies 的产物
            for row in s.get("char_entropies") or []:
                feats[int(row[0])] = (float(row[1]), 0.0, 0.0)
        out.append({"seq": i, "text": s.get("text") or "", "feats": feats})
    return out


def build(segments: list[dict]):
    parts: list[str] = []
    ranges: list[tuple[int, int]] = []
    seg_feats: list[list[tuple[float, float, float] | None]] = []
    seg_of: list[int] = []
    cursor = 0
    for seg in segments:
        norm, idx_map = conservative_norm(seg["text"])
        parts.append(norm)
        ranges.append((cursor, cursor + len(norm)))
        seg_feats.append([seg["feats"].get(orig) for orig in idx_map])
        seg_of.extend([seg["seq"]] * len(norm))
        cursor += len(norm)
    return "".join(parts), ranges, seg_feats, seg_of


def labels_and_ops(ref_text: str, hyp: str):
    ref_norm, _ = conservative_norm(ref_text)
    ops = align_ops(ref_norm, hyp)
    ops, _ = apply_filler_rules(ops, ref_norm, hyp)
    labels: list[int | None] = [None] * len(hyp)
    op_of: list[str | None] = [None] * len(hyp)
    for op, _ri, hi in ops:
        if hi is None:
            continue
        labels[hi] = 1 if op == "C" else 0
        op_of[hi] = op
    return labels, op_of, ops


def pct_rank(values: list[tuple[int, float]]) -> dict[int, float]:
    """把 (idx, value) 转成 [0,1] 百分位（并列取平均秩）。"""
    srt = sorted(values, key=lambda x: x[1])
    n = len(srt)
    out: dict[int, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and srt[j + 1][1] == srt[i][1]:
            j += 1
        avg = (i + j) / 2 / max(n - 1, 1)
        for k in range(i, j + 1):
            out[srt[k][0]] = avg
        i = j + 1
    return out


def describe(name: str, selected: set[int], labels, valid: set[int]) -> None:
    sc = score(selected, labels, valid)
    print(f"  {name:<40} P={sc['precision']:.3f} R={sc['recall']:.3f} "
          f"F1={sc['f1']:.3f} 标{sc['marked']:>4} 命中{sc['tp']:>3}")


def main() -> None:
    stats_path = Path(sys.argv[1])
    segments = load_stats(stats_path)
    ref_text = REF_PATH.read_text(encoding="utf-8")
    hyp, ranges, seg_feats, seg_of = build(segments)
    labels, op_of, ops = labels_and_ops(ref_text, hyp)

    valid = {i for i, f in enumerate(
        v for seg in seg_feats for v in seg) if f is not None}
    # 重新整理成扁平特征数组（下标 = norm 索引）
    flat: list[tuple[float, float, float] | None] = []
    for seg in seg_feats:
        flat.extend(seg)
    valid = {i for i, f in enumerate(flat) if f is not None}
    labelled = [i for i in sorted(valid) if labels[i] is not None]
    errs = [i for i in labelled if labels[i] == 0]

    print("=" * 84)
    print("置信度信号诊断")
    print("=" * 84)
    print(f"统计字符 {len(labelled)}，真实错误 {len(errs)} ({len(errs)/max(len(labelled),1):.2%})")

    # ---- 1. 错误构成：替换 S / 插入 I ----
    from collections import Counter
    cnt = Counter(op_of[i] for i in errs)
    print("\n--- 1. 错误构成（hyp 侧有字符的错误）---")
    print(f"  替换 S: {cnt.get('S', 0)}   插入 I: {cnt.get('I', 0)}   "
          f"（删除 D 无 hyp 字符，不计入标亮范围：{sum(1 for o,_,h in ops if o=='D')} 个）")
    for op in ("S", "I"):
        idxs = [i for i in errs if op_of[i] == op]
        if not idxs:
            continue
        ents = sorted(flat[i][0] for i in idxs)
        print(f"  {op} 的错误字熵: 中位 {ents[len(ents)//2]:.3f}  "
              f"P25 {ents[len(ents)//4]:.3f}  P75 {ents[3*len(ents)//4]:.3f}")

    # ---- 2. 单特征区分力 ----
    print("\n--- 2. 单特征字符级 AUROC / AP（正例=错误）---")
    results: dict = {"counts": {"chars": len(labelled), "errors": len(errs),
                                "S": cnt.get("S", 0), "I": cnt.get("I", 0)}}
    for k, name in enumerate(("entropy", "neg_logprob", "neg_margin")):
        pairs = [(flat[i][k], 1 if labels[i] == 0 else 0) for i in labelled]
        auc, ap = auc_ap(pairs)
        print(f"  {name:<14} AUROC={auc:.4f}  AP={ap:.4f}")
        results[name] = {"auc": auc, "ap": ap}

    # ---- 3. 特征融合（百分位平均）----
    print("\n--- 3. 特征融合（全局百分位平均，错误分数越高越可疑）---")
    ranks = []
    for k in range(3):
        ranks.append(pct_rank([(i, flat[i][k]) for i in valid]))
    fusion_pairs = [((ranks[0][i] + ranks[1][i] + ranks[2][i]) / 3,
                     1 if labels[i] == 0 else 0) for i in labelled]
    auc, ap = auc_ap(fusion_pairs)
    print(f"  fusion(ent+nlp+nmg)  AUROC={auc:.4f}  AP={ap:.4f}")
    results["fusion"] = {"auc": auc, "ap": ap}
    fusion_vals = [(i, (ranks[0][i] + ranks[1][i] + ranks[2][i]) / 3) for i in valid]
    for r in (0.01, 0.02, 0.03, 0.05, 0.1):
        describe(f"fusion global-top-{r:.0%}",
                 top_ratio_selection(fusion_vals, r), labels, valid)
        results.setdefault("fusion_strategies", {})[f"top-{r}"] = score(
            top_ratio_selection(fusion_vals, r), labels, valid)

    # ---- 4. 位置偏差：段内相对位置 vs 熵/错误率 ----
    print("\n--- 4. 位置偏差（段内相对位置十分位）---")
    print("  段内位置   平均熵    错误率    标亮精确率(现状 top10%)")
    # 计算现状 top10% 选择
    cur: set[int] = set()
    for (a, _b), seg_f in zip(ranges, seg_feats):
        vals = [(a + off, f[0]) for off, f in enumerate(seg_f) if f is not None]
        cur |= top_ratio_selection(vals, 0.1)
    for d in range(10):
        ents, err, mark, hit = [], 0, 0, 0
        for (a, b), seg_f in zip(ranges, seg_feats):
            L = b - a
            lo, hi = a + L * d // 10, a + L * (d + 1) // 10
            for i in range(lo, hi):
                if flat[i] is None or labels[i] is None:
                    continue
                ents.append(flat[i][0])
                if labels[i] == 0:
                    err += 1
                if i in cur:
                    mark += 1
                    if labels[i] == 0:
                        hit += 1
        n = max(len(ents), 1)
        p = hit / mark if mark else 0.0
        print(f"  {d*10}-{d*10+10}%   {sum(ents)/n:7.3f}   {err/n:6.2%}   "
              f"{p:6.1%} ({hit}/{mark})")

    # ---- 5. 熵分桶校准 ----
    print("\n--- 5. 熵分桶校准（全局百分位）---")
    ent_rank = ranks[0]
    print("  熵区间(百分位)   字数    错误率    占全部错误")
    for lo in (0, 50, 80, 90, 95, 97, 99):
        hi = {0: 50, 50: 80, 80: 90, 90: 95, 95: 97, 97: 99, 99: 100}[lo]
        idxs = [i for i in labelled if lo <= ent_rank[i] * 100 < hi or (hi == 100 and ent_rank[i] * 100 >= 99)]
        if not idxs:
            continue
        e = sum(1 for i in idxs if labels[i] == 0)
        share = e / max(len(errs), 1)
        print(f"  P{lo}-P{hi}        {len(idxs):>5}   {e/len(idxs):6.2%}   {share:6.1%}")

    # ---- 6. 漏报分析：错误字里熵排名分布 ----
    print("\n--- 6. 漏报分析（错误字的熵百分位分布）---")
    buckets = Counter()
    for i in errs:
        r = ent_rank[i] * 100
        buckets["top1%" if r >= 99 else "top5%" if r >= 95 else
                "top10%" if r >= 90 else "top20%" if r >= 80 else "top50%" if r >= 50 else "bottom50%"] += 1
    for name in ("top1%", "top5%", "top10%", "top20%", "top50%", "bottom50%"):
        print(f"  {name:<9}: {buckets.get(name, 0):>3} 个错误 ({buckets.get(name,0)/max(len(errs),1):.1%})")
    results["error_entropy_rank"] = dict(buckets)

    # ---- 7. 段级信号 ----
    print("\n--- 7. 段级：平均熵/最大熵 vs 段错误率 ---")
    for (a, b), seg_f, seg in zip(ranges, seg_feats, segments):
        vals = [f[0] for f in seg_f if f is not None]
        idxs = [i for i in range(a, b) if labels[i] is not None and flat[i] is not None]
        e = sum(1 for i in idxs if labels[i] == 0)
        print(f"  段{seg['seq']:>2}: 字{len(idxs):>4} 错{e:>3} 错误率{e/max(len(idxs),1):6.2%} "
              f"均值{sum(vals)/max(len(vals),1):.3f} 最大{max(vals) if vals else 0:.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入: {OUT_PATH}")


if __name__ == "__main__":
    main()