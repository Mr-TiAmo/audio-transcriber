# 置信度阈值扫描：用 --dump-token-features 产出的逐字符 entropy，
# 在 bench_10min 上对齐 LiPS 参考标签，评估不同标亮策略的精确率/召回率。
#
# 只读脚本：不修改 DB / 产物，打印报告并写 JSON。
#
# 用法：
#   python scripts/confidence_threshold_sweep.py <stats.json> [输出.json]
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from confidence_build_dataset import conservative_norm, align_ops, apply_filler_rules  # noqa: E402

REF_PATH = Path(r"D:\work\workspace\audio-transcriber\data\bench\bench_10min_reference.txt")
OUT_DEFAULT = PROJECT_ROOT / "output" / "confidence" / "threshold_sweep.json"


def load_stats(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segs = data.get("segments") or []
    out = []
    for i, s in enumerate(segs):
        out.append({
            "seq": i,
            "text": s.get("text") or "",
            # char_entropies: [[orig_char_idx, entropy], ...]（仅信息字符）
            "ent": {int(k): float(v) for k, v in (s.get("char_entropies") or [])},
        })
    return out


def build_corpus(segments: list[dict]):
    """拼接各段归一化文本，返回：
    hyp_norm、segs（归一化区间/段内字符熵列表）、seg_of_char。
    """
    parts: list[str] = []
    ranges: list[tuple[int, int]] = []
    ents: list[list[float]] = []
    seg_of: list[int] = []
    cursor = 0
    for seg in segments:
        norm, idx_map = conservative_norm(seg["text"])
        # norm 索引 -> 原文索引
        seg_ent = [seg["ent"].get(orig) for orig in idx_map]
        parts.append(norm)
        ranges.append((cursor, cursor + len(norm)))
        ents.append(seg_ent)
        seg_of.extend([seg["seq"]] * len(norm))
        cursor += len(norm)
    return "".join(parts), ranges, ents, seg_of


def labels_for(ref_text: str, hyp_norm: str) -> list[int | None]:
    ref_norm, _ = conservative_norm(ref_text)
    ops = align_ops(ref_norm, hyp_norm)
    ops, _ = apply_filler_rules(ops, ref_norm, hyp_norm)
    labels: list[int | None] = [None] * len(hyp_norm)
    for op, _ri, hi in ops:
        if hi is None:
            continue
        labels[hi] = 1 if op == "C" else 0
    return labels


def score(selected: set[int], labels: list[int | None], valid: set[int]) -> dict:
    tp = fp = fn = tn = 0
    for i in valid:
        l = labels[i]
        if l is None:
            continue
        sel = i in selected
        if l == 0:
            if sel:
                tp += 1
            else:
                fn += 1
        else:
            if sel:
                fp += 1
            else:
                tn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": f1, "marked": tp + fp, "errors": tp + fn}


def auc_ap(pairs: list[tuple[float, int]]) -> tuple[float, float]:
    """pairs: (score, label)，label 1=错误（正例）。返回 (AUROC, AP)。"""
    if not pairs:
        return 0.0, 0.0
    pos = sum(1 for _, l in pairs if l == 1)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return 0.0, 0.0
    # AUROC：随机负例得分高于随机正例的概率（并列算 0.5）
    srt = sorted(pairs, key=lambda x: x[0])
    # 用平均秩法处理并列
    ranks: list[float] = [0.0] * len(srt)
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1][0] == srt[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum_pos = sum(ranks[k] for k, (_, l) in enumerate(srt) if l == 1)
    auc = (rank_sum_pos - pos * (pos + 1) / 2) / (pos * neg)
    # AP：按得分降序逐点计算
    desc = sorted(pairs, key=lambda x: -x[0])
    hits = 0
    ap = 0.0
    for k, (_, l) in enumerate(desc, 1):
        if l == 1:
            hits += 1
            ap += hits / k
    ap /= pos
    return auc, ap


def top_ratio_selection(values: list[tuple[int, float]], ratio: float) -> set[int]:
    """从 (index, entropy) 中取熵最高的 ratio 比例（至少 1 个）。"""
    if ratio <= 0 or not values:
        return set()
    n = max(1, int(len(values) * ratio))
    return {i for i, _ in sorted(values, key=lambda x: -x[1])[:n]}


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: confidence_threshold_sweep.py <stats.json> [out.json]", file=sys.stderr)
        sys.exit(2)
    stats_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_DEFAULT

    segments = load_stats(stats_path)
    ref_text = REF_PATH.read_text(encoding="utf-8")
    hyp_norm, ranges, ents, seg_of = build_corpus(segments)
    labels = labels_for(ref_text, hyp_norm)

    # 全量字符级 (norm 索引, 熵)，只保留有熵值且有标签的字符
    flat: list[tuple[int, float]] = []
    for (a, b), seg_ent in zip(ranges, ents):
        for off, ent in enumerate(seg_ent):
            if ent is not None:
                flat.append((a + off, ent))
    valid = {i for i, _ in flat}
    labelled = [i for i in valid if labels[i] is not None]
    n_err = sum(1 for i in labelled if labels[i] == 0)

    print("=" * 78)
    print("置信度阈值扫描 bench_10min")
    print("=" * 78)
    print(f"stats: {stats_path}")
    print(f"归一化字符 {len(hyp_norm)}；有熵值字符 {len(flat)}；有标签字符 {len(labelled)}；真实错误 {n_err}")
    ent_vals = sorted(e for _, e in flat)
    def q(p: float) -> float:
        return ent_vals[min(len(ent_vals) - 1, int(len(ent_vals) * p))]
    print("熵分布: " + "  ".join(f"P{int(p*100)}={q(p):.3f}" for p in
                                  (0, .01, .02, .03, .05, .08, .1, .2, .5, .8, .9, .95, .99, 1)))
    pairs = [(ent, 1 if labels[i] == 0 else 0) for i, ent in flat if labels[i] is not None]
    auc, ap = auc_ap(pairs)
    print(f"字符级 AUROC={auc:.4f}  AP={ap:.4f}")

    results: dict = {"stats": str(stats_path), "auc": auc, "ap": ap, "strategies": {}}

    def record(name: str, selected: set[int]) -> None:
        sc = score(selected, labels, valid)
        results["strategies"][name] = sc
        print(f"  {name:<34} P={sc['precision']:.3f} R={sc['recall']:.3f} "
              f"F1={sc['f1']:.3f}  标{sc['marked']:>4} 命中{sc['tp']:>3}")

    # ---- A. 全局 top-ratio ----
    print("\n--- A. 全局 top-ratio（跨段统一排名）---")
    for r in (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.15, 0.2):
        record(f"global-top-{r:.1%}", top_ratio_selection(flat, r))

    # ---- B. 段内 top-ratio（现状）----
    print("\n--- B. 段内 top-ratio（逐段各自排名，现状=10%）---")
    for r in (0.02, 0.03, 0.05, 0.08, 0.1, 0.15):
        sel: set[int] = set()
        for (a, _b), seg_ent in zip(ranges, ents):
            vals = [(a + off, e) for off, e in enumerate(seg_ent) if e is not None]
            sel |= top_ratio_selection(vals, r)
        record(f"per-seg-top-{r:.1%}", sel)

    # ---- C. 绝对熵阈值 ----
    print("\n--- C. 绝对熵阈值（全段统一）---")
    for t in (q(.90), q(.92), q(.94), q(.95), q(.96), q(.97), q(.98), q(.99)):
        sel = {i for i, e in flat if e >= t}
        record(f"abs-ent>={t:.3f}", sel)

    # ---- D. 段内 top-ratio + 绝对下限 ----
    print("\n--- D. 段内 top-ratio + 绝对熵下限 ---")
    for r in (0.05, 0.1, 0.2):
        for t in (q(.95), q(.97), q(.98)):
            sel = set()
            for (a, _b), seg_ent in zip(ranges, ents):
                vals = [(a + off, e) for off, e in enumerate(seg_ent) if e is not None]
                sel |= {i for i in top_ratio_selection(vals, r) if dict(vals)[i] >= t}
            record(f"seg-top-{r:.0%}+ent>={t:.3f}", sel)

    # ---- E. 段级门控：段内最高熵不够高则整段不标 ----
    print("\n--- E. 段级门控（段最大值 >= 全局阈值才标该段 top-ratio）---")
    seg_max = []
    for (a, _b), seg_ent in zip(ranges, ents):
        vals = [e for e in seg_ent if e is not None]
        seg_max.append(max(vals) if vals else 0.0)
    seq_max = sorted(seg_max)
    def qm(p: float) -> float:
        return seq_max[min(len(seq_max) - 1, int(len(seq_max) * p))]
    for gate_p in (0.5, 0.6, 0.7, 0.8):
        for r in (0.05, 0.1, 0.2):
            gate = qm(gate_p)
            sel = set()
            for (a, _b), seg_ent, mx in zip(ranges, ents, seg_max):
                if mx < gate:
                    continue
                vals = [(a + off, e) for off, e in enumerate(seg_ent) if e is not None]
                sel |= top_ratio_selection(vals, r)
            record(f"gate>={gate:.3f}+top-{r:.0%}", sel)

    # ---- 逐段现状诊断 ----
    print("\n--- 逐段诊断（现状 10%）---")
    cur = set()
    for (a, _b), seg_ent in zip(ranges, ents):
        vals = [(a + off, e) for off, e in enumerate(seg_ent) if e is not None]
        cur |= top_ratio_selection(vals, 0.1)
    for (a, b), seg_ent, seq in zip(ranges, ents, [s["seq"] for s in segments]):
        idxs = [i for i in range(a, b) if labels[i] is not None and i in valid]
        err = sum(1 for i in idxs if labels[i] == 0)
        mark = sum(1 for i in idxs if i in cur)
        hit = sum(1 for i in idxs if i in cur and labels[i] == 0)
        mx = max([e for e in seg_ent if e is not None], default=0.0)
        print(f"  段{seq:>2}: 字{len(idxs):>4} 错{err:>3} 标{mark:>3} 命中{hit:>3} 最大熵{mx:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入: {out_path}")


if __name__ == "__main__":
    main()