# 置信度信号实验：在 bench_10min 上比较「原始熵 / 滑动平滑 / 段内相对化 /
# 段尾加成」等派生分数的区分力，并给出绝对阈值+段级门控的网格。
#
# 只读脚本。用于选定生产策略，避免直接改动 ASR 脚本反复重跑。
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from confidence_build_dataset import conservative_norm  # noqa: E402
from confidence_diagnostics import build, labels_and_ops, load_stats  # noqa: E402
from confidence_threshold_sweep import auc_ap, score, top_ratio_selection  # noqa: E402

REF_PATH = Path(r"D:\work\workspace\audio-transcriber\data\bench\bench_10min_reference.txt")


def smooth(values: list[float], win: int, mode: str = "mean") -> list[float]:
    """对字符级序列做滑动聚合（窗口半径为 win）。"""
    out = []
    n = len(values)
    for i in range(n):
        seg = values[max(0, i - win):min(n, i + win + 1)]
        if mode == "max":
            out.append(max(seg))
        else:
            out.append(sum(seg) / len(seg))
    return out


def main() -> None:
    stats_path = Path(sys.argv[1])
    segments = load_stats(stats_path)
    ref_text = REF_PATH.read_text(encoding="utf-8")
    hyp, ranges, seg_feats, _seg_of = build(segments)
    labels, op_of, _ops = labels_and_ops(ref_text, hyp)

    flat = []
    for seg in seg_feats:
        flat.extend(seg)
    valid = {i for i, f in enumerate(flat) if f is not None}
    labelled = [i for i in sorted(valid) if labels[i] is not None]
    errs = [i for i in labelled if labels[i] == 0]
    print(f"字符 {len(labelled)} 错误 {len(errs)} ({len(errs)/len(labelled):.2%})")

    # ---------- 构造每段分值 ----------
    def per_seg_scores(fn):
        """fn(seg_index, values_in_seg, global_positions) -> list[float]"""
        out: dict[int, list[float]] = {}
        for si, ((a, b), seg) in enumerate(zip(ranges, seg_feats)):
            vals = [f[0] if f is not None else 0.0 for f in seg]
            out[si] = fn(si, vals, a)
        return out

    def flatten(si_scores) -> list[float | None]:
        out: list[float | None] = [None] * len(hyp)
        for si, ((a, b), _seg) in enumerate(zip(ranges, seg_feats)):
            out[a:b] = si_scores[si]
        return out

    def report(name: str, score_arr: list[float | None], top_ratios=(0.02, 0.05, 0.1)) -> None:
        pairs = [(score_arr[i], 1 if labels[i] == 0 else 0) for i in labelled]
        auc, ap = auc_ap(pairs)
        vals = [(i, score_arr[i]) for i in valid]
        line = f"{name:<26} AUROC={auc:.4f} AP={ap:.4f}"
        for r in top_ratios:
            sc = score(top_ratio_selection(vals, r), labels, valid)
            line += f" | top{r:.0%} P={sc['precision']:.3f} R={sc['recall']:.3f}"
        print(line)
        return auc, ap

    # 1. 原始熵
    base = flatten({si: [f[0] if f is not None else 0.0 for f in seg]
                    for si, seg in enumerate(seg_feats)})
    print("\n--- 1. 原始熵 vs 平滑 ---")
    report("raw entropy", base)
    for win in (1, 2, 3):
        for mode in ("mean", "max"):
            s = flatten({si: smooth([f[0] if f is not None else 0.0 for f in seg], win, mode)
                         for si, seg in enumerate(seg_feats)})
            report(f"smooth-{mode}-win{win}", s)

    # 2. 段内相对化
    print("\n--- 2. 段内相对化 ---")
    def relativize(vals, how):
        if not vals:
            return vals
        srt = sorted(vals)
        med = srt[len(srt) // 2]
        if how == "diff":
            return [v - med for v in vals]
        if how == "ratio":
            mx = max(vals) or 1.0
            return [v / mx for v in vals]
        if how == "rank":
            order = sorted(range(len(vals)), key=lambda i: vals[i])
            rk = [0.0] * len(vals)
            for r, i in enumerate(order):
                rk[i] = r / max(len(vals) - 1, 1)
            return rk
        return vals
    for how in ("diff", "ratio", "rank"):
        s = flatten({si: relativize([f[0] if f is not None else 0.0 for f in seg], how)
                     for si, seg in enumerate(seg_feats)})
        report(f"seg-{how}", s)

    # 3. 段尾加成
    print("\n--- 3. 段尾加成（段内末 10% 字符 × 系数）---")
    for alpha in (0.2, 0.5, 1.0, 2.0):
        def tail_boost(vals):
            n = len(vals)
            cut = int(n * 0.9)
            return [v * (1 + alpha) if i >= cut else v for i, v in enumerate(vals)]
        s = flatten({si: tail_boost([f[0] if f is not None else 0.0 for f in seg])
                     for si, seg in enumerate(seg_feats)})
        report(f"tail-boost-{alpha}", s)

    # 4. 绝对阈值 + 段级门控网格
    print("\n--- 4. 绝对阈值 + 段门控网格（P/R/F1）---")
    seg_max = [max([f[0] for f in seg if f is not None], default=0.0) for seg in seg_feats]
    print("  ent>=", "  ".join(f"{t:.2f}" for t in (0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.9)))
    for t in (0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.9):
        for gate in (0.0, 0.7, 0.9, 1.2):
            sel = set()
            for (a, b), seg, mx in zip(ranges, seg_feats, seg_max):
                if mx < gate:
                    continue
                for off, f in enumerate(seg):
                    if f is not None and f[0] >= t:
                        sel.add(a + off)
            sc = score(sel, labels, valid)
            print(f"  ent>={t:.2f} gate>={gate:.1f}  P={sc['precision']:.3f} "
                  f"R={sc['recall']:.3f} F1={sc['f1']:.3f} 标{sc['marked']:>4} 命中{sc['tp']:>3}")

    # 5. token 内位置
    print("\n--- 5. token 内位置（同熵连续字符视为同一 token）---")
    pos_cnt = {"first": [0, 0], "mid": [0, 0], "last": [0, 0], "single": [0, 0]}
    for (a, b), seg in zip(ranges, seg_feats):
        i = 0
        while i < len(seg):
            j = i
            while j + 1 < len(seg) and seg[j + 1] is not None and seg[j] is not None \
                    and abs(seg[j + 1][0] - seg[j][0]) < 1e-9:
                j += 1
            span = range(i, j + 1)
            for k in span:
                if seg[k] is None or labels[a + k] is None:
                    continue
                key = ("single" if len(span) == 1 else
                       "first" if k == i else "last" if k == j else "mid")
                pos_cnt[key][0] += 1
                if labels[a + k] == 0:
                    pos_cnt[key][1] += 1
            i = j + 1
    for k, (n, e) in pos_cnt.items():
        print(f"  {k:<7}: 字{n:>4} 错{e:>3} 错误率{e/max(n,1):6.2%}")

    # ---------- 6. token 长度作为置信特征 ----------
    # 同 token 的字符共享同一熵值，据此切出 token 边界并记录 token 长度。
    token_len: list[int] = [0] * len(hyp)
    for (a, _b), seg in zip(ranges, seg_feats):
        i = 0
        while i < len(seg):
            if seg[i] is None:
                i += 1
                continue
            j = i
            while (j + 1 < len(seg) and seg[j + 1] is not None
                   and abs(seg[j + 1][0] - seg[j][0]) < 1e-9):
                j += 1
            for k in range(i, j + 1):
                token_len[a + k] = j - i + 1
            i = j + 1

    print("\n--- 6. token 长度分层 ---")
    for name, keep in (("single-token", lambda i: token_len[i] == 1),
                       ("multi-token", lambda i: token_len[i] >= 2)):
        sub = {i for i in valid if keep(i)}
        sub_lab = [i for i in sorted(sub) if labels[i] is not None]
        e = sum(1 for i in sub_lab if labels[i] == 0)
        pairs = [(flat[i][0], 1 if labels[i] == 0 else 0) for i in sub_lab]
        auc, ap = auc_ap(pairs)
        print(f"  {name:<13} 字{len(sub_lab):>4} 错{e:>3} 错误率{e/max(len(sub_lab),1):6.2%} "
              f"AUROC={auc:.4f} AP={ap:.4f}")
        for r in (0.05, 0.1, 0.2, 0.3, 1.0):
            sc = score(top_ratio_selection([(i, flat[i][0]) for i in sub], r), labels, valid)
            print(f"      top{r:>4.0%} P={sc['precision']:.3f} R={sc['recall']:.3f} "
                  f"标{sc['marked']:>4} 命中{sc['tp']:>3}")

    # ---------- 7. 仅单字 token + 绝对熵阈值 ----------
    print("\n--- 7. 仅单字 token 的绝对熵阈值 ---")
    for t in (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5):
        sel = {i for i in valid if token_len[i] == 1 and flat[i][0] >= t}
        sc = score(sel, labels, valid)
        print(f"  single-token & ent>={t:.2f}  P={sc['precision']:.3f} R={sc['recall']:.3f} "
              f"F1={sc['f1']:.3f} 标{sc['marked']:>4} 命中{sc['tp']:>3}")

    # ---------- 8. 每段单字 token 错误率（稳定性）----------
    print("\n--- 8. 每段单字 token 错误率 ---")
    for si, ((a, b), seg) in enumerate(zip(ranges, seg_feats)):
        idxs = [i for i in range(a, b)
                if token_len[i] == 1 and labels[i] is not None and flat[i] is not None]
        e = sum(1 for i in idxs if labels[i] == 0)
        print(f"  段{segments[si]['seq']:>2} [{a:>4}-{b:>4}) 单字token {len(idxs):>3} "
              f"错{e:>2} 错误率{e/max(len(idxs),1):6.2%}")

    # ---------- 9. entropy + token 长度加成 ----------
    print("\n--- 9. entropy + 单字 token 加成 ---")
    for beta in (0.05, 0.1, 0.2, 0.5):
        combo = [flat[i][0] + (beta if token_len[i] == 1 else 0.0) for i in range(len(hyp))]
        pairs = [(combo[i], 1 if labels[i] == 0 else 0) for i in labelled]
        auc, ap = auc_ap(pairs)
        line = f"  ent+{beta}*(single)  AUROC={auc:.4f} AP={ap:.4f}"
        for r in (0.02, 0.05, 0.1):
            sc = score(top_ratio_selection([(i, combo[i]) for i in valid], r), labels, valid)
            line += f" | top{r:.0%} P={sc['precision']:.3f} R={sc['recall']:.3f}"
        print(line)

    # ---------- 10. 段内仅单字 token 的 top-ratio（无需全局阈值）----------
    print("\n--- 10. 段内仅单字 token 的 top-ratio ---")
    for r in (0.1, 0.2, 0.3, 0.5):
        sel = set()
        for (a, _b), seg in zip(ranges, seg_feats):
            vals = [(a + off, f[0]) for off, f in enumerate(seg)
                    if f is not None and token_len[a + off] == 1]
            sel |= top_ratio_selection(vals, r)
        sc = score(sel, labels, valid)
        print(f"  段内 single-token top{r:>4.0%}  P={sc['precision']:.3f} R={sc['recall']:.3f} "
              f"F1={sc['f1']:.3f} 标{sc['marked']:>4} 命中{sc['tp']:>3}")
    print("\n--- 11. 段内单字 token top-ratio + 熵下限 ---")
    for r in (0.2, 0.3):
        for t in (0.05, 0.1, 0.2):
            sel = set()
            for (a, _b), seg in zip(ranges, seg_feats):
                vals = [(a + off, f[0]) for off, f in enumerate(seg)
                        if f is not None and token_len[a + off] == 1 and f[0] >= t]
                sel |= top_ratio_selection(vals, r)
            sc = score(sel, labels, valid)
            print(f"  seg-single-top{r:>4.0%}+ent>={t:.2f}  P={sc['precision']:.3f} "
                  f"R={sc['recall']:.3f} F1={sc['f1']:.3f} 标{sc['marked']:>4} 命中{sc['tp']:>3}")

    # ---------- 12. 局部熵突增 ----------
    print("\n--- 12. 局部熵突增（扣除前后邻居均值）---")
    spike = [None] * len(hyp)
    for (a, b), seg in zip(ranges, seg_feats):
        vals = [f[0] if f is not None else 0.0 for f in seg]
        for k in range(len(vals)):
            prev = vals[k - 1] if k > 0 else vals[k]
            nxt = vals[k + 1] if k + 1 < len(vals) else vals[k]
            spike[a + k] = vals[k] - (prev + nxt) / 2
    sub = [(i, spike[i]) for i in valid]
    pairs = [(spike[i], 1 if labels[i] == 0 else 0) for i in labelled]
    auc, ap = auc_ap(pairs)
    print(f"  spike AUROC={auc:.4f} AP={ap:.4f}")
    for r in (0.02, 0.05, 0.1):
        sc = score(top_ratio_selection(sub, r), labels, valid)
        print(f"      top{r:.0%} P={sc['precision']:.3f} R={sc['recall']:.3f}")
    # 仅单字 token 的 spike + 熵下限
    print("  单字 token & spike 排序：")
    for r in (0.02, 0.05, 0.1):
        sub2 = [(i, spike[i]) for i in valid if token_len[i] == 1]
        sc = score(top_ratio_selection(sub2, r), labels, valid)
        print(f"      top{r:.0%} P={sc['precision']:.3f} R={sc['recall']:.3f} "
              f"标{sc['marked']:>4} 命中{sc['tp']:>3}")

    # ---------- 13. 功能词过滤 ----------
    print("\n--- 13. 单字 token & ent>=0.2，排除高频虚词 ---")
    STOP = set("的了是在和与也就都而及或啊呀吧吗呢哦嗯着过地得之其则矣也")
    for label, skip_stop in (("含虚词", False), ("排除虚词", True)):
        sel = {i for i in valid if token_len[i] == 1 and flat[i][0] >= 0.2
               and not (skip_stop and hyp[i] in STOP)}
        sc = score(sel, labels, valid)
        print(f"  {label:<8} P={sc['precision']:.3f} R={sc['recall']:.3f} "
              f"F1={sc['f1']:.3f} 标{sc['marked']:>4} 命中{sc['tp']:>3}")

    # ---------- 14. 段级门控（段内单字 token 错误率不可知，仅用熵代理）----------
    print("\n--- 14. 段级门控：段内单字 token 最大熵 ---")
    for gate in (0.3, 0.5, 0.7, 0.9):
        sel = set()
        for (a, b), seg in zip(ranges, seg_feats):
            singles = [f[0] for off, f in enumerate(seg)
                       if f is not None and token_len[a + off] == 1]
            if not singles or max(singles) < gate:
                continue
            for off, f in enumerate(seg):
                if (f is not None and token_len[a + off] == 1
                        and f[0] >= 0.2):
                    sel.add(a + off)
        sc = score(sel, labels, valid)
        print(f"  gate-single-max>={gate:.1f}  P={sc['precision']:.3f} R={sc['recall']:.3f} "
              f"F1={sc['f1']:.3f} 标{sc['marked']:>4} 命中{sc['tp']:>3}")


if __name__ == "__main__":
    main()