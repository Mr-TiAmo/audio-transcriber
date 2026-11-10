#!/usr/bin/env python3
"""P2 基础置信度特征对照实验（实验计划 V1.0 §9 Phase 2，实验组 E0~E5）。

输入：P1 产出的 utterances.jsonl（含 char_features + C/S/I/D 标签）。
两个评估粒度（§7.2：片段预算为主指标，字符预算用于算法分析）：
  - 字符级：label C=1 / S,I=0（D 无法归属字符，单独报告）
  - 语句级：label has_error（产品复核单位）

实验组：
  E0 随机排序（理论参考：AP=正例率，AUROC=0.5）
  E1 -max_prob（字符取覆盖 token 最差值）
  E2 -token_logprob（字符最差 / 语句 mean/p10/min 聚合）
  E3 entropy（字符最大 / 语句 mean/max 聚合）
  E3b -top2_margin
  E5 特征组合（rank 平均 + train 拟合 Logistic）
  E4 Platt 温度校准（calibration 拟拟合，验证 H2：校准概率 vs 排序）

指标（numpy 手写，无 sklearn 依赖）：
  PR-AUC(AP) / AUROC / Recall@10/20/30% / Precision@K / ECE / Brier
  录音级 Bootstrap 95% CI（1000 次，按 recording 重采样）

用法::

    .venv-funasr/bin/python scripts/confidence_eval_p2.py --data output/confidence/dataset
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------- 指标（全部 numpy 手写） -----------------------------------------

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U 等价实现。"""
    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    # 并列取平均秩
    sorted_s = scores[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    n_pos = int(pos.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    prec = tp / (np.arange(len(y)) + 1)
    # AP = Σ 在每个正例处的 precision × recall 增量（等价 Σ prec[pos] / n_pos）
    return float(prec[y == 1].sum() / n_pos)


def recall_at_budget(scores: np.ndarray, labels: np.ndarray, frac: float) -> tuple[float, float]:
    """按风险分数降序复核前 frac 比例：返回 (recall, precision)。"""
    n_pos = int((labels == 1).sum())
    k = max(1, int(np.ceil(frac * len(labels))))
    order = np.argsort(-scores, kind="mergesort")[:k]
    hit = int((labels[order] == 1).sum())
    return hit / max(n_pos, 1), hit / k


def ece_brier(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    """等宽分桶 ECE 与 Brier。probs 为“正确概率”。"""
    if len(probs) == 0:
        return float("nan"), float("nan")
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.sum():
            ece += m.sum() / len(probs) * abs(probs[m].mean() - labels[m].mean())
    brier = float(((probs - labels) ** 2).mean())
    return float(ece), brier


def bootstrap_ci(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                 metric, n_boot: int = 1000, seed: int = 7) -> tuple[float, float]:
    """录音级（recording 为 cluster）Bootstrap 95% CI。metric(scores_subset, labels_subset)。"""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(groups, pick)
        v = metric(scores[mask], labels[mask])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------- 特征构造 ---------------------------------------------------------

def char_rows(records: list[dict], split: str) -> list[dict]:
    """字符级行。err=1 为错误（S/I）——错误检测的正类；风险分数越高越可疑。"""
    out = []
    for r in records:
        if r["split"] != split:
            continue
        for c in r["char_features"]:
            if c["label"] is None:
                continue
            out.append({
                "rec": r["recording_id"],
                "label": c["label"],        # C=1 / S,I=0（校准指标用）
                "err": 1 - c["label"],      # 错误检测正类
                "neg_logprob": -c["token_logprob"],
                "entropy": c["entropy_gibbs_nats"],
                "neg_margin": -c["top2_margin_logprob"],
                "neg_maxprob": 1.0 - c["max_prob"],
                "logprob": c["token_logprob"],
                "max_prob": c["max_prob"],
                "entropy_raw": c["entropy_gibbs_nats"],
            })
    return out


def utt_rows(records: list[dict], split: str) -> list[dict]:
    out = []
    for r in records:
        if r["split"] != split:
            continue
        cs = r["char_features"]
        if not cs:
            continue
        lps = np.array([c["token_logprob"] for c in cs])
        ents = np.array([c["entropy_gibbs_nats"] for c in cs])
        margins = np.array([c["top2_margin_logprob"] for c in cs])
        maxps = np.array([c["max_prob"] for c in cs])
        out.append({
            "rec": r["recording_id"],
            "label": r["label"]["has_error"],
            "err": r["label"]["has_error"],
            "neg_mean_logprob": -lps.mean(),
            "neg_p10_logprob": -np.percentile(lps, 10),
            "neg_min_logprob": -lps.min(),
            "mean_entropy": ents.mean(),
            "max_entropy": ents.max(),
            "high_entropy_ratio": float((ents > 0.5).mean()),
            "neg_min_margin": -margins.min(),
            "neg_mean_maxprob": -(maxps.mean()),
            "n_chars": len(cs),
        })
    return out


def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 300) -> np.ndarray:
    """简单 L2 正则逻辑回归（牛顿法或梯度下降；特征少，梯度下降够用）。"""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    for _ in range(iters):
        z = Xs @ w + b
        p = 1 / (1 + np.exp(-z))
        g = Xs.T @ (p - y) / n + l2 * w / n
        gb = (p - y).mean()
        w -= 0.5 * g
        b -= 0.5 * gb
    return (lambda X2: 1 / (1 + np.exp(-(((X2 - X.mean(0)) / (X.std(0) + 1e-9)) @ w + b))), w, b)


def platt_scale(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Platt 校准：logit(p) 上拟合 a,b（用简单网格+精修，2 参数足够）。"""
    from scipy.optimize import minimize_scalar

    p = np.clip(probs, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))

    def nll(ab):
        a, b = ab
        zz = a * z + b
        # 数值稳定的 NLL
        ll = np.logaddexp(0, zz) - labels * zz
        return ll.mean() + 1e-3 * (a * a + b * b)

    best = None
    for a0 in (0.5, 1.0, 2.0, 4.0):
        for b0 in (0.0, 2.0, -2.0):
            from scipy.optimize import minimize
            r = minimize(nll, [a0, b0], method="Nelder-Mead")
            if best is None or r.fun < best.fun:
                best = r
    return best.x


# ---------- 主流程 ------------------------------------------------------------

def evaluate_unit(rows: list[dict], feature: str, label: str = "err") -> dict:
    """统一口径：正类 = 错误，分数越高 = 越可疑。"""
    scores = np.array([r[feature] for r in rows])
    labels = np.array([r[label] for r in rows])
    groups = np.array([r["rec"] for r in rows])
    n_pos = int(labels.sum())
    res = {
        "n": len(rows), "n_pos": n_pos, "prevalence": n_pos / max(len(rows), 1),
        "AP": average_precision(scores, labels),
        "AUROC": auroc(scores, labels),
    }
    for frac in (0.10, 0.20, 0.30):
        rec, prec = recall_at_budget(scores, labels, frac)
        res[f"Recall@{int(frac*100)}%"] = rec
        res[f"Precision@{int(frac*100)}%"] = prec
    lo, hi = bootstrap_ci(scores, labels, groups, average_precision)
    res["AP_CI95"] = f"[{lo:.4f},{hi:.4f}]"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROJECT_ROOT / "output/confidence/dataset"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output/confidence/p2"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(Path(args.data) / "utterances.jsonl")]
    splits = {}
    for r in records:
        splits.setdefault(r["split"], []).append(r)
    print(f"[p2] 数据：{len(records)} 条 " +
          " ".join(f"{k}={len(v)}" for k, v in splits.items()), flush=True)

    results: dict[str, dict] = {}

    # ---- 字符级（test split） ------------------------------------------
    chars_test = char_rows(records, "test")
    print(f"[p2] 字符级 test：{len(chars_test)} 字符，错误率 "
          f"{np.mean([c['err'] for c in chars_test])*100:.2f}%", flush=True)
    for feat, name in (("neg_maxprob", "E1:char(-max_prob)"),
                       ("neg_logprob", "E2:char(-token_logprob)"),
                       ("entropy", "E3:char(entropy)"),
                       ("neg_margin", "E3b:char(-top2_margin)")):
        results[name] = evaluate_unit(chars_test, feat)

    # E5 组合（train 拟合 logistic，test 评估）
    chars_train = char_rows(records, "train")
    if chars_train and chars_test:
        fcols = ("neg_logprob", "entropy", "neg_margin", "neg_maxprob")
        Xtr = np.array([[c[f] for f in fcols] for c in chars_train])
        ytr = np.array([c["err"] for c in chars_train], dtype=float)
        predict, w, b = logistic_fit(Xtr, ytr)
        Xte = np.array([[c[f] for f in fcols] for c in chars_test])
        risk = predict(Xte)
        rows_e5 = [dict(c, e5_risk=float(rk)) for c, rk in zip(chars_test, risk)]
        results["E5:char(logistic组合)"] = evaluate_unit(rows_e5, "e5_risk")

    # E4 Platt 校准（calibration 拟拟合 E2 的 exp(logprob) 概率）
    chars_cal = char_rows(records, "calibration")
    if chars_cal and chars_test:
        p_cal = np.exp(np.array([c["logprob"] for c in chars_cal]))
        y_cal = np.array([c["label"] for c in chars_cal], dtype=float)
        a, b = platt_scale(p_cal, y_cal)
        p_te = np.exp(np.array([c["logprob"] for c in chars_test]))
        z_te = a * np.log(np.clip(p_te, 1e-6, 1 - 1e-6) / (1 - np.clip(p_te, 1e-6, 1 - 1e-6))) + b
        p_platt = 1 / (1 + np.exp(-z_te))
        y_te = np.array([c["label"] for c in chars_test], dtype=float)
        ece_raw, br_raw = ece_brier(p_te, y_te)
        ece_pl, br_pl = ece_brier(p_platt, y_te)  # p_platt 即“正确概率”（拟合目标就是 C=1）
        rows_e4 = [dict(c, e4_risk=float(1 - pp)) for c, pp in zip(chars_test, p_platt)]
        results["E4:char(platt校准)"] = evaluate_unit(rows_e4, "e4_risk")
        results["_calibration"] = {
            "platt_a": float(a), "platt_b": float(b),
            "ECE_raw": ece_raw, "Brier_raw": br_raw,
            "ECE_platt": ece_pl, "Brier_platt": br_pl,
        }

    # ---- 语句级（test split，产品主指标） --------------------------------
    utts_test = utt_rows(records, "test")
    print(f"[p2] 语句级 test：{len(utts_test)} 条，含错率 "
          f"{np.mean([u['label'] for u in utts_test])*100:.1f}%", flush=True)
    for feat, name in (("neg_min_logprob", "E2:utt(-min_logprob)"),
                       ("neg_p10_logprob", "E2:utt(-p10_logprob)"),
                       ("neg_mean_logprob", "E2:utt(-mean_logprob)"),
                       ("max_entropy", "E3:utt(max_entropy)"),
                       ("mean_entropy", "E3:utt(mean_entropy)"),
                       ("high_entropy_ratio", "E3:utt(high_entropy_ratio)"),
                       ("neg_min_margin", "E3b:utt(-min_margin)"),
                       ("neg_mean_maxprob", "E1:utt(-mean_max_prob)")):
        results[name] = evaluate_unit(utts_test, feat)

    # E5 语句级组合
    utts_train = utt_rows(records, "train")
    if utts_train and utts_test:
        fcols = ("neg_mean_logprob", "neg_p10_logprob", "mean_entropy",
                 "max_entropy", "neg_min_margin")
        Xtr = np.array([[u[f] for f in fcols] for u in utts_train])
        ytr = np.array([u["err"] for u in utts_train], dtype=float)
        predict, w, b = logistic_fit(Xtr, ytr)
        Xte = np.array([[u[f] for f in fcols] for u in utts_test])
        rows_e5u = [dict(u, e5_risk=float(rk)) for u, rk in zip(utts_test, predict(Xte))]
        results["E5:utt(logistic组合)"] = evaluate_unit(rows_e5u, "e5_risk")

    # ---- 输出 --------------------------------------------------------------
    import csv
    keys = ["n", "n_pos", "prevalence", "AP", "AUROC",
            "Recall@10%", "Precision@10%", "Recall@20%", "Precision@20%",
            "Recall@30%", "Precision@30%", "AP_CI95"]
    with open(out_dir / "baseline_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment"] + keys)
        for name, res in results.items():
            if name.startswith("_"):
                continue
            w.writerow([name] + [f"{res[k]:.4f}" if isinstance(res.get(k), float) else res.get(k)
                                 for k in keys])

    cal = results.get("_calibration", {})
    # 错误类型补充统计（D 事件）
    d_events = sum(len(r["deletion_events"]) for r in splits.get("test", []))
    s_chars = sum(1 for c in char_rows(records, "test") if c["label"] == 0)
    report = f"""# P2 基础置信度特征对照报告

- 数据：{Path(args.data).name}（test split）· 字符 {len(chars_test)} · 语句 {len(utts_test)}
- D 漏识别事件（无法归属字符特征）：{d_events} 条；字符级错误（S/I）：{s_chars} 个

## 字符级排序质量（前 8 行按 AP 排序见 csv）

## 校准（H2 验证）

| | ECE | Brier |
|---|---|---|
| 原始 exp(logprob) | {cal.get('ECE_raw', float('nan')):.4f} | {cal.get('Brier_raw', float('nan')):.4f} |
| Platt 校准后 | {cal.get('ECE_platt', float('nan')):.4f} | {cal.get('Brier_platt', float('nan')):.4f} |

结论要点：校准是否改善排序看 E4 vs E2 的 AP/Recall@20% 是否变化。
"""
    (out_dir / "p2_report.md").write_text(report)
    print(f"[p2] 完成，结果输出 {out_dir}", flush=True)
    for name, res in results.items():
        if name.startswith("_"):
            continue
        print(f"  {name:28s} AP={res['AP']:.4f} AUROC={res['AUROC']:.4f} "
              f"R@20%={res['Recall@20%']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
