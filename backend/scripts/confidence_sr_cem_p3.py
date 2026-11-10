#!/usr/bin/env python3
"""P3 SR-CEM 训练与消融（实验计划 V1.0 §9 Phase 3，实验组 E6）。

输入：confidence_nbest_p4.py 联合采集的 nbest_records.jsonl（全 split）。
特征组消融（文档 §5.2）：
  A. basic     —— E5 等价（beam 侧 logprob / 熵 / margin）
  B. srcem     —— A + rank / Top-1 / 路径累计分 / token 位置 / token 长度（E6）
  C. srcem+nbest —— B + N-best disagreement（E6+E7 组合）
模型：LogReg（线性基线）+ 单隐层 MLP（Hidden=64，SR-CEM 风格），类权重处理不平衡。
校准：MLP logits 上拟合温度 T（calibration split），报告 ECE/Brier 与排序是否变化。

用法::

    .venv-funasr/bin/python scripts/confidence_sr_cem_p3.py --records output/confidence/beam

输出 output/confidence/p3/：srcem_metrics.csv · srcem_model.npz · srcem_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from confidence_eval_p2 import evaluate_unit, ece_brier, logistic_fit  # noqa: E402

OUT_DIR = PROJECT_ROOT / "output" / "confidence" / "p3"

FEATURE_SETS = {
    "A:basic": ["neg_logprob", "local_entropy", "neg_margin"],
    "B:srcem": ["neg_logprob", "local_entropy", "neg_margin", "local_rank",
                "neg_top1", "path_before", "path_after_norm", "token_pos", "token_len"],
    "C:srcem+nbest": ["neg_logprob", "local_entropy", "neg_margin", "local_rank",
                      "neg_top1", "path_before", "path_after_norm", "token_pos",
                      "token_len", "disagree4"],
}


def char_rows(records: list[dict], split: str) -> list[dict]:
    out = []
    for r in records:
        if r["split"] != split:
            continue
        for c in r["char_features"]:
            if c["label"] is None or c["token_logprob"] is None:
                continue
            row = {
                "rec": r["recording_id"], "err": 1 - c["label"],
                "neg_logprob": -c["token_logprob"],
                "local_entropy": c.get("local_entropy") or 0.0,
                "neg_margin": -(c.get("local_top2_margin") or 0.0),
                "local_rank": float(c.get("local_rank") or 1),
                "neg_top1": -(c.get("local_top1_logprob") or 0.0),
                "path_before": c.get("path_before") or 0.0,
                "path_after_norm": c.get("path_after_norm") or 0.0,
                "token_pos": c.get("token_pos") or 0.0,
                "token_len": c.get("token_len_chars") or 1.0,
                "disagree4": 1 - c["agreement4"],
            }
            out.append(row)
    return out


def train_mlp(X: np.ndarray, y: np.ndarray, hidden: int = 64,
              epochs: int = 400, lr: float = 0.2, seed: int = 0):
    """单隐层 ReLU + sigmoid 输出；加权 BCE（正类按不平衡比加权）。"""
    n, d = X.shape
    w_pos = max((y == 0).sum() / max((y == 1).sum(), 1), 1.0)
    sw = np.where(y == 1, w_pos, 1.0).astype(float)
    sw /= sw.sum()
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.01, (d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.01, (hidden, 1))
    b2 = np.zeros(1)
    for _ in range(epochs):
        z1 = X @ W1 + b1
        h = np.maximum(z1, 0)
        p = 1 / (1 + np.exp(-(h @ W2 + b2).ravel()))
        g = ((p - y) * sw)[:, None]
        gW2, gb2 = h.T @ g, g.sum(0)
        gz1 = (g @ W2.T) * (z1 > 0)
        gW1, gb1 = X.T @ gz1, gz1.sum(0)
        W1 -= lr * gW1
        b1 -= lr * gb1
        W2 -= lr * gW2
        b2 -= lr * gb2
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return params, w_pos


def mlp_logits(params, X: np.ndarray) -> np.ndarray:
    h = np.maximum(X @ params["W1"] + params["b1"], 0)
    return (h @ params["W2"] + params["b2"]).ravel()


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    def nll(T: float) -> float:
        p = 1 / (1 + np.exp(-z / T))
        eps = 1e-9
        return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    r = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(r.x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(PROJECT_ROOT / "output/confidence/beam"))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(Path(args.records) / "nbest_records.jsonl")]
    print(f"[p3] 记录：{len(records)} 条 " + " ".join(
        f"{s}={sum(1 for r in records if r['split']==s)}" for s in ("train", "calibration", "test")),
        flush=True)

    chars = {s: char_rows(records, s) for s in ("train", "calibration", "test")}
    for s in ("train", "test"):
        if chars[s]:
            rate = np.mean([c["err"] for c in chars[s]])
            print(f"[p3] {s}：{len(chars[s])} 字符，错误率 {rate*100:.2f}%", flush=True)

    results: dict[str, dict] = {}
    saved_models = {}

    for set_name, feats in FEATURE_SETS.items():
        tr, ca, te = chars["train"], chars["calibration"], chars["test"]
        if not (tr and te):
            print(f"[p3] 跳过 {set_name}（train/test 不足）")
            continue
        Xtr = np.array([[c[f] for f in feats] for c in tr])
        ytr = np.array([c["err"] for c in tr], dtype=float)
        Xte = np.array([[c[f] for f in feats] for c in te])
        yte = np.array([c["err"] for c in te], dtype=float)

        # ---- LogReg 基线（复用 P2 的实现）----
        predict_lr, _, _ = logistic_fit(Xtr, ytr)
        rows_lr = [dict(c, risk=float(rk)) for c, rk in zip(te, predict_lr(Xte))]
        results[f"E6[{set_name}] LogReg"] = evaluate_unit(rows_lr, "risk")

        # ---- SR-CEM MLP（Hidden=64）----
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        params, w_pos = train_mlp((Xtr - mu) / sd, ytr)
        z_te = mlp_logits(params, (Xte - mu) / sd)
        rows_mlp = [dict(c, risk=float(zk)) for c, zk in zip(te, z_te)]
        results[f"E6[{set_name}] MLP"] = evaluate_unit(rows_mlp, "risk")

        if set_name == "B:srcem":
            saved_models.update(params)
            saved_models["mu"], saved_models["w_pos"] = mu, np.array([w_pos])
            np.savez(out_dir / "srcem_model.npz", feats=np.array(feats), **saved_models)

        # ---- 温度校准（仅 B 组 MLP，calibration split）----
        if set_name == "B:srcem" and ca:
            Xca = np.array([[c[f] for f in feats] for c in ca])
            yca = np.array([c["err"] for c in ca], dtype=float)
            z_ca = mlp_logits(params, (Xca - mu) / sd)
            T = fit_temperature(z_ca, yca)
            p_raw = 1 / (1 + np.exp(-z_te))
            p_T = 1 / (1 + np.exp(-z_te / T))
            ece_raw, br_raw = ece_brier(p_raw, yte)
            ece_T, br_T = ece_brier(p_T, yte)
            results["_calibration"] = {
                "T": T, "ECE_raw": ece_raw, "Brier_raw": br_raw,
                "ECE_temp": ece_T, "Brier_temp": br_T,
                # 排序是否受温度影响（应为单调变换，理论上不变）
                "AUROC_raw": results[f"E6[{set_name}] MLP"]["AUROC"],
            }

    # ---- 语句级：SR-CEM + nbest 聚合特征 ----------------------------------
    def utt_rows(split: str) -> list[dict]:
        out = []
        for r in records:
            if r["split"] != split or not r["char_features"]:
                continue
            cs = [c for c in r["char_features"] if c["token_logprob"] is not None]
            if not cs:
                continue
            ranks = [c.get("local_rank") or 1 for c in cs]
            out.append({
                "rec": r["recording_id"], "err": r["label"]["has_error"],
                "neg_mean_logprob": -np.mean([c["token_logprob"] for c in cs]),
                "mean_rank": float(np.mean(ranks)),
                "max_rank": float(np.max(ranks)),
                "mean_local_entropy": float(np.mean([c.get("local_entropy") or 0 for c in cs])),
                "mean_agreement4": r["utt_features"]["mean_agreement4"],
                "min_agreement4": r["utt_features"]["min_agreement4"],
                "mean_editdist": r["utt_features"]["mean_pairwise_editdist"],
            })
        return out

    utt_tr, utt_te = utt_rows("train"), utt_rows("test")
    if utt_tr and utt_te:
        for feats, name in ((["neg_mean_logprob", "mean_local_entropy"], "utt:basic"),
                            (["neg_mean_logprob", "mean_local_entropy", "mean_rank",
                              "max_rank"], "utt:srcem"),
                            (["neg_mean_logprob", "mean_local_entropy", "mean_rank",
                              "max_rank", "mean_agreement4", "min_agreement4",
                              "mean_editdist"], "utt:srcem+nbest")):
            Xtr = np.array([[u[f] for f in feats] for u in utt_tr])
            ytr = np.array([u["err"] for u in utt_tr], dtype=float)
            Xte = np.array([[u[f] for f in feats] for u in utt_te])
            predict, _, _ = logistic_fit(Xtr, ytr)
            rows = [dict(u, risk=float(rk)) for u, rk in zip(utt_te, predict(Xte))]
            results[f"E6[{name}] LogReg"] = evaluate_unit(rows, "risk")

    # ---- 输出 --------------------------------------------------------------
    import csv
    keys = ["n", "n_pos", "prevalence", "AP", "AUROC", "Recall@10%",
            "Recall@20%", "Recall@30%", "AP_CI95"]
    with open(out_dir / "srcem_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment"] + keys)
        for name, res in results.items():
            if name.startswith("_"):
                continue
            w.writerow([name] + [f"{res[k]:.4f}" if isinstance(res.get(k), float) else res.get(k)
                                 for k in keys])

    cal = results.get("_calibration", {})
    lines = ["# P3 SR-CEM 实验报告", "",
             f"- 记录：{len(records)} · 字符 train {len(chars['train'])} / test {len(chars['test'])}",
             f"- MLP：Hidden=64，加权 BCE（按类不平衡加权）",
             "", "## 消融（§5.2 E5 vs E6 vs E6+E7）", ""]
    for name, res in results.items():
        if name.startswith("_"):
            continue
        lines.append(f"- {name}: AP={res['AP']:.4f} AUROC={res['AUROC']:.4f} "
                     f"R@20%={res['Recall@20%']:.3f}")
    if cal:
        lines += ["", "## 温度校准（calibration split 拟合）", "",
                  f"- T={cal['T']:.3f} · ECE {cal['ECE_raw']:.4f} → {cal['ECE_temp']:.4f}"
                  f" · Brier {cal['Brier_raw']:.4f} → {cal['Brier_temp']:.4f}",
                  f"- 排序不受温度影响（单调变换）：AUROC 不变"]
    (out_dir / "srcem_report.md").write_text("\n".join(lines) + "\n")
    print(f"[p3] 完成，结果输出 {out_dir}", flush=True)
    for name, res in results.items():
        if name.startswith("_"):
            continue
        print(f"  {name:34s} AP={res['AP']:.4f} AUROC={res['AUROC']:.4f} "
              f"R@20%={res['Recall@20%']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
