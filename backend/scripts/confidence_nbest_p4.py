#!/usr/bin/env python3
"""P4 N-best 一致性 + P3 SR-CEM 特征联合采集（实验计划 V1.0 §9 Phase 3/4，E6/E7）。

一次 beam-4 / 4 候选推理同时产出：
  P4（E7）：
    - N=2 与 N=4 两组逐字符 agreement（候选 0 与候选 1..3 逐对字符对齐）
    - pairwise 归一化编辑距离、候选长度差
  P3（E6，SR-CEM 特征）：
    - 候选 0 逐 token：transition score、父 beam 局部分布的 rank / Top-1 /
      Top-2 margin / Gibbs 熵、路径前/后累计分数、token 位置与长度
    - 内置语义自检：局部分布 log_softmax 的被选 logprob 应 ≈ transition score
  共用：
    - 候选 0 逐字符特征（token 特征聚合）与错误标签（v2 语气词规则）
    - greedy 与 beam-0 文本/CER/耗时对照

用法::

    # 联合采集（全 split，供 P3 训练 + P4 评估）
    .venv-funasr/bin/python scripts/confidence_nbest_p4.py \
        --data output/confidence/dataset --split all --out output/confidence/beam

输出 <out>/：
  nbest_records.jsonl（含双特征族）· nbest_metrics.csv · blind_spot_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402

from confidence_build_dataset import (  # noqa: E402
    FILLER, align_ops, apply_filler_rules, conservative_norm, load_model,
)
from confidence_eval_p2 import evaluate_unit  # noqa: E402

OUT_DIR = PROJECT_ROOT / "output" / "confidence" / "beam"
MAX_NEW_TOKENS = 200


def pair_matched_flags(cand: str, best: str) -> list[bool]:
    """对齐（cand 为参考，best 为假设）：best 每字符是否与该候选一致。

    语气词互换视为一致（与 v2 标签规则同口径）。
    """
    ops = align_ops(cand, best)
    flags = [False] * len(best)
    for op, ri, hi in ops:
        if hi is None:
            continue
        if op == "C":
            flags[hi] = True
        elif op == "S" and ri is not None and cand[ri] in FILLER and best[hi] in FILLER:
            flags[hi] = True
    return flags


def pair_edit_distance_norm(cand: str, best: str) -> float:
    ops = align_ops(cand, best)
    dist = sum(1 for op, _, _ in ops if op != "C")
    return dist / max(len(cand), 1)


def generate_nbest(processor, model, wav: Path, device: str):
    """beam-4 生成 4 候选。返回 (out, seqs, texts, ts, wall_ms)。"""
    import torch

    inputs = processor.apply_transcription_request(audio=str(wav), language="Chinese")
    inputs = inputs.to(device, model.dtype)
    input_len = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, num_beams=4,
            num_return_sequences=4, do_sample=False,
            return_dict_in_generate=True, output_scores=True)
    wall_ms = (time.perf_counter() - t0) * 1000
    seqs = out.sequences[:, input_len:]
    texts = processor.decode(seqs, return_format="transcription_only")
    ts = model.compute_transition_scores(
        out.sequences, out.scores, out.beam_indices, normalize_logits=False)
    return out, seqs, texts, ts, wall_ms


def sr_cem_token_features(out, seqs, ts, tok) -> tuple[list[dict], dict]:
    """候选 0 逐 token 特征：基础 + SR-CEM（父 beam 局部分布）+ 路径分数。

    返回 (tokens, 自检统计)。语义依据（transformers 5.22.1 源码）：
    beam_indices[i][j] = 假设 i 第 j 步 token 在 scores[j]((beams,vocab)) 的父 beam 行。
    """
    import torch

    gen_ids = seqs[0].tolist()
    beam_idx_row = out.beam_indices[0].tolist() if getattr(out, "beam_indices", None) is not None else None
    tokens, prev_len = [], 0
    check = {"hit": 0, "n": 0}
    for step, tid in enumerate(gen_ids):
        token_text = tok.decode([tid], skip_special_tokens=True)
        prefix = tok.decode(gen_ids[: step + 1], skip_special_tokens=True)
        if not token_text:  # <|im_end|> 等特殊 token
            continue
        feats = {
            "char_start": prev_len, "char_end": len(prefix),
            "token_logprob": float(ts[0, step].item()),
            "token_pos": step, "token_len_chars": len(token_text),
        }
        # SR-CEM：父 beam 局部条件分布
        if beam_idx_row is not None and 0 <= beam_idx_row[step] < out.scores[step].shape[0]:
            row = out.scores[step][beam_idx_row[step]].to(torch.float32)
            local = torch.log_softmax(row, dim=-1)
            sel = float(local[tid].item())
            feats["local_sel_logprob"] = sel
            check["n"] += 1
            if abs(sel - feats["token_logprob"]) < 0.02:
                check["hit"] += 1
            top4 = torch.topk(local, k=4)
            feats["local_rank"] = int((local > local[tid]).sum().item()) + 1
            feats["local_top1_logprob"] = float(top4.values[0].item())
            feats["local_top2_margin"] = float((top4.values[0] - top4.values[1]).item())
            feats["local_entropy"] = float(-(local.exp() * local).sum().item())
            feats["local_top4_sum"] = float(top4.values.sum().item())
        # 路径累计分数（transition score 累计；padding 位为 0 不影响）
        feats["path_before"] = float(ts[0, : step].sum().item())
        feats["path_after"] = float(ts[0, step:].sum().item())
        feats["path_after_norm"] = feats["path_after"] / max(len(gen_ids) - step, 1)
        tokens.append(feats)
        prev_len = len(prefix)
    return tokens, check


def build_record(processor, model, wav: Path, ref_text: str, device: str) -> dict | None:
    import torch  # noqa: F401,PLC0415

    out, seqs, texts, ts, wall_ms = generate_nbest(processor, model, wav, device)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    best_raw = texts[0]
    norm_best, best_map = conservative_norm(best_raw)
    if not norm_best:
        return None

    tokens, check = sr_cem_token_features(out, seqs, ts, tok)

    # 逐字符聚合（字符取覆盖 token 的最差/最值，与 greedy 侧同口径）
    agg_keys = {
        "token_logprob": min, "local_rank": max, "local_entropy": max,
        "local_top2_margin": min, "local_top1_logprob": max, "local_top4_sum": min,
        "path_before": min, "path_after": min, "path_after_norm": min,
        "token_pos": None, "token_len_chars": None,
    }
    char_vals: list[dict] = [{} for _ in norm_best]
    for t in tokens:
        start = end = None
        for k, ri in enumerate(best_map):
            if ri >= t["char_start"] and start is None:
                start = k
            if ri < t["char_end"]:
                end = k + 1
        if start is None or end is None or end <= start:
            continue
        for j in range(start, end):
            cv = char_vals[j]
            for key in agg_keys:
                if key in t:
                    cv.setdefault(key, []).append(t[key])
    # token_pos / token_len 取均值，其余按 min/max
    char_features = []
    for j, cv in enumerate(char_vals):
        row = {"char_idx": j, "char": norm_best[j]}
        for key, fn in agg_keys.items():
            if key not in cv:
                row[key if key != "token_logprob" else "token_logprob"] = None
                continue
            if fn is None:
                row[key] = float(np.mean(cv[key]))
            else:
                row[key] = fn(cv[key])
        char_features.append(row)

    # 一致性：候选 0 vs 候选 1..3
    norm_cands = [conservative_norm(t)[0] for t in texts[1:]]
    n_match4 = [0] * len(norm_best)
    agree2 = [None] * len(norm_best)
    edit_dists = []
    for ci, cand in enumerate(norm_cands):
        if cand:
            flags = pair_matched_flags(cand, norm_best)
            edit_dists.append(pair_edit_distance_norm(cand, norm_best))
        else:
            flags = [False] * len(norm_best)
            edit_dists.append(1.0)
        for j, m in enumerate(flags):
            if m:
                n_match4[j] += 1
                if ci == 0:
                    agree2[j] = 1.0
            elif ci == 0:
                agree2[j] = 0.0
    agree4 = [m / len(norm_cands) for m in n_match4]

    # 候选 0 的标签（v2 规则，对齐参考文本）
    norm_ref, _ = conservative_norm(ref_text)
    ops = align_ops(norm_ref, norm_best)
    ops, filler_ignored = apply_filler_rules(ops, norm_ref, norm_best)
    labels = [None] * len(norm_best)
    deletions = []
    for op, ri, hi in ops:
        if op == "D":
            deletions.append({"ref_char_idx": ri, "ref_char": norm_ref[ri]})
        elif hi is not None:
            labels[hi] = 1 if op == "C" else 0
    s_count = sum(1 for op, _, _ in ops if op == "S")
    i_count = sum(1 for op, _, _ in ops if op == "I")
    d_count = len(deletions)

    for cf, lab, a4, a2 in zip(char_features, labels, agree4, agree2):
        cf["label"] = lab
        cf["agreement4"] = a4
        cf["agreement2"] = a2

    cand_lens = [len(norm_best)] + [len(c) for c in norm_cands]
    return {
        "best_text_raw": best_raw,
        "candidates_raw": texts[1:],
        "norm_best": norm_best,
        "char_features": char_features,
        "deletion_events": deletions,
        "utt_features": {
            "mean_agreement4": float(np.mean(agree4)),
            "min_agreement4": float(np.min(agree4)),
            "mean_agreement2": float(np.mean([a for a in agree2 if a is not None])),
            "mean_pairwise_editdist": float(np.mean(edit_dists)),
            "max_pairwise_editdist": float(np.max(edit_dists)),
            "cand_len_spread": max(cand_lens) - min(cand_lens),
        },
        "label": {
            "rule": "v2_filler_equiv",
            "has_error": int(s_count + i_count + d_count > 0),
            "S": s_count, "I": i_count, "D": d_count,
            "filler_ignored": filler_ignored,
        },
        "transition_check": check,
        "wall_ms": wall_ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-ASR-1.7B-hf")
    ap.add_argument("--data", default=str(PROJECT_ROOT / "output/confidence/dataset"))
    ap.add_argument("--split", default="test", help="test / train / calibration / all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    import torch
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data)

    greedy = {}
    for line in open(data_dir / "utterances.jsonl"):
        g = json.loads(line)
        greedy[g["recording_id"]] = g
    if args.split == "all":
        targets = list(greedy.values())
    else:
        targets = [g for g in greedy.values() if g["split"] == args.split]
    if args.limit:
        targets = targets[: args.limit]
    print(f"[beam] 目标：{len(targets)} 条（split={args.split}），beam-4 / 4 候选，"
          f"双特征族（E7 一致性 + E6 SR-CEM）", flush=True)

    processor, model = load_model(str(PROJECT_ROOT / args.model), device)
    records = []
    t_start = time.perf_counter()
    check_total = {"hit": 0, "n": 0}
    for k, g in enumerate(targets):
        wav = PROJECT_ROOT / g["audio"]
        try:
            rec = build_record(processor, model, wav, g["ref_text"], device)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {g['recording_id']}: {type(e).__name__}: {e}", flush=True)
            continue
        if rec is None:
            continue
        rec["recording_id"] = g["recording_id"]
        rec["speaker_id"] = g["speaker_id"]
        rec["split"] = g["split"]
        rec["greedy_wall_ms"] = g["wall_ms"]
        rec["greedy_norm_hyp"] = g["norm_hyp"]
        rec["greedy_label"] = g["label"]
        rec["same_as_greedy"] = (rec["norm_best"] == g["norm_hyp"])
        rec["greedy_char_features"] = g["char_features"]
        records.append(rec)
        check_total["hit"] += rec["transition_check"]["hit"]
        check_total["n"] += rec["transition_check"]["n"]
        if (k + 1) % 40 == 0:
            el = time.perf_counter() - t_start
            ratio = sum(r["wall_ms"] for r in records) / max(
                1, sum(r["greedy_wall_ms"] for r in records))
            print(f"[beam] {k+1}/{len(targets)} | beam/greedy {ratio:.2f}x | "
                  f"自检 {check_total['hit']}/{check_total['n']} | "
                  f"ETA {el/(k+1)*(len(targets)-k-1)/60:.0f}min", flush=True)

    with open(out_dir / "nbest_records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[beam] 落盘 {len(records)} 条 -> {out_dir/'nbest_records.jsonl'}", flush=True)
    print(f"[beam] transition 语义自检：{check_total['hit']}/{check_total['n']} "
          f"({check_total['hit']/max(check_total['n'],1)*100:.1f}% 吻合)", flush=True)

    # ---------- 评估（仅 test split；P4 一致性 + SR-CEM 基础特征） -----------
    results = {}
    test_recs = [r for r in records if r["split"] == "test"]
    chars = []
    for r in test_recs:
        for c in r["char_features"]:
            if c["label"] is None or c["token_logprob"] is None:
                continue
            chars.append({"rec": r["recording_id"], "err": 1 - c["label"],
                          "disagree4": 1 - c["agreement4"],
                          "disagree2": 1 - (c["agreement2"] if c["agreement2"] is not None else 1),
                          "neg_logprob": -c["token_logprob"]})
    if chars:
        for feat in ("disagree4", "neg_logprob"):
            s = np.array([c[feat] for c in chars])
            rank = s.argsort().argsort() / max(len(s) - 1, 1)
            for c, rk in zip(chars, rank):
                c[f"_rank_{feat}"] = float(rk)
        for c in chars:
            c["combo"] = c["_rank_disagree4"] * 0.5 + c["_rank_neg_logprob"] * 0.5
        results["E7:char(1-agreement4)"] = evaluate_unit(chars, "disagree4")
        results["E7:char(1-agreement2)"] = evaluate_unit(chars, "disagree2")
        results["E2beam:char(-logprob)"] = evaluate_unit(chars, "neg_logprob")
        results["E7+E2beam:char(组合)"] = evaluate_unit(chars, "combo")

    utts = [{"rec": r["recording_id"], "err": r["label"]["has_error"],
             **r["utt_features"]} for r in test_recs]
    if utts:
        results["E7:utt(1-mean_agree4)"] = evaluate_unit(utts, "mean_agreement4")
        results["E7:utt(1-min_agree4)"] = evaluate_unit(utts, "min_agreement4")
        results["E7:utt(pairwise_editdist)"] = evaluate_unit(utts, "mean_pairwise_editdist")
        results["E7:utt(max_editdist)"] = evaluate_unit(utts, "max_pairwise_editdist")
        results["E7:utt(len_spread)"] = evaluate_unit(utts, "cand_len_spread")

    import csv
    keys = ["n", "n_pos", "prevalence", "AP", "AUROC", "Recall@10%",
            "Recall@20%", "Recall@30%", "AP_CI95"]
    with open(out_dir / "nbest_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment"] + keys)
        for name, res in results.items():
            w.writerow([name] + [f"{res[k]:.4f}" if isinstance(res.get(k), float) else res.get(k)
                                 for k in keys])

    # ---------- 盲区分析：greedy 自信错误被 N-best 分歧捕获率 --------------------
    blind_total = blind_caught = 0
    blind_cases = []
    agree_on_correct = []
    for r in test_recs:
        if not r["same_as_greedy"] or len(r["greedy_char_features"]) != len(r["char_features"]):
            continue
        for gc, bc in zip(r["greedy_char_features"], r["char_features"]):
            if gc["label"] == 1:
                if bc["agreement4"] is not None:
                    agree_on_correct.append(bc["agreement4"])
                continue
            if gc["max_prob"] > 0.99:
                blind_total += 1
                if bc["agreement4"] < 1:
                    blind_caught += 1
                else:
                    blind_cases.append((r["recording_id"], bc["char"],
                                        bc["agreement4"], bc["token_logprob"]))

    def cer_of(rows, key):
        s = sum(r[key]["S"] + r[key]["I"] + r[key]["D"] for r in rows)
        n_ref = sum(len(conservative_norm(greedy[r["recording_id"]]["ref_text"])[0])
                    for r in rows)
        return s / max(n_ref, 1)

    cer_g = cer_of(test_recs, "greedy_label")
    cer_b = cer_of(test_recs, "label")

    n_err = sum(1 for c in chars if c["err"] == 1)
    summary = {
        "n_utts": len(records), "n_test": len(test_recs),
        "n_chars": len(chars), "n_err_chars": n_err,
        "same_as_greedy_ratio": (sum(1 for r in test_recs if r["same_as_greedy"])
                                 / max(len(test_recs), 1)),
        "cer_greedy": cer_g, "cer_beam0": cer_b,
        "beam_vs_greedy_wall_ratio": (sum(r["wall_ms"] for r in test_recs)
                                      / max(1, sum(r["greedy_wall_ms"] for r in test_recs))),
        "transition_check": check_total,
        "blind_spot_total": blind_total, "blind_spot_caught": blind_caught,
        "blind_spot_catch_rate": blind_caught / max(blind_total, 1),
        "mean_agreement_on_correct_chars": float(np.mean(agree_on_correct)) if agree_on_correct else None,
        "uncaught_confident_errors": [
            {"rec": rid, "char": ch, "agreement4": a, "logprob": lp}
            for rid, ch, a, lp in blind_cases[:20]],
    }
    (out_dir / "blind_spot_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[beam] test: {len(test_recs)} 条（与 greedy 一致率 "
          f"{summary['same_as_greedy_ratio']*100:.1f}%，CER {cer_g*100:.2f}% vs "
          f"{cer_b*100:.2f}%）", flush=True)
    print(f"[beam] 自信错误盲区：{blind_caught}/{blind_total} 被捕获"
          f"（{summary['blind_spot_catch_rate']*100:.0f}%）", flush=True)
    for name, res in results.items():
        print(f"  {name:30s} AP={res['AP']:.4f} AUROC={res['AUROC']:.4f} "
              f"R@20%={res['Recall@20%']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
