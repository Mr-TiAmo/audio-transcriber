#!/usr/bin/env python3
"""Windows 置信度探针（qwen-asr 0.0.6 API 适配版）。

macOS 调研的 P0/P2/P4 实验脚本用的是旧 processor API
（apply_transcription_request / decode(return_format=...)），在 Windows
qwen-asr 0.0.6 下不可用。本脚本用生产路径同款 API（apply_chat_template +
normalize_audios + batch_decode）复现核心实验，验证文档结论在本机成立：

  实验一  token 级 logprob / 熵 / top-2 margin（greedy + output_scores）
  实验二  N-best agreement（beam-4 × 4 候选，候选间字符对齐）
  实验三  n-gram 重复（免费信号，无额外推理）

评估口径与 confidence_eval_p2 一致：
  - 字符级标签：C=1（正确）/ S,I=0（错误）；D 无法归属字符不计入
  - 语句级标签：has_error（产品复核单位）
  - 分数方向统一为"错误风险"（越大越可能错）：AP>正例率即有信号

用法::

    .venv/Scripts/python.exe -I scripts/confidence_probe_win.py \
        --model models/Qwen3-ASR-0.6B --limit 60

输出 output/confidence/win_probe_report.json + 控制台摘要。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

LIPS_META = PROJECT_ROOT / "datasets/chinese-lips/meta_test.csv"
LIPS_WAV = PROJECT_ROOT / "datasets/chinese-lips/processed_test"
OUT_DIR = PROJECT_ROOT / "output" / "confidence"

from confidence_build_dataset import (  # noqa: E402
    apply_filler_rules, conservative_norm, align_ops,
)
from qwen3_asr_transcribe import _parse_raw_output, _prepare_asr_inputs  # noqa: E402

MAX_NEW_TOKENS = 256


# ---------------------------------------------------------------- 指标
def average_precision(pairs: list[tuple[float, int]]) -> float | None:
    """PR-AUC（AP）：pairs = (分数, 标签)。分数方向=错误风险。"""
    pos = sum(1 for _, y in pairs if y)
    if pos == 0 or pos == len(pairs):
        return None
    ranked = sorted(pairs, key=lambda p: -p[0])
    hits, ap, seen = 0, 0.0, 0
    for rank, (_, y) in enumerate(ranked, 1):
        if y:
            hits += 1
            ap += hits / rank
        seen += 1
    return ap / pos


def auroc(pairs: list[tuple[float, int]]) -> float | None:
    """AUROC（秩法，并列取平均秩）。"""
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    # 平均秩
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    pos_rank = sum(ranks[i] for i in range(len(pairs)) if pairs[i][1])
    n_pos, n_neg = len(pos), len(neg)
    return (pos_rank - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ---------------------------------------------------------------- 推理
def token_char_spans(tokenizer, gen_ids: list[int]) -> list[tuple[int, int]]:
    """逐 token 的归一化字符区间（prefix decode 归一化长度差分）。"""
    spans: list[tuple[int, int]] = []
    acc = 0
    for i in range(len(gen_ids)):
        prefix = tokenizer.decode(gen_ids[: i + 1], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
        n = len(conservative_norm(prefix)[0])
        if n < acc:  # 分词边界把字符归一化长度拉回（异常，防越界）
            n = acc
        spans.append((acc, n))
        acc = n
    return spans


def greedy_with_scores(processor, model, wav: Path, device: str) -> dict:
    """greedy + output_scores：返回 {text, gen_ids, scores}（scores 每步 vocab logits）。"""
    import torch

    texts, wavs = _prepare_asr_inputs(processor, [str(wav)], None, "Chinese")
    inputs = processor(text=texts, audio=wavs, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.no_grad():
        # qwen-asr 0.0.6 包装层强制 return_dict_in_generate=True（外部再传会
        # 报"multiple values"）；output_scores 透传给 thinker 即可拿到逐步 logits
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, output_scores=True)
    input_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[0, input_len:].tolist()
    raw = processor.batch_decode(out.sequences[:, input_len:],
                                 skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]
    return {"text": _parse_raw_output(raw, "Chinese"), "gen_ids": gen_ids,
            "scores": out.scores}


def nbest_candidates(processor, model, wav: Path) -> list[str]:
    """beam-4 × 4 候选：返回按分数排序的候选文本列表。"""
    import torch

    texts, wavs = _prepare_asr_inputs(processor, [str(wav)], None, "Chinese")
    inputs = processor(text=texts, audio=wavs, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             num_beams=4, num_return_sequences=4,
                             do_sample=False)
    output_ids = out.sequences if hasattr(out, "sequences") else out
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    raws = processor.batch_decode(generated, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
    return [_parse_raw_output(r, "Chinese") for r in raws]


def token_features(scores, gen_ids: list[int]) -> list[dict]:
    """每 token 的错误风险特征（neg_logprob / entropy / neg_margin）。"""
    import torch

    feats = []
    for step, (logits, tid) in enumerate(zip(scores, gen_ids)):
        z = logits[0].to(torch.float32)
        logp = torch.log_softmax(z, dim=-1)
        neg_logprob = -float(logp[tid].item())
        p = logp.exp()
        entropy = float(-(p * logp).sum().item())
        top2 = torch.topk(logp, k=2).values
        neg_margin = -float((top2[0] - top2[1]).item())
        feats.append({"neg_logprob": neg_logprob, "entropy": entropy,
                      "neg_margin": neg_margin})
    return feats


# ---------------------------------------------------------------- 标签与对齐
def char_labels(ref_text: str, hyp_norm: str) -> list[int | None]:
    """参考 vs 假设（归一化后）对齐 → 假设每字符标签（1 正确 / 0 错误 / None 语气词不计）。"""
    ref_norm, _ = conservative_norm(ref_text)
    ops = align_ops(ref_norm, hyp_norm)
    ops, _ = apply_filler_rules(ops, ref_norm, hyp_norm)
    labels: list[int | None] = [None] * len(hyp_norm)
    for op, ri, hi in ops:
        if hi is None:
            continue
        if op == "C":
            labels[hi] = 1
        elif op in ("S", "I"):
            labels[hi] = 0
    return labels


def ngram_repetition_mask(hyp_norm: str, n: int = 3, min_count: int = 2) -> list[int]:
    """免费信号：字符是否处于重复 n-gram 内（幻觉/卡壳的代理）。"""
    count: dict[str, int] = {}
    for i in range(len(hyp_norm) - n + 1):
        gram = hyp_norm[i:i + n]
        count[gram] = count.get(gram, 0) + 1
    mask = [0] * len(hyp_norm)
    for i in range(len(hyp_norm) - n + 1):
        if count.get(hyp_norm[i:i + n], 0) >= min_count:
            for k in range(i, i + n):
                mask[k] = 1
    return mask


def agreement_mask(cands: list[str]) -> list[float] | None:
    """候选 0 的每字符与候选 1..3 的对齐一致率（0, 1/3, 2/3, 1）。"""
    if not cands:
        return None
    base_norm, _ = conservative_norm(cands[0])
    if not base_norm:
        return None
    agree = [0] * len(base_norm)
    for cand in cands[1:]:
        other_norm, _ = conservative_norm(cand)
        ops = align_ops(other_norm, base_norm)
        for op, ri, hi in ops:
            if hi is not None and op == "C":
                agree[hi] += 1
    return [a / max(len(cands) - 1, 1) for a in agree]


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-ASR-0.6B")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(LIPS_META, encoding="utf-8"))
            if (LIPS_WAV / f"{r['ID']}.wav").exists()]
    # 跨说话人轮询抽样（复用 P1 的防偏策略）
    by_spk: dict[str, list[dict]] = {}
    for r in rows:
        by_spk.setdefault(r["ID"].rsplit("_", 1)[0], []).append(r)
    pools = [list(v) for v in by_spk.values()]
    random.Random(args.seed).shuffle(pools)
    picked: list[dict] = []
    while pools and len(picked) < args.limit:
        for pool in list(pools):
            if pool:
                picked.append(pool.pop(0))
            if not pool:
                pools.remove(pool)
            if len(picked) >= args.limit:
                break
    rows = picked
    print(f"[probe] LiPS 抽样 {len(rows)} 条（跨 {len({r['ID'].rsplit('_', 1)[0] for r in rows})} 个说话人组）", flush=True)

    import torch
    from transformers import AutoModel, AutoProcessor

    # transformers 4.57.6 无 AutoModelForMultimodalLM；qwen-asr 包在 import 时
    # 向 AutoModel/AutoProcessor 注册 Qwen3ASR（与生产脚本 load_model 同款）
    import qwen_asr  # noqa: F401

    model_path = (PROJECT_ROOT / args.model).resolve()
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
                                               trust_remote_code=False)
    model = AutoModel.from_pretrained(
        str(model_path), dtype=torch.float16, local_files_only=True,
        trust_remote_code=False).to(args.device)
    model.eval()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # 字符级 (特征, 标签)；语句级 (分数, has_error)
    char_pairs: dict[str, list[tuple[float, int]]] = {}
    utt_pairs: dict[str, list[tuple[float, int]]] = {}

    def add(feat: str, score: float, label: int) -> None:
        char_pairs.setdefault(feat, []).append((score, label))

    t0 = time.perf_counter()
    n_done = 0
    for k, r in enumerate(rows):
        wav = LIPS_WAV / f"{r['ID']}.wav"
        try:
            g = greedy_with_scores(processor, model, wav, args.device)
            cands = nbest_candidates(processor, model, wav)
        except Exception as e:  # noqa: BLE001 - 探针记录失败形态
            print(f"[warn] {r['ID']}: {type(e).__name__}: {e}", flush=True)
            continue

        hyp_norm, _ = conservative_norm(g["text"])
        if not hyp_norm:
            continue
        labels = char_labels(r["TEXT"], hyp_norm)
        rep_mask = ngram_repetition_mask(hyp_norm)
        agree = agreement_mask(cands)

        # token 特征 → 字符区间
        feats = token_features(g["scores"], g["gen_ids"])
        spans = token_char_spans(tokenizer, g["gen_ids"])
        tok_feats_by_char: list[dict | None] = [None] * len(hyp_norm)
        total_span = spans[-1][1] if spans else 0
        if total_span == len(hyp_norm):
            for f, (a, b) in zip(feats, spans):
                for ci in range(a, b):
                    if ci < len(tok_feats_by_char):
                        tok_feats_by_char[ci] = f

        has_error = 0
        for ci, lab in enumerate(labels):
            if lab is None:
                continue  # 语气词不计
            # 正类统一为"错误"（lab=0 正确 -> err=0；lab=1 正确 -> err=0；
            # lab=0 错误 -> err=1），与"错误风险"分数方向一致
            err = 0 if lab == 1 else 1
            if err:
                has_error = 1
            tf = tok_feats_by_char[ci]
            if tf is not None:
                for key in ("neg_logprob", "entropy", "neg_margin"):
                    add(key, tf[key], err)
            add("repetition", float(rep_mask[ci]), err)
            if agree is not None and ci < len(agree):
                add("nbest_disagreement", 1.0 - agree[ci], err)

        # 语句级：聚合分数 = 该句字符特征均值
        utt_feats: dict[str, float] = {}
        for key in ("neg_logprob", "entropy", "neg_margin", "nbest_disagreement", "repetition"):
            vals = [v for v in (
                (tok_feats_by_char[ci] or {}).get(key) if key != "repetition" else rep_mask[ci]
                for ci in range(len(hyp_norm)) if labels[ci] is not None and (
                    tok_feats_by_char[ci] is not None or key == "repetition"))
                if v is not None]
            if key == "nbest_disagreement" and agree is not None:
                vals = [1.0 - agree[ci] for ci in range(len(hyp_norm))
                        if labels[ci] is not None and ci < len(agree)]
            if vals:
                utt_feats[key] = sum(vals) / len(vals)
        for key, score in utt_feats.items():
            utt_pairs.setdefault(key, []).append((score, has_error))

        n_done += 1
        if (k + 1) % 10 == 0:
            print(f"[probe] {k + 1}/{len(rows)} 条完成（{time.perf_counter() - t0:.0f}s）", flush=True)

    # 汇总
    def report(pairs: dict) -> list[dict]:
        out = []
        n_pos = sum(1 for _, y in next(iter(pairs.values()), [(0, 0)]) if y)
        for feat, ps in sorted(pairs.items()):
            pos_rate = sum(1 for _, y in ps if y) / max(len(ps), 1)
            out.append({
                "feature": feat, "n": len(ps),
                "pos_rate": round(pos_rate, 4),
                "AP": round(average_precision(ps), 4) if average_precision(ps) is not None else None,
                "AUROC": round(auroc(ps), 4) if auroc(ps) is not None else None,
            })
        return out

    result = {
        "n_utterances": n_done,
        "model": str(model_path),
        "label_rule": "v2_filler_equiv; C=1 / S,I=0; D 不归属",
        "char_level": report(char_pairs),
        "utt_level": report(utt_pairs),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "win_probe_report.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[probe] 完成 {n_done} 条，耗时 {time.perf_counter() - t0:.0f}s -> {out_path}")
    print("\n== 字符级（分数=错误风险，AP>正例率即有信号） ==")
    print(f"{'特征':<20}{'样本':>8}{'正例率':>10}{'AP':>10}{'AUROC':>10}")
    for row in result["char_level"]:
        print(f"{row['feature']:<20}{row['n']:>8}{row['pos_rate']:>10.4f}"
              f"{row['AP']!s:>10}{row['AUROC']!s:>10}")
    print("\n== 语句级（has_error） ==")
    print(f"{'特征':<20}{'样本':>8}{'正例率':>10}{'AP':>10}{'AUROC':>10}")
    for row in result["utt_level"]:
        print(f"{row['feature']:<20}{row['n']:>8}{row['pos_rate']:>10.4f}"
              f"{row['AP']!s:>10}{row['AUROC']!s:>10}")


if __name__ == "__main__":
    main()