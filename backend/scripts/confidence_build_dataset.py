#!/usr/bin/env python3
"""P1 置信度实验数据集构建（实验计划 V1.0 §9 Phase 1 + Phase 2 特征一次采集）。

数据源：Chinese-LiPS test（3908 条独立语句，21 个说话人组，~10.3 小时）。

一次 Greedy 推理同时产出：
  1. 假设文本（asr_hypotheses）
  2. 逐 Token 分数特征 → 逐字符特征（P2 baseline_features）
  3. 字符级 C/S/I/D 错误标签（char_alignment）
  4. 按说话人分组的数据集切分（防泄漏：同一说话人不跨集合）

规范化口径（冻结，见 data_quality_report.md）：
  - 字符级标签用保守归一化：仅剥离标点/空白（_KEEP 同款过滤，不做数字转换），
    保持 raw 字符 -> 归一化字符 1:1 映射，供特征-标签对齐。
  - 段级 CER 同时报告两种口径：保守口径 + eval_cer 的数字归一口径。

用法::

    .venv-funasr/bin/python scripts/confidence_build_dataset.py [--limit N] [--seed 42]

输出 output/confidence/dataset/：
  utterances.jsonl · dataset_split.json · recordings_manifest.csv · data_quality_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT_DIR = PROJECT_ROOT / "output" / "confidence" / "dataset"
LIPS_META = PROJECT_ROOT / "datasets/chinese-lips/meta_test.csv"
LIPS_WAV = PROJECT_ROOT / "datasets/chinese-lips/processed_test"

# 说话人分组切分（21 组 -> train 15 / calibration 2 / test 4）
SPLIT_PLAN = {"train": 15, "calibration": 2, "test": 4}
MAX_NEW_TOKENS = 200
_KEEP = re.compile(r"[^一-鿿㐀-䶿a-zA-Z0-9]")  # 与 eval_cer._KEEP 同款：仅保 CJK/字母/数字
# 语气词等价集（与 eval_cer._FILL 同款）：互换/增删不计为错误（标签规则 v2）
FILLER = set("啊嘛哈呢吧哦呀嗯呃")
LABEL_RULE = "v2_filler_equiv"


def conservative_norm(text: str) -> tuple[str, list[int]]:
    """保守归一化：剥离标点/空白，返回 (归一化文本, 原文本字符下标映射)。"""
    kept, idx_map = [], []
    for i, ch in enumerate(text):
        if _KEEP.search(ch) is None:
            kept.append(ch)
            idx_map.append(i)
    return "".join(kept), idx_map


def align_ops(ref: str, hyp: str) -> list[tuple[str, int | None, int | None]]:
    """Levenshtein 回溯，返回逐操作对齐 (op, ref_idx, hyp_idx)。

    op ∈ {C, S, D, I}：D=参考有假设无（漏识别），I=假设多出（插入）。
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri, row, prev = ref[i - 1], dp[i], dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            ops.append(("C" if ref[i - 1] == hyp[j - 1] else "S", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("D", i - 1, None))
            i -= 1
        else:
            ops.append(("I", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def apply_filler_rules(ops: list[tuple], ref: str, hyp: str) -> tuple[list[tuple], int]:
    """标签规则 v2：语气词互换/插入/漏识别不计为错误。

    依据：参考文本为人工转写，语气词记录风格不稳定（"嗯" vs "呃"）；
    项目 CER 口径（eval_cer._FILL）本就剥离语气词。规则：
      - S 且双方均为语气词 -> 记 C
      - I 且假设字符为语气词 -> 记 C（参考未转写语气词）
      - D 且参考字符为语气词 -> 丢弃事件
    返回 (调整后 ops, 被忽略的语气词事件数)。
    """
    adj, ignored = [], 0
    for op, ri, hi in ops:
        if op == "S" and ri is not None and hi is not None \
                and ref[ri] in FILLER and hyp[hi] in FILLER:
            adj.append(("C", ri, hi))
            ignored += 1
        elif op == "I" and hi is not None and hyp[hi] in FILLER:
            adj.append(("C", None, hi))
            ignored += 1
        elif op == "D" and ri is not None and ref[ri] in FILLER:
            ignored += 1
        else:
            adj.append((op, ri, hi))
    return adj, ignored


def load_model(model_path: str, device: str):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path, dtype=dtype, local_files_only=True, trust_remote_code=False)
    model.to(device)
    model.eval()
    return processor, model


def transcribe_with_trace(processor, model, audio_path: str, device: str) -> dict:
    """Greedy + scores：返回 {text, tokens: [{...特征+字符区间...}], hit_limit}。"""
    import torch

    inputs = processor.apply_transcription_request(audio=str(audio_path), language="Chinese")
    inputs = inputs.to(device, model.dtype)
    input_len = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            return_dict_in_generate=True, output_scores=True)
    wall_ms = (time.perf_counter() - t0) * 1000

    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    gen_ids = out.sequences[0, input_len:].tolist()
    raw_text = tok.decode(gen_ids, skip_special_tokens=True)
    text = processor.decode(out.sequences[:, input_len:], return_format="transcription_only")[0]

    tokens, prev_len = [], 0
    for step, (tid, score_row) in enumerate(zip(gen_ids, out.scores)):
        # 设备上算 log_softmax，只回传标量（P0 结论：scores 与 raw logits 等价）
        logp = torch.log_softmax(score_row[0].to(torch.float32), dim=-1)
        sel_lp = float(logp[tid].item())
        top2 = torch.topk(logp, k=2)
        p = logp.exp()
        prefix = tok.decode(gen_ids[: step + 1], skip_special_tokens=True)
        token_text = tok.decode([tid], skip_special_tokens=True)
        if not token_text:  # <|im_end|> 等特殊 token：不计入特征
            continue
        tokens.append({
            "step": step,
            "token_id": tid,
            "token_text": token_text,
            "char_start": prev_len,
            "char_end": len(prefix),
            "token_logprob": sel_lp,
            "max_prob": float(p.max().item()),
            "entropy_gibbs_nats": float(-(p * logp).sum().item()),
            "top2_margin_logprob": float((top2.values[0] - top2.values[1]).item()),
            "token_rank": int((logp > logp[tid]).sum().item()) + 1,
        })
        prev_len = len(prefix)
    hit_limit = len(gen_ids) >= MAX_NEW_TOKENS
    return {"text": text, "raw_text": raw_text, "tokens": tokens,
            "hit_limit": hit_limit, "wall_ms": round(wall_ms, 1),
            "gen_tokens": len(gen_ids)}


def build_char_features(tokens: list[dict], norm_hyp: str, raw_idx_map: list[int]) -> list[dict]:
    """字符级特征：每个归一化假设字符回溯到 raw 下标 -> 覆盖它的 token -> 聚合。"""
    n = len(norm_hyp)
    if n == 0 or not tokens:
        return []
    # token 的 raw 字符区间 -> 归一化字符区间（raw_idx_map 单调，线性扫描）
    tok_norm = []
    for t in tokens:
        start = None
        end = None
        for k, ri in enumerate(raw_idx_map):
            if ri >= t["char_start"] and start is None:
                start = k
            if ri < t["char_end"]:
                end = k + 1
        if start is not None and end is not None and end > start:
            tok_norm.append((start, end, t))
    chars = [{"char": norm_hyp[j], "token_logprobs": [], "entropies": [], "margins": [],
              "max_probs": [], "ranks": []} for j in range(n)]
    for start, end, t in tok_norm:
        for j in range(start, end):
            c = chars[j]
            c["token_logprobs"].append(t["token_logprob"])
            c["entropies"].append(t["entropy_gibbs_nats"])
            c["margins"].append(t["top2_margin_logprob"])
            c["max_probs"].append(t["max_prob"])
            c["ranks"].append(t["token_rank"])
    out = []
    for j, c in enumerate(chars):
        lp = c["token_logprobs"]
        out.append({
            "char_idx": j,
            "char": c["char"],
            "label": None,  # 由 align_ops 填
            "token_logprob": min(lp) if lp else None,   # 字符取最差覆盖 token（保守）
            "mean_token_logprob": (sum(lp) / len(lp)) if lp else None,
            "entropy_gibbs_nats": max(c["entropies"]) if c["entropies"] else None,
            "top2_margin_logprob": min(c["margins"]) if c["margins"] else None,
            "max_prob": min(c["max_probs"]) if c["max_probs"] else None,
            "token_rank": max(c["ranks"]) if c["ranks"] else None,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-ASR-1.7B-hf")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（0=全部）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    import torch
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader(open(LIPS_META))
            if (LIPS_WAV / f"{r['ID']}.wav").exists()]
    speakers: dict[str, list[str]] = {}
    for r in rows:
        speakers.setdefault(r["ID"].rsplit("_", 1)[0], []).append(r["ID"])
    spk_list = sorted(speakers)
    rng = random.Random(args.seed)
    rng.shuffle(spk_list)
    split_of_speaker = {}
    pos = 0
    for name, count in SPLIT_PLAN.items():
        for s in spk_list[pos:pos + count]:
            split_of_speaker[s] = name
        pos += count
    split = {s: split_of_speaker[s] for s in sorted(spk_list)}
    (out_dir / "dataset_split.json").write_text(
        json.dumps({"seed": args.seed, "split_by": "speaker_group",
                    "speaker_split": split,
                    "counts": {k: sum(len(speakers[s]) for s in spk_list
                                      if split_of_speaker[s] == k)
                               for k in SPLIT_PLAN}}, ensure_ascii=False, indent=2))

    if args.limit:
        # 跨说话人轮询抽样，避免 meta 按说话人排序导致 limit 截断破坏 split 覆盖
        by_spk: dict[str, list[dict]] = {}
        for r in rows:
            by_spk.setdefault(r["ID"].rsplit("_", 1)[0], []).append(r)
        picked: list[dict] = []
        pools = [list(v) for v in by_spk.values()]
        while pools and len(picked) < args.limit:
            for pool in list(pools):
                if pool:
                    picked.append(pool.pop(0))
                if not pool:
                    pools.remove(pool)
                if len(picked) >= args.limit:
                    break
        rows = picked

    processor, model = load_model(str(PROJECT_ROOT / args.model), device)
    import eval_cer  # noqa: E402 - scripts/ 下的现成归一化 + S/D/I 统计

    master = out_dir / "utterances.jsonl"
    manifest_rows = []
    stats = {"n": 0, "chars": 0, "S": 0, "I": 0, "D": 0, "err_utt": 0,
             "wall_s": 0.0, "audio_s": 0.0, "empty": 0, "hit_limit": 0,
             "filler_ignored": 0}
    t_start = time.perf_counter()
    with open(master, "w") as fm, open(out_dir / "transcripts_reference.jsonl", "w") as fr, \
            open(out_dir / "asr_hypotheses.jsonl", "w") as fh, \
            open(out_dir / "char_alignment.jsonl", "w") as fa:
        for k, r in enumerate(rows):
            uid = r["ID"]
            wav = LIPS_WAV / f"{uid}.wav"
            spk = uid.rsplit("_", 1)[0]
            try:
                rec = transcribe_with_trace(processor, model, wav, device)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {uid}: {type(e).__name__}: {e}", flush=True)
                continue
            hyp_raw = rec["raw_text"]
            norm_hyp, hyp_map = conservative_norm(hyp_raw)
            norm_ref, _ = conservative_norm(r["TEXT"])

            ops = align_ops(norm_ref, norm_hyp) if norm_hyp else [("D", i, None) for i in range(len(norm_ref))]
            ops, filler_ignored = apply_filler_rules(ops, norm_ref, norm_hyp)
            labels = [None] * len(norm_hyp)
            deletions = []
            for op, ri, hi in ops:
                if op == "D":
                    deletions.append({"ref_char_idx": ri, "ref_char": norm_ref[ri]})
                elif hi is not None:
                    labels[hi] = 1 if op == "C" else 0

            char_features = build_char_features(rec["tokens"], norm_hyp, hyp_map)
            for cf, lab in zip(char_features, labels):
                cf["label"] = lab

            s_count = sum(1 for op, _, _ in ops if op == "S")
            i_count = sum(1 for op, _, _ in ops if op == "I")
            d_count = len(deletions)
            cer_cons = (s_count + i_count + d_count) / max(len(norm_ref), 1)
            # 数字归一口径的段级 CER（沿用项目统一口径）
            n_ref_full = eval_cer.normalize(r["TEXT"])
            n_hyp_full = eval_cer.normalize(hyp_raw)
            sub2, dele2, ins2, dist2 = eval_cer.edit_stats(n_ref_full, n_hyp_full)
            cer_itn = dist2 / max(len(n_ref_full), 1)

            record = {
                "recording_id": uid,
                "speaker_id": spk,
                "split": split_of_speaker[spk],
                "audio": str(wav.relative_to(PROJECT_ROOT)),
                "ref_text": r["TEXT"],
                "hyp_text_raw": hyp_raw,
                "hyp_text_display": rec["text"],
                "norm_ref": norm_ref,
                "norm_hyp": norm_hyp,
                "token_count": len(rec["tokens"]),
                "gen_tokens": rec["gen_tokens"],
                "hit_limit": rec["hit_limit"],
                "wall_ms": rec["wall_ms"],
                "label": {
                    "rule": LABEL_RULE,
                    "has_error": int(s_count + i_count + d_count > 0),
                    "char_error_rate": cer_cons,
                    "cer_itn_norm": cer_itn,
                    "S": s_count, "I": i_count, "D": d_count,
                    "filler_ignored": filler_ignored,
                },
                "char_features": char_features,
                "deletion_events": deletions,
            }
            fm.write(json.dumps(record, ensure_ascii=False) + "\n")
            fr.write(json.dumps({"recording_id": uid, "ref_text": r["TEXT"]}, ensure_ascii=False) + "\n")
            fh.write(json.dumps({"recording_id": uid, "hyp_text": hyp_raw,
                                 "wall_ms": rec["wall_ms"]}, ensure_ascii=False) + "\n")
            fa.write(json.dumps({"recording_id": uid, "ops": [
                {"op": op, "ref": ri, "hyp": hi} for op, ri, hi in ops],
                "S": s_count, "I": i_count, "D": d_count}, ensure_ascii=False) + "\n")

            dur = float(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(wav)],
                capture_output=True, text=True).stdout.strip() or 0)
            manifest_rows.append({"recording_id": uid, "speaker_id": spk, "split": split_of_speaker[spk],
                                  "duration_s": round(dur, 2), "ref_chars": len(norm_ref),
                                  "S": s_count, "I": i_count, "D": d_count,
                                  "cer": round(cer_cons, 4), "wall_ms": rec["wall_ms"]})

            stats["n"] += 1
            stats["chars"] += len(norm_ref)
            stats["S"] += s_count
            stats["I"] += i_count
            stats["D"] += d_count
            stats["filler_ignored"] += filler_ignored
            stats["err_utt"] += int(s_count + i_count + d_count > 0)
            stats["wall_s"] += rec["wall_ms"] / 1000
            stats["audio_s"] += dur
            stats["hit_limit"] += int(rec["hit_limit"])
            if not norm_hyp:
                stats["empty"] += 1
            if (k + 1) % 50 == 0:
                el = time.perf_counter() - t_start
                print(f"[p1] {k+1}/{len(rows)} 条 | CER 累计 {(stats['S']+stats['I']+stats['D'])/max(stats['chars'],1)*100:.2f}% "
                      f"| 推理 {stats['wall_s']:.0f}s / 音频 {stats['audio_s']:.0f}s | ETA {el/(k+1)*(len(rows)-k-1)/60:.0f}min", flush=True)

    with open(out_dir / "recordings_manifest.csv", "w", newline="") as fc:
        w = csv.DictWriter(fc, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    total_err = stats["S"] + stats["I"] + stats["D"]
    report = f"""# P1 数据质量报告（LiPS 置信度实验数据集）

- 生成时间：{time.time():.0f} · seed={args.seed} · limit={args.limit or '全部'}
- 模型：{args.model} · device={device} · max_new_tokens={MAX_NEW_TOKENS}

## 规模

| 指标 | 值 |
|---|---|
| 语句数 | {stats['n']} |
| 音频总时长 | {stats['audio_s']/3600:.2f} h |
| 参考字符数（保守口径） | {stats['chars']} |
| 含错误语句数 | {stats['err_utt']} ({stats['err_utt']/max(stats['n'],1)*100:.1f}%) |

## 错误分布（保守归一化：仅剥标点，不做数字转换）

| 类型 | 数量 | 占比 |
|---|---|---|
| S 替换 | {stats['S']} | {stats['S']/max(total_err,1)*100:.1f}% |
| I 插入 | {stats['I']} | {stats['I']/max(total_err,1)*100:.1f}% |
| D 漏识别 | {stats['D']} | {stats['D']/max(total_err,1)*100:.1f}% |
| 合计 CER | — | {(total_err)/max(stats['chars'],1)*100:.2f}% |

## 链路健康度

- 空转写：{stats['empty']} 条
- 语气词事件（v2 规则忽略，不计错误）：{stats['filler_ignored']} 个
- 触顶（max_new_tokens 截断）：{stats['hit_limit']} 条
- 推理总耗时：{stats['wall_s']:.0f}s（RTF={stats['wall_s']/max(stats['audio_s'],1):.3f}）

## 已知口径说明

1. 字符级标签用**保守归一化**（仅剥标点/空白，保留语气词与数字原样），
   保证 raw 字符 -> 归一化字符 1:1 映射、特征-标签可对齐；
   数字书写差异（如 "1.5" vs "一点五"）在此口径下计为替换错误。
2. **标签规则 v2（{LABEL_RULE}）**：语气词（{"".join(sorted(FILLER))}）互换/插入/漏识别
   不计为错误——人工转写对语气词的记录风格不稳定，且项目 CER 口径本就剥离语气词。
   每条记录的 label.filler_ignored 保留被忽略的事件数。
2. 段级 `cer_itn_norm` 同时按项目统一口径（eval_cer 数字归一）计算，
   两口径差异即数字/虚词风格差，报告在 utterances.jsonl 中并存。
3. 同一说话人的所有语句在同一 split（train/calibration/test），无跨集合泄漏。
"""
    (out_dir / "data_quality_report.md").write_text(report)
    print(f"[p1] 完成：{stats['n']} 条，总 CER {(total_err)/max(stats['chars'],1)*100:.2f}%，"
          f"含错语句 {stats['err_utt']}，输出 {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
