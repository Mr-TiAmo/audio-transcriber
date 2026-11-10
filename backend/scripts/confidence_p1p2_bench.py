#!/usr/bin/env python3
"""P1（覆盖率控制/文本侧/对齐异常/段级排序）与 P2（beam rank/采样一致性）离线实验。

对 bench_10min 的 10 个 chunk 重新做 greedy / beam-4 / 采样 ×2 / ForcedAligner，
把多路信号映射到字符级，与 LiPS 参考逐字符对账，输出：
  - 单特征 AUROC / AP
  - 融合后的覆盖率-精确率曲线（1% / 2% / 4% / 8% / 10%）
  - 与当前生产策略（无全局预算）的对照
  - 段级风险排序的含错段召回

用法::

    python scripts/confidence_p1p2_bench.py --stage cache   # 推理并缓存（几分钟）
    python scripts/confidence_p1p2_bench.py --stage eval    # 读缓存评估

产物：output/confidence/p1p2/{features.json,metrics.json,report.md}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from confidence_build_dataset import (  # noqa: E402
    align_ops, apply_filler_rules, conservative_norm,
)
from confidence_eval_p2 import average_precision, auroc  # noqa: E402
from local_asr.asr.segmentation import keep_char  # noqa: E402
from qwen3_asr_transcribe import (  # noqa: E402
    _char_feature_series, _prepare_asr_inputs, _token_char_spans, load_model,
)

DATA = PROJECT_ROOT.parent
STATS = DATA / "data/bench/analysis4/bench_10min_raw.stats.json"
WAV = DATA / "data/bench/bench_10min.wav"
REF = DATA / "data/bench/bench_10min_reference.txt"
MODEL = DATA / "models/Qwen3-ASR-0.6B"
ALIGNER = DATA / "models/Qwen3-ForcedAligner-0.6B"
OUT = PROJECT_ROOT / "output/confidence/p1p2"
EOS_IDS = (151645, 151643)

# 与生产策略保持一致的门控参数
MIN_ENTROPY = 0.2
SEGMENT_MAX_ENTROPY = 0.7


# ---------------------------------------------------------------------------
# 音频与模型
# ---------------------------------------------------------------------------

def slice_wav(src: Path, dst: Path, start_s: float, end_s: float) -> None:
    """按秒切 PCM16 16k 单声道 WAV（bench 音频固定格式）。"""
    with wave.open(str(src), "rb") as r:
        assert r.getframerate() == 16000 and r.getnchannels() == 1, "bench 音频须为 16k 单声道"
        r.setpos(int(start_s * 16000))
        data = r.readframes(int((end_s - start_s) * 16000))
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(data)


def _decode_text(processor, row) -> str:
    raw = processor.batch_decode(
        row.unsqueeze(0), skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]
    if "<asr_text>" in raw:
        head, _, raw = raw.partition("<asr_text>")
        if "language none" in head.lower():
            return ""
    return raw


def _generate(processor, model, wav: Path, *, beams: int = 0,
              samples: int = 0, max_new_tokens: int = 512):
    """统一入口：greedy / beam / sampling。返回 (texts, token 特征, 原始 output)。"""
    import torch

    texts, wavs = _prepare_asr_inputs(processor, [str(wav)], None, "Chinese")
    inputs = processor(text=texts, audio=wavs, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    input_len = inputs["input_ids"].shape[1]
    kwargs = {"max_new_tokens": max_new_tokens, "output_scores": True}
    if beams:
        kwargs.update(num_beams=beams, num_return_sequences=beams, do_sample=False)
    elif samples:
        kwargs.update(do_sample=True, temperature=0.7, top_p=0.9,
                      num_return_sequences=samples)
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    gen = out.sequences[:, input_len:]
    texts = [_decode_text(processor, row) for row in gen]
    tok = getattr(processor, "tokenizer", None)
    return texts, gen, out, input_len, tok


def _greedy_token_features(out, gen_row, tok) -> list[dict]:
    """greedy 逐 token 基础特征（与主脚本同口径）。"""
    import torch

    tids = []
    for t in range(len(gen_row)):
        tid = int(gen_row[t])
        if tid in EOS_IDS:
            break
        tids.append(tid)
    feats = []
    for t, tid in enumerate(tids):
        logp = torch.log_softmax(out.scores[t][0].float(), dim=-1)
        sel = float(logp[tid])
        p = logp.exp()
        ent = float(-(p * logp).sum())
        top2 = torch.topk(logp, k=2).values
        feats.append({
            "entropy": ent,
            "neg_logprob": -sel,
            "neg_margin": -float((top2[0] - top2[1]).item()),
            "token_len": 1,  # 占位，后续按字符跨度回填
        })
    # 按字符跨度回填 token 长度
    return feats


def _beam_token_features(out, gen_row, beam_row: int, tok) -> list[dict]:
    """beam 候选 beam_row 的 token 特征（SR-CEM 风格）。

    依据：qwen-asr 0.0.6 下 out.beam_indices 与生成步一一对应（不含 prompt），
    其值为 scores[t] 的父 beam 行号。local 分布 = log_softmax(scores[t][parent])。
    """
    import torch

    tids = []
    for t in range(len(gen_row)):
        tid = int(gen_row[t])
        if tid in EOS_IDS:
            break
        tids.append(tid)
    beam_idx = out.beam_indices[beam_row].tolist() if out.beam_indices is not None else []
    n_beams = out.scores[0].shape[0]
    feats = []
    for t, tid in enumerate(tids):
        parent = beam_idx[t] if t < len(beam_idx) else beam_row
        if parent < 0 or parent >= n_beams:
            parent = beam_row
        local = torch.log_softmax(out.scores[t][parent].float(), dim=-1)
        sel = float(local[tid])
        rank = int((local > local[tid]).sum().item()) + 1
        top2 = torch.topk(local, k=2).values
        p = local.exp()
        feats.append({
            "entropy": float(-(p * local).sum()),
            "neg_logprob": -sel,
            "neg_margin": -float((top2[0] - top2[1]).item()),
            "beam_rank": float(rank),
            "beam_top1": float(top2[0]),
            "token_len": 1,
        })
    return feats


def _attach_token_len(feats: list[dict], text: str, tok) -> None:
    """用重新 tokenize 的跨度回填每个 token 的字符长度。"""
    if not feats or tok is None:
        return
    spans = _token_char_spans(tuple(feats), text, tok)
    for (s, e, f) in spans:
        kept = sum(1 for i in range(s, min(e, len(text))) if keep_char(text[i]))
        f["token_len"] = kept


# ---------------------------------------------------------------------------
# 对齐 / 文本侧 / 采样一致性
# ---------------------------------------------------------------------------

def _aligner_kept(ch: str) -> bool:
    import unicodedata
    return ch == "'" or unicodedata.category(ch)[0] in ("L", "N")


def align_char_features(aligner, wav: Path, text: str) -> list[dict]:
    """对文本做强制对齐，返回原文每个字符的时长/重叠特征（无对齐则 None）。"""
    out: list[dict | None] = [None] * len(text)
    try:
        res = aligner.align(str(wav), text, "Chinese")[0]
    except Exception as exc:  # noqa: BLE001
        print(f"[align] 失败: {exc}", flush=True)
        return out
    items = list(res.items)
    kept_pos = [i for i, ch in enumerate(text) if _aligner_kept(ch)]
    # 把 item.text 展开成字符，与 kept_pos 顺序对应
    seq: list[tuple[int, float, float]] = []
    for it in items:
        for ch in it.text:
            if seq and len(seq) >= len(kept_pos):
                break
            if len(seq) < len(kept_pos):
                seq.append((kept_pos[len(seq)], float(it.start_time), float(it.end_time)))
    durs = [max(0.0, e - s) for _, s, e in seq]
    med = float(np.median([d for d in durs if d > 0])) if durs else 0.0
    mad = float(np.median([abs(d - med) for d in durs])) if durs else 0.0
    prev_end = None
    for (pos, s, e) in seq:
        dur = max(0.0, e - s)
        z = (dur - med) / (mad + 1e-6) if mad else 0.0
        overlap = max(0.0, prev_end - s) if prev_end is not None else 0.0
        out[pos] = {
            "align_dur": dur,
            "align_dur_z": z,
            "align_overlap": overlap,
            "align_short": int(dur < 0.06 or (med and dur < 0.25 * med)),
            "align_long": int(dur > 1.2 or (med and dur > 3.5 * med)),
        }
        prev_end = e
    return out


PUNCT = set("，。！？；：、,.!?;:…—-()（）《》\"'“”‘’")
LATIN_DIGIT = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def text_char_features(text: str) -> list[dict]:
    """文本侧启发式：重复片段、连续标点、长拉丁/数字串、段尾位置。"""
    n = len(text)
    out = [{"text_repeat": 0, "text_punct_run": 0,
            "text_latin_run": 0, "text_tail": 0} for _ in range(n)]
    for i in range(n):
        if i >= 0.9 * n and n > 10:
            out[i]["text_tail"] = 1
        # 连续标点
        if text[i] in PUNCT:
            j = i
            while j + 1 < n and text[j + 1] in PUNCT:
                j += 1
            if j > i:
                for k in range(i, j + 1):
                    out[k]["text_punct_run"] = 1
        # 长拉丁/数字串
        if text[i] in LATIN_DIGIT:
            j = i
            while j + 1 < n and text[j + 1] in LATIN_DIGIT:
                j += 1
            if j - i + 1 >= 4:
                for k in range(i, j + 1):
                    out[k]["text_latin_run"] = 1
        # 相邻重复：k=1..3
        for k in (1, 2, 3):
            if i + 2 * k <= n and text[i:i + k] == text[i + k:i + 2 * k]:
                for p in range(i, i + 2 * k):
                    out[p]["text_repeat"] = 1
    return out


def sample_agreement(sample_texts: list[str], text: str) -> list[float | None]:
    """greedy 每个字符被采样候选支持的平均比例（C 视为支持）。"""
    flags = np.zeros(len(text), dtype=float)
    valid = 0
    for cand in sample_texts:
        if not cand:
            continue
        ops = align_ops(cand, text)
        hit = [False] * len(text)
        for op, ri, hi in ops:
            if hi is not None and op == "C":
                hit[hi] = True
        flags += np.array([1.0 if h else 0.0 for h in hit])
        valid += 1
    if not valid:
        return [None] * len(text)
    return [float(x / valid) for x in flags]


# ---------------------------------------------------------------------------
# 标签
# ---------------------------------------------------------------------------

def label_segments(segments: list[dict], ref_text: str) -> None:
    """全局拼接对齐后，把 C/S/I 标签写回每段每个原文索引。"""
    parts, mapping = [], []
    for si, seg in enumerate(segments):
        norm, idx_map = conservative_norm(seg["text"])
        for ni, orig in enumerate(idx_map):
            mapping.append((si, orig))
        parts.append(norm)
    hyp = "".join(parts)
    ref_norm, _ = conservative_norm(ref_text)
    ops = align_ops(ref_norm, hyp)
    ops, _ = apply_filler_rules(ops, ref_norm, hyp)
    labels: dict[tuple[int, int], int] = {}
    for op, ri, hi in ops:
        if hi is None:
            continue
        labels[mapping[hi]] = 1 if op == "C" else 0
    for si, seg in enumerate(segments):
        seg["labels"] = [labels.get((si, i)) for i in range(len(seg["text"]))]
        # 标签回填到 chars（collect 与 eval 两处都可调用，幂等）
        for c, lab in zip(seg.get("chars", []), seg["labels"]):
            c["label"] = lab


# ---------------------------------------------------------------------------
# 采集
# ---------------------------------------------------------------------------

def collect(limit_segments: int = 0) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    raw_segments = stats["segments"]
    if limit_segments:
        raw_segments = raw_segments[:limit_segments]
    segments = [{"seq": i, "text": s["text"], "start_s": s["start_s"],
                 "end_s": s["end_s"]} for i, s in enumerate(raw_segments)]

    print(f"[cache] 段数 {len(segments)}，加载模型 ...", flush=True)
    processor, model = load_model(str(MODEL), "cuda")
    tok = getattr(processor, "tokenizer", None)
    aligner = None
    if ALIGNER.is_dir():
        try:
            from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner
            import torch
            aligner = Qwen3ForcedAligner.from_pretrained(
                str(ALIGNER), dtype=torch.float16, local_files_only=True)
            print("[cache] ForcedAligner 已加载", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[cache] ForcedAligner 加载失败（跳过对齐特征）: {exc}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for si, seg in enumerate(segments):
        wav = OUT / f"seg_{si:02d}.wav"
        slice_wav(WAV, wav, seg["start_s"], seg["end_s"])
        text = seg["text"]

        # greedy
        gtexts, ggen, gout, _, _ = _generate(processor, model, wav)
        gtext = gtexts[0]
        if gtext.strip() != text.strip():
            # 重跑文本与 stats 不一致时，以重跑文本为准（同输入同参数应一致）
            print(f"[cache] 段{si} greedy 文本与 stats 不一致，以重跑为准", flush=True)
        gfeats = _greedy_token_features(gout, ggen[0], tok)
        _attach_token_len(gfeats, gtext, tok)

        # beam-4
        btexts, bgen, bout, _, _ = _generate(processor, model, wav, beams=4)
        bfeats = _beam_token_features(bout, bgen[0], 0, tok)
        _attach_token_len(bfeats, btexts[0], tok)

        # 采样 ×2
        stexts, _, _, _, _ = _generate(processor, model, wav, samples=2)

        # 对齐
        afeats = align_char_features(aligner, wav, gtext) if aligner else [None] * len(gtext)

        # 字符级合并
        chars = []
        ent = dict(_char_feature_series(tuple(gfeats), gtext, tok, "entropy"))
        nlp = dict(_char_feature_series(tuple(gfeats), gtext, tok, "neg_logprob"))
        nmg = dict(_char_feature_series(tuple(gfeats), gtext, tok, "neg_margin"))
        # token 长度：单字 token 标记
        single = {}
        for s, e, f in _token_char_spans(tuple(gfeats), gtext, tok):
            kept = [i for i in range(s, min(e, len(gtext))) if keep_char(gtext[i])]
            if len(kept) == 1:
                single[kept[0]] = 1
        # beam 特征映射：beam-0 文本与 greedy 一致时直接按索引，否则按对齐
        beam_map: dict[int, dict] = {}
        if btexts[0].strip() == gtext.strip():
            bent = dict(_char_feature_series(tuple(bfeats), btexts[0], tok, "entropy"))
            brank = dict(_char_feature_series(tuple(bfeats), btexts[0], tok, "beam_rank"))
            btop = dict(_char_feature_series(tuple(bfeats), btexts[0], tok, "beam_top1"))
            for pos in bent:
                if pos < len(gtext):
                    beam_map[pos] = {"beam_entropy": bent[pos],
                                     "beam_rank": brank.get(pos, 1.0),
                                     "beam_top1": btop.get(pos, 0.0)}
        # 采样一致性
        sagree = sample_agreement(stexts, gtext)
        # 文本侧
        tfeats = text_char_features(gtext)

        for i in range(len(gtext)):
            af = afeats[i] if i < len(afeats) else None
            chars.append({
                "idx": i, "ch": gtext[i],
                "entropy": ent.get(i),
                "neg_logprob": nlp.get(i),
                "neg_margin": nmg.get(i),
                "single_token": single.get(i, 0),
                "beam_entropy": beam_map.get(i, {}).get("beam_entropy"),
                "beam_rank": beam_map.get(i, {}).get("beam_rank"),
                "beam_top1": beam_map.get(i, {}).get("beam_top1"),
                "sample_agree": sagree[i] if i < len(sagree) else None,
                **tfeats[i],
                **(af or {}),
            })
        seg["chars"] = chars
        # 标签按重跑文本生成，必须让 seg["text"] 与 chars 对齐
        # （stats 文本来自带 overlap 的音频，边界字可能不同）
        seg["text"] = gtext
        seg["stats_text"] = text
        seg["greedy_text"] = gtext
        seg["beam_texts"] = btexts
        seg["sample_texts"] = stexts
        seg["beam_same"] = int(btexts[0].strip() == gtext.strip())
        seg["align_ok"] = int(sum(1 for c in chars if c.get("align_dur") is not None))
        print(f"[cache] 段{si}: {len(gtext)} 字，beam 一致={seg['beam_same']}，"
              f"对齐={seg['align_ok']}", flush=True)
        del ggen, gout, bgen, bout

    label_segments(segments, REF.read_text(encoding="utf-8"))
    OUT.joinpath("features.json").write_text(
        json.dumps({"segments": segments, "params": {
            "min_entropy": MIN_ENTROPY, "segment_max_entropy": SEGMENT_MAX_ENTROPY,
        }}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cache] 写入 {OUT/'features.json'}", flush=True)


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

def _z(vals: np.ndarray) -> np.ndarray:
    sd = float(vals.std())
    return (vals - float(vals.mean())) / (sd + 1e-9)


def _rank01(vals: np.ndarray) -> np.ndarray:
    order = vals.argsort(kind="mergesort")
    out = np.empty(len(vals), dtype=float)
    out[order] = np.arange(len(vals)) / max(len(vals) - 1, 1)
    return out


def eval_rows(segments: list[dict]) -> tuple[list[dict], list[dict]]:
    """返回 (字符行, 段行)。字符行只保留可判定标签的信息字符。"""
    rows, seg_rows = [], []
    for seg in segments:
        seg_rows.append({
            "seq": seg["seq"], "n": len(seg["chars"]),
            "errors": sum(1 for c in seg["chars"] if c.get("label") == 0),
            "max_single_entropy": max(
                [c["entropy"] for c in seg["chars"]
                 if c.get("single_token") and c.get("entropy") is not None] or [0.0]),
            "mean_entropy": float(np.mean(
                [c["entropy"] for c in seg["chars"] if c.get("entropy") is not None] or [0.0])),
            "max_entropy": max(
                [c["entropy"] for c in seg["chars"] if c.get("entropy") is not None] or [0.0]),
            "text_anomaly": sum(1 for c in seg["chars"]
                                if c.get("text_repeat") or c.get("text_punct_run")
                                or c.get("text_latin_run")),
        })
        for c in seg["chars"]:
            if c.get("label") is None or c.get("entropy") is None:
                continue
            rows.append({"rec": seg["seq"], **c, "err": 1 - c["label"]})
    return rows, seg_rows


def summarize(scores: np.ndarray, errs: np.ndarray) -> dict:
    return {"AUROC": auroc(scores, errs), "AP": average_precision(scores, errs)}


def policy_metrics(rows: list[dict], score_key: str, ratio: float,
                   *, require_single: bool = True, absolute: float | None = MIN_ENTROPY,
                   gate: bool = True) -> dict:
    """模拟生产策略：过滤 → 段级门控 → 全局 top ratio 预算 → P/R。"""
    by_seg: dict[int, list[dict]] = {}
    for r in rows:
        by_seg.setdefault(r["rec"], []).append(r)
    cands: list[tuple[float, dict]] = []
    for rec, rs in by_seg.items():
        singles = [r for r in rs if r.get("single_token")]
        max_ent = max([r["entropy"] for r in singles] or [0.0])
        if gate and max_ent < SEGMENT_MAX_ENTROPY:
            continue
        for r in rs:
            if require_single and not r.get("single_token"):
                continue
            s = r.get(score_key)
            if s is None:
                continue
            if absolute is not None and r["entropy"] < absolute:
                continue
            cands.append((float(s), r))
    cands.sort(key=lambda x: x[0], reverse=True)
    n_errors = sum(1 for r in rows if r["err"] == 1)
    k = max(1, int(math.ceil(ratio * len(rows))))
    picked = cands[:k]
    tp = sum(1 for _, r in picked if r["err"] == 1)
    return {"marked": len(picked), "tp": tp,
            "precision": tp / max(len(picked), 1),
            "recall": tp / max(n_errors, 1),
            "k": k, "n_errors": n_errors, "n_rows": len(rows)}


def evaluate() -> None:
    data = json.loads(OUT.joinpath("features.json").read_text(encoding="utf-8"))
    segments = data["segments"]
    # 缓存可能早于标签回填逻辑：这里重算一次（幂等，不依赖推理）
    label_segments(segments, REF.read_text(encoding="utf-8"))
    rows, seg_rows = eval_rows(segments)
    errs = np.array([r["err"] for r in rows], dtype=float)
    print(f"[eval] 字符 {len(rows)}，错误 {int(errs.sum())}，"
          f"错误率 {errs.mean()*100:.2f}%", flush=True)

    # 派生分数（方向统一为「越高越可疑」）
    for r in rows:
        r["beam_rank_s"] = (r.get("beam_rank") or 1.0)
        r["beam_entropy_s"] = (r.get("beam_entropy") or 0.0)
        r["neg_sample_agree"] = 1.0 - (r["sample_agree"] if r.get("sample_agree") is not None else 1.0)
        r["align_short_s"] = float(r.get("align_short") or 0)
        r["align_long_s"] = float(r.get("align_long") or 0)
        r["align_dur_absz"] = abs(float(r.get("align_dur_z") or 0.0))
        r["text_risk"] = float((r.get("text_repeat") or 0)
                               or (r.get("text_punct_run") or 0)
                               or (r.get("text_latin_run") or 0))

    base_feats = {
        "entropy": "entropy", "neg_logprob": "neg_logprob", "neg_margin": "neg_margin",
        "beam_rank": "beam_rank_s", "beam_entropy": "beam_entropy_s",
        "neg_sample_agree": "neg_sample_agree",
        "align_short": "align_short_s", "align_long": "align_long_s",
        "align_dur_absz": "align_dur_absz", "text_risk": "text_risk",
    }
    metrics: dict = {"n": len(rows), "errors": int(errs.sum()),
                     "prevalence": float(errs.mean()), "single": {}}
    for name, key in base_feats.items():
        vals = np.array([float(r.get(key) or 0.0) for r in rows])
        if np.all(vals == vals[0]):
            continue
        metrics["single"][name] = summarize(vals, errs)

    # 融合：rank 平均（所有特征按全片 rank01）
    def combo(keys: list[str], name: str) -> None:
        parts = []
        for k in keys:
            v = np.array([float(r.get(k) or 0.0) for r in rows])
            parts.append(_rank01(v))
        score = np.mean(parts, axis=0)
        for r, s in zip(rows, score):
            r[f"_combo_{name}"] = float(s)
        metrics.setdefault("combo", {})[name] = summarize(score, errs)

    combo(["entropy"], "entropy")
    combo(["entropy", "beam_rank_s"], "entropy+beam_rank")
    combo(["entropy", "neg_sample_agree"], "entropy+sample")
    combo(["entropy", "align_dur_absz"], "entropy+align")
    combo(["entropy", "text_risk"], "entropy+text")
    combo(["entropy", "beam_rank_s", "neg_sample_agree", "align_dur_absz", "text_risk"],
          "full")

    # 决策级：覆盖率曲线（生产策略口径）
    # 两组候选池：
    #   strict  = 单字 token 且 entropy>=0.2（当前生产策略，候选常不足 4%）
    #   relaxed = 单字 token、不设熵下限（用融合分数决定，吃满预算）
    metrics["policy"] = {}
    for ratio in (0.01, 0.02, 0.04, 0.08, 0.10):
        metrics["policy"][f"top{int(ratio*100)}%"] = {
            "strict:entropy": policy_metrics(rows, "entropy", ratio),
            "strict:entropy+sample": policy_metrics(rows, "_combo_entropy+sample", ratio),
            "relaxed:entropy": policy_metrics(rows, "entropy", ratio, absolute=0.0),
            "relaxed0.1:entropy": policy_metrics(rows, "entropy", ratio, absolute=0.1),
            "relaxed0.15:entropy": policy_metrics(rows, "entropy", ratio, absolute=0.15),
            "relaxed:entropy+sample": policy_metrics(
                rows, "_combo_entropy+sample", ratio, absolute=0.0),
            "relaxed:entropy+align": policy_metrics(
                rows, "_combo_entropy+align", ratio, absolute=0.0),
            "relaxed:full": policy_metrics(rows, "_combo_full", ratio, absolute=0.0),
            "relaxed:full(all-token)": policy_metrics(
                rows, "_combo_full", ratio, absolute=0.0, require_single=False),
        }

    # 单字 token 子集上的增量（低熵同音错检测）
    singles = [r for r in rows if r.get("single_token")]
    if singles:
        se = np.array([r["err"] for r in singles], dtype=float)
        metrics["single_token_subset"] = {
            "n": len(singles), "errors": int(se.sum()),
            **{name: summarize(np.array([float(r.get(key) or 0.0) for r in singles]), se)
               for name, key in (("entropy", "entropy"), ("beam_rank", "beam_rank_s"),
                                 ("neg_sample_agree", "neg_sample_agree"),
                                 ("align_dur_absz", "align_dur_absz"))},
        }
        metrics["single_token_subset"]["combo_beam+sample"] = summarize(
            np.mean([_rank01(np.array([float(r["entropy"] or 0.0) for r in singles])),
                     _rank01(np.array([float(r["beam_rank"] or 1.0) for r in singles])),
                     _rank01(np.array([r["neg_sample_agree"] for r in singles]))], axis=0), se)

    # 段级排序：按 max_single_entropy + 尾部位加成 排序
    for s in seg_rows:
        seg = next(x for x in segments if x["seq"] == s["seq"])
        tail = [c["entropy"] for c in seg["chars"][-max(1, len(seg["chars"]) // 10):]
                if c.get("single_token") and c.get("entropy") is not None]
        s["tail_max_entropy"] = max(tail or [0.0])
        s["risk_score"] = (s["max_single_entropy"]
                           + 0.3 * s["tail_max_entropy"]
                           + 0.1 * s["mean_entropy"]
                           + 0.2 * s["text_anomaly"])
    seg_sorted = sorted(seg_rows, key=lambda x: x["risk_score"], reverse=True)
    metrics["segment"] = {
        "rows": seg_sorted,
        "top3_error_segments": sum(1 for s in seg_sorted[:3] if s["errors"] > 0),
        "top5_error_segments": sum(1 for s in seg_sorted[:5] if s["errors"] > 0),
        "total_error_segments": sum(1 for s in seg_rows if s["errors"] > 0),
    }

    OUT.joinpath("metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(metrics, seg_rows)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("policy", "segment")}, ensure_ascii=False, indent=2))


def write_report(m: dict, seg_rows: list[dict]) -> None:
    lines = ["# P1/P2 离线实验报告（bench_10min）", "",
             f"- 字符 {m['n']}，错误 {m['errors']}（{m['prevalence']*100:.2f}%）",
             "", "## 单特征排序质量", "", "| 特征 | AUROC | AP |", "|---|---|---|"]
    for k, v in m["single"].items():
        lines.append(f"| {k} | {v['AUROC']:.4f} | {v['AP']:.4f} |")
    lines += ["", "## 融合（全片 rank 平均）", "", "| 组合 | AUROC | AP |", "|---|---|---|"]
    for k, v in m.get("combo", {}).items():
        lines.append(f"| {k} | {v['AUROC']:.4f} | {v['AP']:.4f} |")
    lines += ["", "## 生产策略口径：覆盖率 → 精确率/召回率", "",
              "| 覆盖率 | 策略 | 标亮 | TP | P | R |", "|---|---|---|---|---|---|"]
    for ratio, d in m["policy"].items():
        for name, v in d.items():
            lines.append(f"| {ratio} | {name} | {v['marked']} | {v['tp']} | "
                         f"{v['precision']*100:.1f}% | {v['recall']*100:.1f}% |")
    st = m.get("single_token_subset")
    if st:
        lines += ["", "## 单字 token 子集（同音错主战场）", "",
                  f"- n={st['n']}，错误 {st['errors']}", "",
                  "| 特征 | AUROC | AP |", "|---|---|---|"]
        for k, v in st.items():
            if isinstance(v, dict):
                lines.append(f"| {k} | {v['AUROC']:.4f} | {v['AP']:.4f} |")
    lines += ["", "## 段级风险排序（前 5）", "",
              "| seq | 字数 | 错误 | max单字熵 | 尾熵 | 风险分 |", "|---|---|---|---|---|---|"]
    for s in m["segment"]["rows"][:5]:
        lines.append(f"| {s['seq']} | {s['n']} | {s['errors']} | "
                     f"{s['max_single_entropy']:.3f} | {s['tail_max_entropy']:.3f} | "
                     f"{s['risk_score']:.3f} |")
    lines += ["", f"- 前 3 段覆盖含错段 {m['segment']['top3_error_segments']}/"
              f"{m['segment']['total_error_segments']}；"
              f"前 5 段 {m['segment']['top5_error_segments']}/"
              f"{m['segment']['total_error_segments']}"]
    OUT.joinpath("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["cache", "eval", "all"])
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 段（调试）")
    args = ap.parse_args()
    if args.stage in ("cache", "all"):
        collect(args.limit)
    if args.stage in ("eval", "all"):
        evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())