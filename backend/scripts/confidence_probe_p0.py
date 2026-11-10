#!/usr/bin/env python3
"""P0 置信度分数提取可行性 Probe（实验计划 V1.0 §9 Phase 0）。

同一模型、同一条短音频，依次验证：
  R1 基线 Greedy（当前生产路径的调用方式）
  R2 Greedy + return_dict_in_generate + output_scores
  R3 Greedy + output_logits（raw logits）
  R4 num_beams=2 + num_return_sequences=2
  R5 num_beams=4 + num_return_sequences=4

每组记录：字段是否可用、文本是否与 R1 一致、耗时、峰值 RSS。
R3 额外导出逐 Token trace（logprob/rank/entropy/margin/前缀解码文本）。

交付物（output/confidence/p0/）：
  env_report.json / generation_probe.json / token_trace_sample.jsonl

用法::

    .venv-funasr/bin/python scripts/confidence_probe_p0.py \
        --model models/Qwen3-ASR-1.7B-hf \
        --audio output/confidence/p0/probe_20s.wav
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "output" / "confidence" / "p0"

import torch  # noqa: E402
import transformers  # noqa: E402


def peak_rss_mb() -> float:
    # darwin 的 ru_maxrss 单位是字节，linux 是 KB
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1 << 20) if sys.platform == "darwin" else raw / 1024.0


def sha256(path: Path, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if limit_bytes:
            h.update(f.read(limit_bytes))
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def load_model(model_path: str, device: str):
    """与 scripts/qwen3_asr_transcribe.py::load_model 完全一致的加载方式。"""
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path, dtype=dtype, local_files_only=True, trust_remote_code=False)
    model.to(device)
    model.eval()
    return processor, model


def make_inputs(processor, model, audio_path: str, language: str, device: str):
    inputs = processor.apply_transcription_request(audio=audio_path, language=language)
    return inputs.to(device, model.dtype)


def decode_text(processor, ids) -> str:
    """与生产路径一致的解码口径（transcription_only）。"""
    return processor.decode(ids, return_format="transcription_only")[0]


def run_generation(model, inputs, label: str, **gen_kwargs) -> dict:
    """跑一次 generate，返回 {ok, sequences/scores/logits/beam_indices, wall_ms, error}。"""
    torch.mps.synchronize() if torch.backends.mps.is_available() else None
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, **gen_kwargs)
        torch.mps.synchronize() if torch.backends.mps.is_available() else None
        rec = {"ok": True, "out": out, "wall_ms": round((time.perf_counter() - t0) * 1000, 1)}
    except Exception as e:  # noqa: BLE001 - Probe 需要记录所有失败形态
        rec = {"ok": False, "error": f"{type(e).__name__}: {e}",
               "wall_ms": round((time.perf_counter() - t0) * 1000, 1)}
    rec["label"] = label
    return rec


def token_level_stats(logits_row: torch.Tensor, token_id: int) -> dict:
    """从单步 raw logits（float32 化后）计算 logprob/rank/margin/entropy。

    在设备上计算低维结果，只把标量转回 CPU。
    """
    z = logits_row.to(torch.float32)
    logp = torch.log_softmax(z, dim=-1)
    sel_lp = logp[token_id].item()
    top2_lp, top2_ids = torch.topk(logp, k=2)
    rank = int((logp > logp[token_id]).sum().item()) + 1
    p = logp.exp()
    entropy = float(-(p * logp).sum().item())  # Gibbs，nats
    return {
        "token_logprob": sel_lp,
        "token_rank": rank,
        "top1_logprob": top2_lp[0].item(),
        "top2_logprob": top2_lp[1].item(),
        "top2_margin_logprob": (top2_lp[0] - top2_lp[1]).item(),
        "max_prob": float(p.max().item()),
        "entropy_gibbs_nats": entropy,
        "top1_token_id": int(top2_ids[0].item()),
        "top2_token_id": int(top2_ids[1].item()),
    }


def build_token_trace(processor, sequences, logits, input_len: int) -> list[dict]:
    """R3 路径：raw logits -> 逐 Token 统计 + 前缀解码文本映射。

    sequences: (1, input_len + gen_len)；logits: tuple(gen_len) of (1, vocab)。
    只处理第一行（batch=1）。
    """
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    ids = sequences[0, input_len:].tolist()
    trace = []
    prev_text = ""
    for step, (tid, logit_row) in enumerate(zip(ids, logits)):
        stats = token_level_stats(logit_row[0], tid)
        # 前缀解码：观察文本是追加还是改写（Token 边界不稳定检测）
        prefix = tok.decode(ids[: step + 1], skip_special_tokens=True)
        appended = prefix.startswith(prev_text)
        trace.append({
            "step": step,
            "token_id": tid,
            "token_text": tok.decode([tid], skip_special_tokens=False),
            "prefix_text": prefix,
            "prefix_is_append": appended,
            **stats,
        })
        prev_text = prefix
    return trace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-ASR-1.7B-hf")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = (PROJECT_ROOT / args.model).resolve()
    audio_path = (PROJECT_ROOT / args.audio).resolve()

    # ---- 环境记录 -------------------------------------------------------
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "dtype": "float16" if args.device in ("mps", "cuda") else "float32",
        "model_path": str(model_path),
        "model_config_sha256": sha256(model_path / "config.json"),
        "model_weights_sha256_prefix": sha256(model_path / "model.safetensors", limit_bytes=1 << 20),
        "model_weights_bytes": (model_path / "model.safetensors").stat().st_size,
        "audio": str(audio_path),
        "max_new_tokens": 256,
    }
    (OUT_DIR / "env_report.json").write_text(json.dumps(env, ensure_ascii=False, indent=2))

    processor, model = load_model(str(model_path), args.device)
    inputs = make_inputs(processor, model, str(audio_path), args.language, args.device)
    input_len = inputs["input_ids"].shape[1]

    probe: dict = {"env": env, "runs": {}}
    print(f"[probe] input_ids 长度={input_len}，device={args.device}", flush=True)

    # ---- R1 基线 --------------------------------------------------------
    r1 = run_generation(model, inputs, "R1_baseline")
    probe["runs"]["R1"] = {k: v for k, v in r1.items() if k != "out"}
    if not r1["ok"]:
        print(f"[probe] R1 失败: {r1['error']}")
        (OUT_DIR / "generation_probe.json").write_text(json.dumps(probe, ensure_ascii=False, indent=2))
        return 1
    out1 = r1["out"]
    base_ids = out1[0, input_len:].tolist()
    base_text = decode_text(processor, out1[:, input_len:])
    probe["runs"]["R1"].update({"gen_tokens": len(base_ids), "text": base_text})
    print(f"[probe] R1 基线 {len(base_ids)} tokens: {base_text[:60]}...", flush=True)

    # ---- R2 Greedy + output_scores --------------------------------------
    r2 = run_generation(model, inputs, "R2_scores",
                        do_sample=False, return_dict_in_generate=True, output_scores=True)
    entry = {k: v for k, v in r2.items() if k != "out"}
    if r2["ok"]:
        out2 = r2["out"]
        text2 = decode_text(processor, out2.sequences[:, input_len:])
        entry.update({
            "has_sequences": hasattr(out2, "sequences"),
            "has_scores": bool(out2.scores),
            "n_score_steps": len(out2.scores) if out2.scores else 0,
            "score_step_shape": list(out2.scores[0].shape) if out2.scores else None,
            "has_logits": hasattr(out2, "logits") and out2.logits is not None,
            "text": text2,
            "text_matches_baseline": text2 == base_text,
            "gen_tokens": int(out2.sequences.shape[1] - input_len),
        })
    probe["runs"]["R2"] = entry
    print(f"[probe] R2 ok={r2['ok']} text_match={entry.get('text_matches_baseline')}", flush=True)

    # ---- R3 Greedy + output_logits --------------------------------------
    r3 = run_generation(model, inputs, "R3_logits",
                        do_sample=False, return_dict_in_generate=True,
                        output_scores=True, output_logits=True)
    entry = {k: v for k, v in r3.items() if k != "out"}
    trace = None
    if r3["ok"]:
        out3 = r3["out"]
        text3 = decode_text(processor, out3.sequences[:, input_len:])
        entry.update({
            "has_logits": hasattr(out3, "logits") and out3.logits is not None,
            "n_logit_steps": len(out3.logits) if getattr(out3, "logits", None) else 0,
            "logit_step_shape": list(out3.logits[0].shape) if getattr(out3, "logits", None) else None,
            "text": text3,
            "text_matches_baseline": text3 == base_text,
            "gen_tokens": int(out3.sequences.shape[1] - input_len),
        })
        if getattr(out3, "logits", None):
            trace = build_token_trace(processor, out3.sequences, out3.logits, input_len)
            with open(OUT_DIR / "token_trace_sample.jsonl", "w") as f:
                for row in trace:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            # scores 与 logits 的语义对照（前 5 步）：log_softmax 后是否一致
            if out3.scores:
                cmp = []
                gen_ids = out3.sequences[0, input_len:].tolist()
                for i in range(min(5, len(gen_ids))):
                    s = torch.log_softmax(out3.scores[i][0].to(torch.float32), dim=-1)[gen_ids[i]].item()
                    l = torch.log_softmax(out3.logits[i][0].to(torch.float32), dim=-1)[gen_ids[i]].item()
                    cmp.append({"step": i, "score_logprob": s, "logit_logprob": l,
                                "abs_diff": abs(s - l)})
                entry["score_vs_logit_logprob_first5"] = cmp
    probe["runs"]["R3"] = entry
    print(f"[probe] R3 ok={r3['ok']} logits={entry.get('has_logits')} "
          f"text_match={entry.get('text_matches_baseline')}", flush=True)

    # ---- R4 / R5 Beam ----------------------------------------------------
    for label, nb, nr in (("R4_beam2", 2, 2), ("R5_beam4", 4, 4)):
        r = run_generation(model, inputs, label,
                           num_beams=nb, num_return_sequences=nr,
                           do_sample=False, return_dict_in_generate=True, output_scores=True)
        entry = {k: v for k, v in r.items() if k != "out"}
        if r["ok"]:
            out = r["out"]
            seqs = out.sequences[:, input_len:]
            texts = processor.decode(seqs, return_format="transcription_only")
            entry.update({
                "sequences_shape": list(out.sequences.shape),
                "n_returned": int(out.sequences.shape[0]),
                "has_beam_indices": hasattr(out, "beam_indices") and out.beam_indices is not None,
                "beam_indices_shape": (list(out.beam_indices.shape)
                                       if getattr(out, "beam_indices", None) is not None else None),
                "texts": texts,
                "beam0_text_matches_baseline": (texts[0] == base_text) if texts else None,
            })
        probe["runs"][label] = entry
        print(f"[probe] {label} ok={r['ok']} "
              f"n_returned={entry.get('n_returned')} beam_indices={entry.get('has_beam_indices')}",
              flush=True)

    # ---- 汇总：beam 路径 transition scores 单独重跑一次（生命周期简单）----
    r5b = run_generation(model, inputs, "R5b_transition_scores",
                         num_beams=4, num_return_sequences=4, do_sample=False,
                         return_dict_in_generate=True, output_scores=True)
    if r5b["ok"]:
        out = r5b["out"]
        try:
            # transformers 5.x：compute_transition_scores 是 GenerationMixin 方法
            ts = model.compute_transition_scores(
                out.sequences, out.scores, out.beam_indices, normalize_logits=False)
            probe["runs"]["R5_transition_scores"] = {
                "ok": True,
                "shape": list(ts.shape),
                "first_row_first5": [round(float(v), 4) for v in ts[0][:5].tolist()],
                "n_nan_or_inf": int(torch.isnan(ts).sum() + torch.isinf(ts).sum()),
            }
        except Exception as e:  # noqa: BLE001
            probe["runs"]["R5_transition_scores"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    else:
        probe["runs"]["R5_transition_scores"] = {"ok": False, "error": r5b.get("error")}

    probe["peak_rss_mb"] = round(peak_rss_mb(), 1)
    (OUT_DIR / "generation_probe.json").write_text(json.dumps(probe, ensure_ascii=False, indent=2))
    print(f"[probe] 完成，peak RSS {probe['peak_rss_mb']} MB，结果见 {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
