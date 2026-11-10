#!/usr/bin/env python3
"""使用 Qwen3-ASR（transformers 原生接口）转录音频，只产出原始文本。

用法::

    python scripts/qwen3_asr_transcribe.py \
        --model models/Qwen3-ASR-1.7B-hf \
        --audio tests/测试音频.mp3 \
        --output output/测试音频_qwen3-asr-1.7b.txt \
        --language Chinese

职责边界：
- 本脚本**只做识别**，输出就是事实底稿（raw）。
- 数字规范化与语义纠错统一由 ``scripts/qwen3_asr_postprocess.py`` 完成，
  以保证同一份 raw 走同一套 wetext 规则与同一份规则指纹。
- 跨环境调用请使用 ``scripts/local_asr.py transcribe``（编排层）。

结构化事件：关键节点向 stdout 输出 ``@@EVENT@@ {json}`` 行，
供编排层解析进度；不影响人类阅读其余日志。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from local_asr.asr.batching import (  # noqa: E402
    BatchConfig, group_budget, plan_buckets, select_batch_size, split_groups,
)
from local_asr.asr.generation import (  # noqa: E402
    GenerationResult, aggregate_status, classify_generation, eos_ids_from,
    flag_empty_transcript, segment_payload,
)
from local_asr.asr.segmentation import (  # noqa: E402
    SegmentationConfig, VadSegment, build_chunks, decide_mode, effective_len,
    estimate_max_new_tokens, merge_short_gaps, merge_texts,
    plan_balanced_cuts, plan_cuts, plan_dp_cuts, keep_char,
    apply_dedup_to_texts,
)
from local_asr.asr.confidence import (  # noqa: E402
    DEFAULT_MAX_RATIO, LOW_CONF_MIN_ENTROPY, LOW_CONF_SCORE_FLOOR,
    LOW_CONF_SEGMENT_MAX_ENTROPY, SegmentSignals, align_char_signals,
    char_feature_series, collect_segment_signals, finalize_segment_risk,
    merge_positions, sample_agreement, select_low_conf, token_char_spans,
    cross_model_flags,
)
from local_asr.asr.features import extract_vad_profile  # noqa: E402
from local_asr.asr.quality import profile_audio  # noqa: E402
from local_asr.asr.strategy import (  # noqa: E402
    StrategyDecision, apply_decision_to_config, decide_strategy,
)
from local_asr.audio import (  # noqa: E402
    audio_duration, can_convert_audio, is_pcm16k_mono_wav, prepare_pcm16k,
    slice_wav,
)
from local_asr.postprocess.artifacts import (  # noqa: E402
    json_hash, model_manifest, peak_rss_mb, preflight_paths, publish_text,
    same_path, text_hash,
)
from local_asr.postprocess.corrector import require_module, validate_local_model  # noqa: E402
from local_asr.protocol import emit_event as _emit_event  # noqa: E402


# ── 常驻模型缓存 ─────────────────────────────────────────────────────
# 单独跑脚本时每个进程只加载一次，不改变原有行为；由
# ``qwen3_asr_worker.py`` 复用同一进程连续处理多个任务时，第二次起
# 直接命中缓存（实测省 5.5–10.5s 模型加载 + 2–9s 对齐模型加载）。
# 只保留一组 ASR 权重：0.6B 与 1.7B 同时驻留会在 8GB 卡上溢出到 WDDM
# 共享显存，推理反而慢约 20 倍。
_MODEL_CACHE: dict[str, tuple[object, object, str]] = {}
_ALIGNER_CACHE: dict[str, object] = {}
_VAD_CACHE: dict[str, object] = {}


def release_cached_models() -> None:
    """清空常驻权重并归还显存（切换模型 / worker 退出前调用）。"""
    import gc

    _MODEL_CACHE.clear()
    _ALIGNER_CACHE.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - 释放失败不影响主流程
        pass


def load_model(model_path: str, device: str):
    """从本地目录加载 processor 与模型（绝不联网）。

    - ``local_files_only=True``：preflight 已设 HF_HUB_OFFLINE，双保险；
    - ``trust_remote_code=False``：只用 transformers 原生多模态接口，
      不执行模型仓库自带的自定义代码；
    - dtype：mps/cuda 用 float16（省一半显存、推理更快），cpu 用
      float32（CPU 上 fp16 反而慢且部分算子不支持）。

    返回 ``(processor, model)``，均已放到目标 device 并切到 eval 模式。
    同一进程重复调用同一路径时命中模块级缓存（常驻 worker 场景）。
    """
    import torch
    from transformers import AutoModel, AutoProcessor

    key = str(Path(model_path).resolve())
    cached = _MODEL_CACHE.get(key)
    if cached is not None and cached[2] == device:
        print(f"[qwen3-asr] 复用常驻模型 {key}（device={device}）", flush=True)
        return cached[0], cached[1]
    if _MODEL_CACHE:
        print("[qwen3-asr] 释放已加载权重，切换模型...", flush=True)
        release_cached_models()

    # transformers 4.57.6 无 AutoModelForMultimodalLM（旧 API 已移除）；
    # qwen-asr 包自带 Qwen3ASR 实现并在 import 时注册到 AutoModel/AutoProcessor
    import qwen_asr  # noqa: F401  触发 Qwen3ASRConfig 注册

    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False)
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model = AutoModel.from_pretrained(
        model_path, dtype=dtype, local_files_only=True, trust_remote_code=False)
    model.to(device)
    model.eval()
    _MODEL_CACHE[key] = (processor, model, device)
    return processor, model


def _parse_raw_output(raw: str, language: str | None) -> str:
    """剥离 qwen-asr 元数据（``language X<asr_text>`` 头），保留原文空白。

    与 qwen_asr.parse_asr_output 的差异：不 strip、不去重——原始文本的
    空白/换行必须原样传给后处理（后处理管道负责清洗与数值保真校验）。
    """
    s = str(raw)
    if "<asr_text>" in s:
        head, _, s = s.partition("<asr_text>")
        if language is None and "language none" in head.lower():
            return ""  # 空音频：模型回显 language None
    return s


def _prepare_asr_inputs(processor, audio_paths: list, prompt: str | None,
                        language: str | None) -> tuple:
    """构建 qwen-asr 0.0.6 批量输入：文本 prompt + 16kHz 波形列表。

    - prompt 作为 system 上下文进 chat template；
    - language 强制纯文本输出（prompt 末尾追加 ``language X<asr_text>``）；
    - 文件路径经 normalize_audios 加载为 float32 ndarray（processor 不收路径）。
    """
    from qwen_asr.inference.utils import normalize_audios

    msgs = [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    text_prompt = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False)
    if language:
        text_prompt = text_prompt + f"language {language}<asr_text>"
    texts = [text_prompt] * len(audio_paths)
    wavs = normalize_audios(list(audio_paths))
    return texts, wavs


def _content_token_ids(generated_row, eos_ids: set) -> list[int]:
    """从生成序列中提取内容 token（去除 EOS / padding）。"""
    tids = []
    for t in range(len(generated_row)):
        tid = int(generated_row[t].item())
        if tid in eos_ids:
            break
        tids.append(tid)
    return tids


def _token_features_from_scores(scores, generated: "torch.Tensor", eos_ids: set,
                                batch_size: int, *, beam_size: int = 1,
                                beam_indices=None) -> list[tuple | None]:
    """从逐步 logits 提取每个样本（候选 0）的逐 token 熵/对数概率。

    scores: generate 返回的 scores tuple；greedy 每步 (batch, vocab)，
        beam 每步 (batch*num_beams, vocab)
    generated: 生成 token 矩阵；beam 时行数为 batch*num_beams
    beam_indices: beam 时每步的父 beam 行号（不含 prompt），形状
        (batch*num_beams, steps)
    返回 list[tuple[dict, ...] | None]，每个样本可能为 None（无有效 token）。

    beam 特征只在候选 0 上取；额外保留 ``beam_rank`` / ``beam_top1``，
    但 bench 实测 rank 对 Qwen3-ASR 无增量信息，默认 beam_size=1 不启用。
    """
    import torch

    if scores is None:
        return [None] * batch_size
    out: list[tuple | None] = []
    for i in range(batch_size):
        row = i * beam_size
        if row >= generated.shape[0]:
            out.append(None)
            continue
        tids = _content_token_ids(generated[row], eos_ids)
        if not tids:
            out.append(None)
            continue
        feats: list[dict] = []
        for t in range(len(tids)):
            parent = row
            if beam_size > 1 and beam_indices is not None and t < beam_indices.shape[1]:
                rel = int(beam_indices[row, t])
                if 0 <= rel < beam_size:
                    parent = i * beam_size + rel
            if parent >= scores[t].shape[0]:
                parent = row
            logits = scores[t][parent].to(torch.float32)
            logp = torch.log_softmax(logits, dim=-1)
            tid = tids[t]
            sel_lp = float(logp[tid].item())
            p = logp.exp()
            entropy = float(-(p * logp).sum().item())
            top2 = torch.topk(logp, k=2).values
            neg_margin = -float((top2[0] - top2[1]).item())
            feats.append({
                "entropy": entropy,
                "neg_logprob": -sel_lp,
                "neg_margin": neg_margin,
                "beam_rank": float(int((logp > logp[tid]).sum().item()) + 1),
                "beam_top1": float(top2[0]),
            })
        out.append(tuple(feats))
    return out


# 低置信策略常量统一由 local_asr.asr.confidence 提供（P0/P1/P2 实测见
# docs/asr-error-detection-research-2026-11.md）。这里只保留离线对比用的
# 单段旧实现（无全局预算），生产路径走 _build_segment_payloads。
def _compute_low_conf_spans(
    token_features: tuple | None,
    text: str,
    tokenizer,
) -> list[list[int]]:
    """标出低置信字符区间。

    仅保留「单字 token 且 entropy >= LOW_CONF_MIN_ENTROPY」的信息字符；
    段级门控：单字 token 最高熵低于 LOW_CONF_SEGMENT_MAX_ENTROPY 时整段不标。
    返回的是 text 的码点索引区间。
    """
    spans = _token_char_spans(token_features, text, tokenizer)
    if not spans:
        return []
    candidates: list[tuple[int, float]] = []  # (字符位置, 熵)
    for start, end, feats in spans:
        chars = [i for i in range(start, min(end, len(text))) if keep_char(text[i])]
        if len(chars) != 1:
            continue  # 多字 token：整词共享熵，标亮会大面积误报
        candidates.append((chars[0], float(feats.get("entropy", 0.0))))
    if not candidates or max(e for _, e in candidates) < LOW_CONF_SEGMENT_MAX_ENTROPY:
        return []
    positions = {pos for pos, ent in candidates if ent >= LOW_CONF_MIN_ENTROPY}
    return _merge_positions(sorted(positions))


def _compute_low_conf_spans_with_ratio(
    token_features: tuple | None,
    text: str,
    tokenizer,
    ratio: float = 0.1,
) -> list[list[int]]:
    """按指定比例取高熵字符并合并为连续区间（ratio=0 表示不标亮）。

    保留给离线实验/对比使用；生产默认策略见 _compute_low_conf_spans。
    """
    if ratio <= 0:
        return []
    indexed = _char_entropy_series(token_features, text, tokenizer)
    if not indexed:
        return []
    indexed = sorted(indexed, key=lambda x: x[1], reverse=True)
    top_n = max(1, int(len(indexed) * ratio))
    high_positions = {pos for pos, _ in indexed[:top_n]}
    return _merge_positions(sorted(high_positions))


def _merge_positions(sorted_pos: list[int]) -> list[list[int]]:
    """兼容包装：区间合并实现在 local_asr.asr.confidence。"""
    return merge_positions(sorted_pos)


def _char_entropy_series(
    token_features: tuple | None,
    text: str,
    tokenizer,
) -> list[tuple[int, float]]:
    """返回每个信息字符的 (text 字符索引, entropy)。"""
    return char_feature_series(token_features, text, tokenizer, "entropy")


def _token_char_spans(
    token_features: tuple | None,
    text: str,
    tokenizer,
) -> list[tuple[int, int, dict]]:
    """兼容包装：token → 字符跨度实现在 local_asr.asr.confidence。"""
    return token_char_spans(token_features, text, tokenizer)


def _char_feature_series(
    token_features: tuple | None,
    text: str,
    tokenizer,
    key: str,
) -> list[tuple[int, float]]:
    """兼容包装：逐字符特征展开实现在 local_asr.asr.confidence。"""
    return char_feature_series(token_features, text, tokenizer, key)


def start_cross_model(args, audio: Path, work_dir: Path):
    """主 ASR 推理前在后台启动第二模型（FunASR Paraformer，CPU）。

    交叉验证的固定开销约 30s（加载 13.5s + 推理 16.4s / 10min 音频），
    原先在推理结束后串行执行，长音频 wall 时间额外增加 40–50s。
    改为与主推理并行：CPU 与 GPU 输出逐字符一致（实测），且不占显存
    （主 ASR batch=4 已接近 8GB 上限，GPU 并行会触发 WDDM 共享显存）。

    返回 ``(process, out_path, log_path, started_at)``；未启用 auto 或
    启动失败返回 None（调用方退回同步路径）。
    """
    if getattr(args, "cross_check", None) != "auto":
        return None
    script = Path(__file__).with_name("funasr_transcribe.py")
    if not script.is_file():
        print(f"[qwen3-asr] 交叉验证跳过: 找不到 {script}", file=sys.stderr, flush=True)
        return None
    out = work_dir / "cross_model.txt"
    log = work_dir / "cross_model.log"
    cmd = [sys.executable, str(script), "--model", "paraformer-zh",
           "--audio", str(audio), "--output", str(out), "--device", "cpu"]
    print("[qwen3-asr] 交叉验证: 后台启动 FunASR（CPU，与主推理并行）...", flush=True)
    try:
        with log.open("wb") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    except OSError as exc:
        print(f"[qwen3-asr] 交叉验证启动失败，忽略: {exc}",
              file=sys.stderr, flush=True)
        return None
    return proc, out, log, time.perf_counter()


def finish_cross_model(job, timeout: float = 1800.0) -> tuple[str | None, float]:
    """等待后台交叉验证结束，返回（文本, 从启动到可用的墙钟秒数）。"""
    proc, out, log, started = job
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        print("[qwen3-asr] 交叉验证超时，忽略并继续", file=sys.stderr, flush=True)
        return None, time.perf_counter() - started
    if code != 0 or not out.is_file():
        try:
            tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
        except OSError:
            tail = []
        print("[qwen3-asr] 交叉验证失败，忽略并继续: " + " | ".join(tail),
              file=sys.stderr, flush=True)
        return None, time.perf_counter() - started
    return out.read_text(encoding="utf-8"), time.perf_counter() - started


def resolve_cross_model_text(args, audio: Path, work_dir: Path) -> str | None:
    """获取第二模型文本（同步兜底：传路径直接读取；auto 现场 CPU 推理）。

    正常路径由 :func:`start_cross_model` 预启动并在推理后
    :func:`finish_cross_model` 收尾，这里只处理未预启动的情况。
    """
    value = getattr(args, "cross_check", None)
    if not value:
        return None
    if value != "auto":
        path = Path(value)
        if not path.is_file():
            print(f"[qwen3-asr] 交叉验证文本不存在，忽略: {path}",
                  file=sys.stderr, flush=True)
            return None
        return path.read_text(encoding="utf-8")

    script = Path(__file__).with_name("funasr_transcribe.py")
    if not script.is_file():
        print(f"[qwen3-asr] 交叉验证跳过: 找不到 {script}", file=sys.stderr, flush=True)
        return None
    out = work_dir / "cross_model.txt"
    cmd = [sys.executable, str(script), "--model", "paraformer-zh",
           "--audio", str(audio), "--output", str(out), "--device", "cpu"]
    print("[qwen3-asr] 交叉验证: 调用 FunASR Paraformer（CPU）...", flush=True)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[qwen3-asr] 交叉验证失败，忽略并继续: {exc}",
              file=sys.stderr, flush=True)
        return None
    if proc.returncode != 0 or not out.is_file():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        print("[qwen3-asr] 交叉验证失败，忽略并继续: " + " | ".join(tail),
              file=sys.stderr, flush=True)
        return None
    return out.read_text(encoding="utf-8")


def build_cross_flags(
    merged_text: str, segments: list["Segment"], dedup: list[dict], other_text: str,
) -> dict[int, list[bool]]:
    """把「与第二模型不一致」映射到每个 chunk 原文的字符坐标。

    ``merged_text`` 是去重后的合并全文；各段最终文本（去重裁剪后）是它的
    连续子串，因此用顺序 ``find`` 定位偏移。返回 ``{段序号: 原文坐标 flags}``，
    与 ``seg.chars[i].index``（chunk 原文下标）对齐，供打分使用。
    """
    flags = cross_model_flags(merged_text, other_text)
    orig_texts = [s.result.text for s in segments]
    final_texts = (apply_dedup_to_texts(orig_texts, dedup)
                   if dedup else orig_texts)
    out: dict[int, list[bool]] = {}
    cursor = 0
    for i, (orig, final) in enumerate(zip(orig_texts, final_texts)):
        if not final:
            continue
        pos = merged_text.find(final, cursor)
        if pos < 0:  # 顺序查找失败时兜底从头找（重叠裁剪可能改变顺序假设）
            pos = merged_text.find(final)
        if pos < 0:
            continue
        seg_flags = [False] * len(orig)
        lo = orig.find(final)
        lo = 0 if lo < 0 else lo
        for k, flag in enumerate(flags[pos:pos + len(final)]):
            if flag and lo + k < len(seg_flags):
                seg_flags[lo + k] = True
        out[i] = seg_flags
        cursor = pos + len(final)
    return out


def _build_segment_payloads(
    segments: list["Segment"], processor, args,
    sample_texts: dict[int, list[str]] | None = None,
    align_feats: dict[int, list[dict | None]] | None = None,
    char_times: dict[int, list[tuple[float, float] | None]] | None = None,
    audio_offsets: dict[int, int] | None = None,
    cross_flags: dict[int, list[bool]] | None = None,
) -> tuple[list[dict], list[SegmentSignals]]:
    """构建 stats.json 的 segments 数组，并返回段级信号（供 review_order）。

    P1：全局预算截断（默认 4%）+ 段级门控 + 文本/对齐风险。
    P2：可选采样一致性（quality 档）与强制对齐（recall 档）。
    ``char_times``：字符级对齐时间（相对块 WAV 的秒）；``audio_offsets``
    为各块在全局音频中的起点毫秒，用于换算成全局时间写入 stats。
    """
    tokenizer = getattr(processor, "tokenizer", None)
    dump = bool(getattr(args, "dump_token_features", False)) and tokenizer is not None
    seg_signals: list[SegmentSignals] = []
    for i, s in enumerate(segments):
        sig = collect_segment_signals(
            s.result.token_features, s.result.text, tokenizer, seq=i,
            sample_agreement=(sample_agreement(sample_texts[i], s.result.text)
                              if sample_texts and i in sample_texts else None),
            align_features=(align_feats.get(i) if align_feats else None),
        )
        # 第二模型交叉验证：本段成功映射才标记 cross_checked（映射失败
        # 不能当作「两模型一致」，否则会误取消该段全部低置信词）。
        # 不一致字符只影响后续「低置信词核验」，不参与选词打分。
        seg_cross = (cross_flags or {}).get(i)
        if seg_cross is not None:
            sig.cross_checked = True
            for ch_sig in sig.chars:
                if ch_sig is not None and ch_sig.index < len(seg_cross) \
                        and seg_cross[ch_sig.index]:
                    ch_sig.cross_mismatch = True
        seg_signals.append(sig)
    marked = select_low_conf(
        seg_signals,
        mode=getattr(args, "low_conf_mode", "balanced"),
        max_ratio=getattr(args, "low_conf_ratio", DEFAULT_MAX_RATIO),
        score_floor=getattr(args, "low_conf_floor", LOW_CONF_SCORE_FLOOR),
    )
    for sig in seg_signals:
        finalize_segment_risk(sig)

    payloads: list[dict] = []
    for i, (s, sig) in enumerate(zip(segments, seg_signals)):
        payload = segment_payload(
            s.result, index=i, start_s=s.start_s, end_s=s.end_s,
            cut_silence_ms=s.cut_silence_ms, include_text=True,
            low_conf_spans=sig.spans or None)
        payload["risk_flags"] = sig.risk_flags
        payload["review_priority"] = round(sig.risk_score, 4)
        if char_times and i in char_times:
            # 字符级时间戳：[[start_ms, end_ms], ...] 与 text 等长，
            # 未对齐字符（标点/未覆盖）为 None；已换算为全局音频毫秒。
            offset_ms = int((audio_offsets or {}).get(i, 0))
            payload["char_times"] = [
                ([int(round(t[0] * 1000)) + offset_ms,
                  int(round(t[1] * 1000)) + offset_ms] if t is not None else None)
                for t in char_times[i]
            ]
        if dump:
            # 三种候选信号都展开到字符，便于离线比较/调参
            ent = dict(_char_entropy_series(s.result.token_features, s.result.text, tokenizer))
            nlp = dict(_char_feature_series(
                s.result.token_features, s.result.text, tokenizer, "neg_logprob"))
            nmg = dict(_char_feature_series(
                s.result.token_features, s.result.text, tokenizer, "neg_margin"))
            payload["char_entropies"] = [[idx, round(e, 6)] for idx, e in ent.items()]
            payload["char_features"] = [
                [idx, round(e, 6), round(nlp.get(idx, 0.0), 6), round(nmg.get(idx, 0.0), 6)]
                for idx, e in ent.items()]
            # 完整信号（含 token 长度 + 可选采样/对齐/交叉验证），供离线重放
            # 预算/门控策略。列：[index, entropy, neg_logprob, neg_margin,
            #   token_len, single, sample_agree, align_abs_z, align_short,
            #   align_long, cross_mismatch]
            payload["char_signals"] = [
                [c.index, round(c.entropy, 6), round(c.neg_logprob, 6),
                 round(c.neg_margin, 6), int(c.token_len), int(c.single_token),
                 (round(c.sample_agree, 4) if c.sample_agree is not None else None),
                 round(c.align_abs_z, 4),
                 int(c.align_short), int(c.align_long), int(c.cross_mismatch)]
                for c in sig.chars if c is not None]
        payloads.append(payload)
    if tokenizer is not None:
        print(f"[qwen3-asr] 低置信策略: mode={getattr(args, 'low_conf_mode', 'balanced')} "
              f"ratio={getattr(args, 'low_conf_ratio', DEFAULT_MAX_RATIO):.0%} "
              f"标亮 {marked} 字", flush=True)
    return payloads, seg_signals


def decode_window(processor, model, audio_path: str, language: str | None, device: str,
                  max_new_tokens: int, prompt: str | None = None,
                  beam_size: int = 1) -> GenerationResult:
    """识别一段音频，并判定生成是否完整（EOS / 触顶）。

    qwen-asr 0.0.6 processor 无 apply_transcription_request（旧 API 已移除）：
    文本 prompt 经 apply_chat_template 构建，解码用 batch_decode，
    再经 _parse_raw_output 剥离元数据（保留原文空白交给后处理）。
    output_scores 启用时同步收集逐 token 熵；beam_size>1 时取候选 0。
    """
    import torch

    texts, wavs = _prepare_asr_inputs(processor, [audio_path], prompt, language)
    inputs = processor(text=texts, audio=wavs, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    kwargs = {"max_new_tokens": max_new_tokens, "output_scores": True}
    if beam_size > 1:
        kwargs.update(num_beams=beam_size, num_return_sequences=beam_size,
                      do_sample=False)
    with torch.no_grad():
        output = model.generate(**inputs, **kwargs)
    output_ids = output.sequences if hasattr(output, "sequences") else output
    seqs = output_ids[:, inputs["input_ids"].shape[1]:]
    generated = seqs[:1]  # 候选 0
    raw = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    text = _parse_raw_output(raw, language)
    token_count = int(generated.shape[1])
    last_id = int(generated[0, -1]) if token_count else -1
    eos_ids = eos_ids_from(model.generation_config)
    hit_eos, hit_limit = classify_generation(
        last_token_id=last_id,
        generated_tokens=token_count,
        max_new_tokens=max_new_tokens,
        eos_ids=eos_ids,
    )
    scores = getattr(output, "scores", None)
    beam_indices = getattr(output, "beam_indices", None) if beam_size > 1 else None
    feats = (_token_features_from_scores(
        scores, seqs, eos_ids, 1, beam_size=beam_size, beam_indices=beam_indices)
        if scores else [None])
    return GenerationResult(
        text=text, generated_tokens=token_count, hit_eos=hit_eos, hit_limit=hit_limit,
        token_features=feats[0] if feats else None)


@dataclass
class Segment:
    """一段识别结果及其时间位置（秒）与所用 token 预算。"""

    result: GenerationResult
    start_s: float
    end_s: float
    cut_silence_ms: int | None
    max_new_tokens: int


def _load_vad(vad_model: Path):
    """加载并缓存 FunASR VAD；常驻 worker 下跨任务复用（每次省 1–3s）。"""
    key = str(Path(vad_model).resolve())
    model = _VAD_CACHE.get(key)
    if model is None:
        from funasr import AutoModel

        model = AutoModel(model=str(vad_model), disable_update=True, disable_pbar=True,
                          device="cpu")
        _VAD_CACHE[key] = model
    else:
        print("[qwen3-asr] 复用常驻 VAD 模型", flush=True)
    return model


def detect_speech(wav: Path, vad_model: Path) -> list[VadSegment]:
    """用 FunASR FSMN-VAD 检测语音段（毫秒区间）。"""
    model = _load_vad(vad_model)
    raw = model.generate(input=str(wav), cache={})[0]["value"]
    return [VadSegment(int(start), int(end)) for start, end in raw]


def resolve_max_new_tokens(requested: int | None, seconds: float, chunked: bool) -> int:
    """显式值优先；未指定时整段沿用 4096，分块按块长估算。"""
    if requested:
        return requested
    return estimate_max_new_tokens(seconds) if chunked else 4096


def decode_group(processor, model, audio_paths: list[str], language: str | None, device: str,
                 max_new_tokens: int, prompt: str | None = None,
                 beam_size: int = 1) -> list[GenerationResult]:
    """批量识别同组音频，返回与输入同序的结果。

    batch 内会按最长序列 padding，因此每行 token 宽度相同；
    ``generated_tokens`` 记录该宽度（上界），``hit_eos`` 仍按行末 token 判定。
    同时收集逐 token 的熵/对数概率（output_scores），供置信度标亮。
    beam_size>1 时只取每个样本的候选 0。
    """
    import torch

    texts, wavs = _prepare_asr_inputs(processor, audio_paths, prompt, language)
    inputs = processor(text=texts, audio=wavs, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    kwargs = {"max_new_tokens": max_new_tokens, "output_scores": True}
    if beam_size > 1:
        kwargs.update(num_beams=beam_size, num_return_sequences=beam_size,
                      do_sample=False)
    with torch.no_grad():
        output = model.generate(**inputs, **kwargs)
    output_ids = output.sequences if hasattr(output, "sequences") else output
    seqs = output_ids[:, inputs["input_ids"].shape[1]:]
    rows = [i * beam_size for i in range(len(audio_paths))]
    generated = seqs[rows]
    raws = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    texts = [_parse_raw_output(r, language) for r in raws]
    eos_ids = eos_ids_from(model.generation_config)
    width = int(generated.shape[1])
    # 逐 token 特征（beam 时按 beam_indices 取父 beam 分布）
    scores = getattr(output, "scores", None)
    beam_indices = getattr(output, "beam_indices", None) if beam_size > 1 else None
    batch_feats = (_token_features_from_scores(
        scores, seqs, eos_ids, len(audio_paths),
        beam_size=beam_size, beam_indices=beam_indices)
        if scores else [None] * len(audio_paths))
    results: list[GenerationResult] = []
    for i, (row, text) in enumerate(zip(generated, texts)):
        last_id = int(row[-1]) if width else -1
        hit_eos, hit_limit = classify_generation(
            last_token_id=last_id, generated_tokens=width,
            max_new_tokens=max_new_tokens, eos_ids=eos_ids)
        results.append(GenerationResult(
            text=text, generated_tokens=width, hit_eos=hit_eos, hit_limit=hit_limit,
            token_features=batch_feats[i] if i < len(batch_feats) else None))
    return results


def mps_driver_mb() -> float:
    """当前 MPS driver 已分配内存（MB）；不可用时返回 0。"""
    try:
        import torch
    except ImportError:
        return 0.0
    try:
        if torch.backends.mps.is_available():
            return torch.mps.driver_allocated_memory() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


@dataclass(frozen=True)
class PiecePlan:
    """一块待识别音频的时间边界。"""

    index: int
    core_start_s: float
    core_end_s: float
    audio_start_ms: int
    audio_end_ms: int
    cut_silence_ms: int | None = None


def plan_batch(args, seconds: dict[int, float]) -> tuple[BatchConfig, int, list[list[int]], float]:
    """按块长与内存决定批大小和分组；批大小为 1 时保持逐条顺序。"""
    config = BatchConfig(
        tolerance=args.batch_tolerance,
        max_batch=args.batch_max,
        max_driver_mb=args.batch_max_driver_mb,
    )
    driver_mb = mps_driver_mb()
    longest = max(seconds.values(), default=0.0)
    size = select_batch_size(
        requested=args.batch_size, auto=args.batch_auto,
        driver_mb=driver_mb, longest_seconds=longest, config=config)
    if size <= 1:
        groups = [[index] for index in sorted(seconds)]
    else:
        # 先按「实际可用批大小」分桶再切分：否则按 max_batch 分出的 5 块组
        # 会被 split_groups 切成 4+1，多出一批只有 1 块的组（32 分钟实测
        # 8 组变 13 组，推理 140 s → 178 s）。
        groups = split_groups(plan_buckets(seconds, replace(config, max_batch=size)), size)
    return config, size, groups, driver_mb


def decode_pieces(
    processor, model, args, language: str | None, work: Path, pcm: Path,
    pieces: list[PiecePlan], *, merge: bool,
) -> tuple[list[Segment], str, dict, list[dict], dict[int, Path]]:
    """切块 → 分桶 → 批量识别；失败时整组回退逐条重跑。

    返回额外包含 ``paths``（块序号 → 临时 WAV 路径），供可选的
    采样一致性 / 强制对齐等后处理复用，避免重复切音频。
    """
    paths: dict[int, Path] = {}
    seconds: dict[int, float] = {}
    for piece in pieces:
        path = work / f"chunk_{piece.index:03d}.wav"
        slice_wav(pcm, piece.audio_start_ms, piece.audio_end_ms, path)
        paths[piece.index] = path
        seconds[piece.index] = (piece.audio_end_ms - piece.audio_start_ms) / 1000.0

    config, batch_size, groups, driver_mb = plan_batch(args, seconds)
    if batch_size > 1:
        print(f"[qwen3-asr] 分桶: {len(groups)} 组，批大小 ≤{batch_size}，"
              f"当前 MPS driver {driver_mb:.0f}MB（上限 {config.max_driver_mb}MB）", flush=True)
    elif args.batch_auto or args.batch_size > 1:
        print(f"[qwen3-asr] 批大小降为 1：当前 MPS driver {driver_mb:.0f}MB，"
              f"上限 {config.max_driver_mb}MB", flush=True)
    _emit_event("asr_batch_planned", batch_size=batch_size, groups=groups,
                driver_mb=round(driver_mb))

    results: dict[int, GenerationResult] = {}
    budgets: dict[int, int] = {}
    fallbacks: list[dict] = []
    # 每批耗时（真实耗时 / 音频秒），用于识别异常慢批次
    group_timings: list[dict] = []
    done = 0

    def _single(index: int) -> GenerationResult:
        budget = budgets.setdefault(index, resolve_max_new_tokens(
            args.max_new_tokens, seconds[index], chunked=True))
        return decode_window(processor, model, str(paths[index]), language,
                             args.device, budget, args.prompt,
                             beam_size=getattr(args, "beam_size", 1))

    def _mark_degraded(index: int, exc: Exception) -> None:
        results[index] = GenerationResult("", 0, False, False, degraded=True)
        fallbacks.append({"group": [index], "error": f"单条重跑失败: {exc}"})
        print(f"[qwen3-asr] 警告: 第 {index} 块识别失败，标记 degraded: {exc}",
              file=sys.stderr, flush=True)

    for group in groups:
        group_started = time.perf_counter()
        if len(group) == 1:
            try:
                results[group[0]] = _single(group[0])
            except Exception as exc:
                _mark_degraded(group[0], exc)
        else:
            budget = group_budget(group, seconds)
            for index in group:
                budgets[index] = budget
            try:
                outputs = decode_group(
                    processor, model, [str(paths[i]) for i in group], language,
                    args.device, budget, args.prompt,
                    beam_size=getattr(args, "beam_size", 1))
            except Exception as exc:
                fallbacks.append({"group": list(group), "error": str(exc)})
                print(f"[qwen3-asr] 警告: 批量推理失败，回退逐条重跑 {group}: {exc}",
                      file=sys.stderr, flush=True)
                for index in group:
                    try:
                        results[index] = _single(index)
                    except Exception as inner:
                        _mark_degraded(index, inner)
            else:
                results.update(zip(group, outputs))
        done += len(group)
        group_timings.append({
            "group": list(group),
            "seconds": round(time.perf_counter() - group_started, 3),
            "audio_seconds": round(sum(seconds[i] for i in group), 2),
        })
        _emit_event("asr_window", index=done, total=len(pieces))

    segments = [
        Segment(flag_empty_transcript(results[p.index], seconds[p.index]),
                p.core_start_s, p.core_end_s, p.cut_silence_ms,
                budgets.setdefault(p.index, resolve_max_new_tokens(
                    args.max_new_tokens, seconds[p.index], chunked=True)))
        for p in pieces
    ]
    if merge:
        # 每个边界的重叠音频秒数 -> 大致字符数，供去重限定裁剪长度
        expected = []
        for left, left_seg, right, right_seg in zip(
                pieces, segments, pieces[1:], segments[1:]):
            shared = (max(0.0, left.audio_end_ms / 1000.0 - left.core_end_s)
                      + max(0.0, right.core_start_s - right.audio_start_ms / 1000.0))
            rates = []
            for piece, seg in ((left, left_seg), (right, right_seg)):
                duration = max(1e-6, piece.core_end_s - piece.core_start_s)
                rates.append(effective_len(seg.result.text) / duration)
            expected.append(int(round(shared * (rates[0] + rates[1]) / 2)))
        # 去重只作用于合并全文（导出/复制）；各 chunk 原文原样进 stats，
        # 由服务层 build_segments_from_stats 按 dedup 记录统一裁一次后入库。
        # 切勿在此处回写 segments：会造成入库文本被裁两遍（句首误裁）。
        text, dedup = merge_texts([s.result.text for s in segments],
                                  expected_overlap=expected,
                                  fuzzy=getattr(args, "dedup_mode", "fuzzy") != "exact")
    else:
        text, dedup = "\n".join(s.result.text for s in segments), []

    # 波动治理：以「耗时 / 音频秒」的中位数为基线，超过 2× 的批次标记出来
    # （共享显存、降频、批内长度不均已排除）。仅观测，不改变识别结果。
    timed = [(t, t["seconds"] / max(t["audio_seconds"], 1e-6)) for t in group_timings]
    rates = sorted(rate for _, rate in timed)
    median_rate = rates[len(rates) // 2] if rates else 0.0
    slow_groups = [t for t, rate in timed if median_rate > 0 and rate > 2 * median_rate]

    batch_stats = {
        "requested_size": args.batch_size,
        "auto": bool(args.batch_auto),
        "limit_size": batch_size,
        "applied_size": max((len(group) for group in groups), default=1),
        "tolerance": args.batch_tolerance,
        "max_batch": args.batch_max,
        "max_driver_mb": args.batch_max_driver_mb,
        "driver_mb_before": round(driver_mb, 0),
        "groups": groups,
        "fallbacks": fallbacks,
        "group_timings": group_timings,
        "median_seconds_per_audio_second": round(median_rate, 4),
        "slow_groups": slow_groups,
    }
    return segments, text, batch_stats, dedup, paths


def sample_window_texts(processor, model, audio_path: str, language: str | None,
                        device: str, max_new_tokens: int, prompt: str | None,
                        count: int) -> list[str]:
    """同一段音频采样多次，返回候选文本（P2 采样一致性信号）。

    仅在后处理阶段按需调用；默认关闭，开启后推理成本约为 ×count。
    """
    import torch

    texts, wavs = _prepare_asr_inputs(processor, [audio_path], prompt, language)
    inputs = processor(text=texts, audio=wavs, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.7, top_p=0.9, num_return_sequences=count)
    seqs = output.sequences[:, inputs["input_ids"].shape[1]:]
    raws = processor.batch_decode(
        seqs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [_parse_raw_output(r, language) for r in raws]


def load_forced_aligner(path: str, device: str):
    """加载 Qwen3-ForcedAligner；失败由调用方降级为无对齐特征。

    常驻 worker 下按 (路径, device) 缓存：同一模型连续任务复用，
    省掉每次 2–9s 的加载（长音频占比虽小，短音频几乎翻倍）。
    """
    import torch
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

    key = f"{Path(path).resolve()}|{device}"
    cached = _ALIGNER_CACHE.get(key)
    if cached is not None:
        print("[qwen3-asr] 复用常驻对齐模型", flush=True)
        return cached
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    aligner = Qwen3ForcedAligner.from_pretrained(
        path, dtype=dtype, local_files_only=True)
    aligner.model.to(device)
    _ALIGNER_CACHE[key] = aligner
    return aligner


def collect_aux_signals(processor, model, args, language: str | None,
                        paths: dict[int, Path],
                        segments: list["Segment"]) -> tuple[dict, dict, dict]:
    """收集可选的采样一致性、强制对齐特征与字符级时间戳（默认全部关闭）。

    一次对齐同时产出时长特征（置信度 recall 档）与字符时间（句级跳转），
    避免对同一块音频重复推理。
    """
    sample_texts: dict[int, list[str]] = {}
    count = int(getattr(args, "sample_check", 0) or 0)
    if count > 0:
        for i, seg in enumerate(segments):
            if not seg.result.text.strip():
                continue
            try:
                sample_texts[i] = sample_window_texts(
                    processor, model, str(paths[i]), language, args.device,
                    seg.max_new_tokens, args.prompt, count)
            except Exception as exc:  # noqa: BLE001 - 附加信号失败不阻断转写
                print(f"[qwen3-asr] 警告: 第 {i} 段采样一致性失败: {exc}",
                      file=sys.stderr, flush=True)
        print(f"[qwen3-asr] 采样一致性: {count} 次/段，覆盖 {len(sample_texts)} 段",
              flush=True)

    align_feats: dict[int, list[dict | None]] = {}
    char_times: dict[int, list[tuple[float, float] | None]] = {}
    aligner_path = getattr(args, "forced_aligner", None)
    if aligner_path:
        try:
            aligner = load_forced_aligner(str(aligner_path), args.device)
        except Exception as exc:  # noqa: BLE001 - 可选增强
            print(f"[qwen3-asr] 警告: ForcedAligner 加载失败，跳过对齐: {exc}",
                  file=sys.stderr, flush=True)
        else:
            covered = total = 0
            for i, seg in enumerate(segments):
                if not seg.result.text.strip():
                    continue
                align_feats[i], char_times[i] = align_char_signals(
                    aligner, str(paths[i]), seg.result.text,
                    language or "Chinese",
                    max_seconds=audio_duration(paths[i]))
                covered += sum(1 for t in char_times[i] if t is not None)
                total += len(char_times[i])
            print(f"[qwen3-asr] 强制对齐: 覆盖 {len(align_feats)} 段，"
                  f"字符时间 {covered}/{total}（{covered / max(total, 1):.1%}）",
                  flush=True)
    return sample_texts, align_feats, char_times


class GpuSampler:
    """推理期间后台采样 GPU 状态，用于定位耗时波动。

    实测波动主要来自两类环境因素：显存溢出到 WDDM 共享内存、GPU 降频。
    每 2s 采一次 ``utilization/memory/temperature/sm_clock``，结束时
    汇总峰值与降频判定；没有 ``nvidia-smi``（CPU / MPS）时静默禁用，
    不让观测代码影响主流程。
    """

    _QUERY = "utilization.gpu,memory.used,temperature.gpu,clocks.sm"

    def __init__(self, device: str, interval: float = 2.0) -> None:
        self.enabled = device == "cuda"
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        first = self._sample()
        if first is None:  # nvidia-smi 不可用：禁用而不是每 2s 报错
            self.enabled = False
            return
        self.samples.append(first)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            sample = self._sample()
            if sample is not None:
                self.samples.append(sample)

    @classmethod
    def _sample(cls) -> dict | None:
        try:
            proc = subprocess.run(
                ["nvidia-smi", f"--query-gpu={cls._QUERY}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
        try:
            return {
                "utilization_pct": int(parts[0]),
                "memory_mb": int(parts[1]),
                "temperature_c": int(parts[2]),
                "sm_clock_mhz": int(parts[3]),
            }
        except (IndexError, ValueError):
            return None

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if not self.samples:
            return {"enabled": False}
        mem = [s["memory_mb"] for s in self.samples]
        util = [s["utilization_pct"] for s in self.samples]
        temp = [s["temperature_c"] for s in self.samples]
        clock = [s["sm_clock_mhz"] for s in self.samples]
        # 降频判定只看 GPU 真正在工作（利用率 ≥30%）的样本：
        # 切块/后处理阶段的空闲采样常降到几百 MHz，直接取全样本会误报。
        active = [s["sm_clock_mhz"] for s in self.samples
                  if s["utilization_pct"] >= 30]
        return {
            "enabled": True,
            "samples": len(self.samples),
            "active_samples": len(active),
            "utilization_mean_pct": round(sum(util) / len(util), 1),
            "memory_peak_mb": max(mem),
            "temperature_peak_c": max(temp),
            "sm_clock_min_mhz": min(clock),
            "sm_clock_max_mhz": max(clock),
            # 降频：工作时间的最低 SM 频率低于最高值 80%（笔记本功耗墙）
            "throttled": (len(active) >= 2 and max(active) > 0
                          and min(active) < 0.8 * max(active)),
        }


def check_device(device: str) -> None:
    """校验目标加速器真实可用；不可用时立刻失败，不静默回退 cpu。

    静默回退会让 32 分钟音频在 CPU 上跑数小时，宁可报错让用户显式改参数。
    放在 preflight（加载模型之前）调用，避免浪费几十秒的加载时间。
    """
    import torch
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前环境的 MPS 不可用，请显式选择 cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境的 CUDA 不可用，请显式选择 cpu")


@dataclass
class RunPlan:
    """preflight 的产物：跑之前就确定的全部输入与策略。"""

    audio: Path
    output: Path
    raw: Path
    duration: float
    mode: str  # window | vad | full
    config: SegmentationConfig
    vad_model: Path


def segmentation_config(args) -> SegmentationConfig:
    """把 CLI 参数装配成切分配置（参数名与配置字段一一对应）。

    ``getattr`` 兜底是给测试与编排层留的：构造简化 args 时
    缺省字段也能得到与 CLI 默认值一致的行为。
    """
    return SegmentationConfig(
        mode=args.segmentation,
        full_max_seconds=args.full_max_seconds,
        target_chunk_seconds=args.target_chunk,
        max_chunk_seconds=args.max_chunk,
        min_silence_ms=args.min_silence_ms,
        overlap_seconds=args.overlap,
        chunk_strategy=getattr(args, "chunk_strategy", "silence"),
        balance_tolerance=getattr(args, "balance_tolerance", 0.15),
        balance_min_silence_ms=getattr(args, "balance_min_silence_ms", 40),
        balance_jitter_seconds=getattr(args, "balance_jitter", 0.0),
        dp_forced_penalty=getattr(args, "dp_forced_penalty", 0.3),
        dp_gap_bonus=getattr(args, "dp_gap_bonus", 0.15),
    )


def adaptive_strategy_pipeline(
    pcm: Path, vad_segments_raw: list[VadSegment], total_ms: int,
    config: SegmentationConfig,
) -> tuple[SegmentationConfig, StrategyDecision, list[VadSegment]]:
    """自适应策略流水线：VAD 后处理 → 特征提取 → 策略决策。"""
    # 1. VAD 后处理：合并短间隙（用于下游切分）
    vad_segments = merge_short_gaps(vad_segments_raw)

    # 2. 音频质量检测
    audio_profile = profile_audio(pcm)

    # 3. VAD 特征提取（用原始段保留停顿密度信号，不做合并）
    vad_profile = extract_vad_profile(vad_segments_raw, total_ms)

    # 4. 策略决策
    decision = decide_strategy(audio_profile, vad_profile)

    # 5. 应用决策到 config
    new_config = apply_decision_to_config(decision, config)

    return new_config, decision, vad_segments


def preflight(args) -> RunPlan:
    """识别前的全部检查；任何一项不满足都在加载模型前失败。"""
    audio, output = Path(args.audio), Path(args.output)
    if not audio.is_file():
        raise FileNotFoundError(f"音频不存在: {audio}")
    if args.window < 0 or (args.max_new_tokens is not None and args.max_new_tokens <= 0):
        raise ValueError("window 不能为负数，max-new-tokens 必须为正数")
    if not 0.0 <= getattr(args, "low_conf_ratio", DEFAULT_MAX_RATIO) <= 0.5:
        raise ValueError("low-conf-ratio 必须在 0~0.5 之间")
    if not 0.0 <= getattr(args, "low_conf_floor", LOW_CONF_SCORE_FLOOR) <= 5.0:
        raise ValueError("low-conf-floor 必须在 0~5 之间")
    if not 1 <= getattr(args, "beam_size", 1) <= 8:
        raise ValueError("beam-size 必须在 1~8 之间")
    if not 0 <= getattr(args, "sample_check", 0) <= 4:
        raise ValueError("sample-check 必须在 0~4 之间")
    aligner_path = getattr(args, "forced_aligner", None)
    if aligner_path and not Path(aligner_path).is_dir():
        raise FileNotFoundError(f"ForcedAligner 模型目录不存在: {aligner_path}")

    raw = Path(args.raw_output) if args.raw_output else output
    paths = {"output": output, "asr_stats": output.with_suffix(".stats.json")}
    if not same_path(raw, output):
        paths["raw"] = raw
    preflight_paths(paths, inputs=(audio,))

    validate_local_model(Path(args.model), asr=True)
    require_module("torch", "请用 .venv-funasr/bin/python 执行识别。")
    require_module("transformers", "请用 .venv-funasr/bin/python 执行识别。")

    config = segmentation_config(args)
    if args.batch_size < 1:
        raise ValueError("batch-size 必须 >= 1")
    try:
        BatchConfig(tolerance=args.batch_tolerance, max_batch=args.batch_max,
                    max_driver_mb=args.batch_max_driver_mb)
    except ValueError as exc:
        raise ValueError(f"batch 参数非法: {exc}") from exc
    if not is_pcm16k_mono_wav(audio) and not can_convert_audio():
        raise RuntimeError(
            "输入不是 16 kHz 单声道 WAV，且既没有 ffmpeg 也没有 PyAV；请先转换：\n"
            f"  ffmpeg -i {audio} -ac 1 -ar 16000 -c:a pcm_s16le out.wav")
    duration = audio_duration(audio)
    check_device(args.device)
    if args.window and args.window > 0:
        mode = "window"
    else:
        mode = decide_mode(duration, config)

    if args.vad_model:
        vad_model = Path(args.vad_model)
    else:
        # 默认模型目录兼容两种布局；backend/models 下可能只有 snapshots 空壳
        cands = [PROJECT_ROOT / "models/funasr-fsmn-vad",
                 PROJECT_ROOT.parent / "models/funasr-fsmn-vad"]
        vad_model = next((c for c in cands if (c / "config.yaml").is_file()),
                         cands[0])
    if mode == "vad":
        require_module("funasr", "VAD 切分需要 funasr；请用 .venv-funasr/bin/python 执行识别。")
        if not vad_model.is_dir():
            raise FileNotFoundError(f"VAD 模型不存在: {vad_model}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return RunPlan(audio, output, raw, duration, mode, config, vad_model)


def main(argv: list[str] | None = None) -> int:
    """命令行入口；``argv=None`` 读 ``sys.argv``。

    显式接收 argv 让常驻 worker 直接复用同一进程（见
    ``qwen3_asr_worker.py``），无需改写 ``sys.argv``。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--output", required=True, help="原始文本输出路径（即事实底稿）")
    ap.add_argument("--language", default="Chinese", help="语言提示，如 Chinese / 空字符串表示自动检测")
    ap.add_argument("--prompt", default=None, help="上下文/格式提示，引导数字写法等")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"], help="mps / cpu / cuda")
    ap.add_argument("--segmentation", default="auto", choices=["auto", "full", "vad"],
                    help="切分策略：auto 按时长自动选择；full 整段识别；vad 按语音边界切分")
    ap.add_argument("--strategy", default="manual", choices=["manual", "adaptive"],
                    help="切片策略选择：manual 按下面参数手动控制；adaptive 基于 VAD 分析自动选择"
                         "最优方案（silence/balanced/jitter/batch）")
    ap.add_argument("--full-max-seconds", type=float, default=300.0,
                    help="auto 模式下允许整段识别的时长上限（秒）")
    ap.add_argument("--target-chunk", type=float, default=120.0,
                    help="VAD 分块的目标长度（秒）")
    ap.add_argument("--max-chunk", type=float, default=180.0,
                    help="目标区间内无可用静音时的硬切上限（秒）")
    ap.add_argument("--min-silence-ms", type=int, default=200,
                    help="可作为切点的最小静音长度（毫秒）")
    ap.add_argument("--chunk-strategy", default="silence",
                    choices=["silence", "balanced", "dp"],
                    help="silence：从当前块找最长静音切（原策略）；"
                         "balanced：先均分块数，把切点逐点吸附到理想位置附近的静音；"
                         "dp：全局动态规划，在「块长偏离目标」与「切在静音上」之间求总代价最小（长音频）")
    ap.add_argument("--dp-forced-penalty", type=float, default=0.3,
                    help="dp：硬切（切点不在静音上）的代价；越大越偏向块长均匀")
    ap.add_argument("--dp-gap-bonus", type=float, default=0.15,
                    help="dp：切在静音上的奖励；越大越偏向吸附静音")
    ap.add_argument("--balance-tolerance", type=float, default=0.15,
                    help="balanced 策略下切点偏离理想位置的比例上限")
    ap.add_argument("--balance-min-silence-ms", type=int, default=40,
                    help="balanced 策略下可吸附的最小静音（毫秒）")
    ap.add_argument("--balance-jitter", type=float, default=0.0,
                    help="balanced 策略下切点在理想位置附近的微调范围（秒），"
                         "用于避开词边界；实测 0.25 常找不到停顿，建议 1~2")
    ap.add_argument("--overlap", type=float, default=1.5,
                    help="每块两侧扩展的重叠秒数（边界声学上下文，去重惰性触发）；"
                         "0.5 时边界词常被截断且短重复无法去重，1.5 实测明显更连贯")
    ap.add_argument("--dedup-mode", default="fuzzy", choices=["fuzzy", "exact"],
                    help="边界去重：fuzzy（精确失败后模糊对齐）/ exact（只做精确后缀匹配）")
    ap.add_argument("--vad-model", default=None,
                    help="FunASR VAD 模型目录；默认 models/funasr-fsmn-vad")
    ap.add_argument("--window", type=float, default=0.0,
                    help="[已弃用] 固定窗口秒数；请改用 --segmentation。0 表示不切分")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="生成 token 上限；默认整段 4096、分块按块长自动估算")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="批大小上限；1 表示逐条识别（默认，产物与逐条完全一致）")
    ap.add_argument("--batch-auto", action="store_true",
                    help="按块长与 MPS 内存自动选择批大小（最多 --batch-max）")
    ap.add_argument("--batch-tolerance", type=float, default=0.2,
                    help="同一批内块长的相对差上限")
    ap.add_argument("--batch-max", type=int, default=3,
                    help="自动模式下的批大小上限（16 GB 机器实测 3）")
    ap.add_argument("--batch-max-driver-mb", type=int, default=8000,
                    help="MPS driver 内存上限，超过则自动降档")
    ap.add_argument("--raw-output", default=None,
                    help="原始文本检查点路径；默认与 --output 相同，禁止覆盖不同内容")
    ap.add_argument("--dump-token-features", action="store_true",
                    help="调试用：把每个信息字符的 entropy 写入 stats.json（char_entropies）")
    ap.add_argument("--low-conf-mode", default="balanced",
                    choices=["balanced", "quality", "recall"],
                    help="低置信标亮档位：balanced=entropy+全局预算（默认）；"
                         "quality=采样一致性加权（适合约 1%% 覆盖率）；"
                         "recall=强制对齐异常加权（适合 8%%-10%% 覆盖率）")
    ap.add_argument("--low-conf-floor", type=float, default=LOW_CONF_SCORE_FLOOR,
                    help="候选最低分（熵）；<0.8 为平衡档，0.8~1.2 为高精度档")
    ap.add_argument("--low-conf-ratio", type=float, default=DEFAULT_MAX_RATIO,
                    help="全片标亮预算（信息字符占比），默认 0.01（精度优先；"
                         "旧平衡档 0.04 误报明显偏多）")
    ap.add_argument("--beam-size", type=int, default=1,
                    help="beam search 宽度；>1 时启用 beam 并提取 rank 特征。"
                         "bench 实测本模型 rank 无增量信息，默认 1（greedy）")
    ap.add_argument("--sample-check", type=int, default=0,
                    help="对每段额外采样 N 次用于一致性特征（成本 ×N，默认 0 关闭）")
    ap.add_argument("--cross-check", nargs="?", const="auto", default=None,
                    help="第二模型交叉验证：auto=现场调用 FunASR Paraformer；"
                         "也可传入已生成的第二模型文本路径。两模型不一致的字符"
                         "在低置信打分中固定加分（bench 召回约翻倍，默认关闭）")
    ap.add_argument("--forced-aligner", default=None,
                    help="Qwen3-ForcedAligner 模型目录；启用后同时产出字符级"
                         "时间戳（stats.char_times，供句级跳转）与字符时长异常"
                         "（额外加载 0.6B 模型，默认关闭）")
    args = ap.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"[qwen3-asr] 失败: {exc}", file=sys.stderr)
        return 2


def run(args) -> int:
    """识别主流程：preflight → 加载模型 → 按模式解码 → 发布产物与统计。

    三种执行模式（由 preflight 决定）：

    - ``vad``：VAD 语音段 → 切分计划 → 切块 → （可选 batch）逐组识别 →
      边界去重合并成连续文本。长音频的主路径。
    - ``window``：已弃用的固定等长切分，无重叠、不合并（按换行拼接），
      仅为兼容旧参数保留。
    - ``full``：短音频整段一次识别。

    产物发布顺序是刻意的：先 ``publish_text(raw, immutable=True)`` 落
    事实底稿，再写 stats。stats 写失败时 raw 已保住，重跑可复用
    （immutable 语义：同内容幂等，不同内容拒绝覆盖）。
    """
    started = time.perf_counter()
    plan = preflight(args)
    audio, out_path, raw_path, duration = plan.audio, plan.output, plan.raw, plan.duration
    manifest = model_manifest(Path(args.model))
    model_config_hash = json_hash(json.loads((Path(args.model) / "config.json").read_text(encoding="utf-8")))

    print(f"[qwen3-asr] 模型: {args.model}", flush=True)
    print(f"[qwen3-asr] 音频: {audio} (device={args.device})", flush=True)
    _emit_event("asr_started", audio_seconds=round(duration, 3), device=args.device,
                model=str(args.model))

    t_load = time.perf_counter()
    processor, model = load_model(args.model, args.device)
    load_s = time.perf_counter() - t_load
    _emit_event("asr_model_loaded", load_seconds=round(load_s, 3))
    print(f"[qwen3-asr] 模型加载完成 耗时={load_s:.1f}s", flush=True)

    language = args.language or None
    segment_stats: dict[str, object] = {
        "requested_mode": args.segmentation,
        "applied": plan.mode,
        "full_max_seconds": plan.config.full_max_seconds,
        "target_chunk_seconds": plan.config.target_chunk_seconds,
        "max_chunk_seconds": plan.config.max_chunk_seconds,
        "min_silence_ms": plan.config.min_silence_ms,
        "overlap_seconds": plan.config.overlap_seconds,
        "chunk_strategy": plan.config.chunk_strategy,
        "balance_tolerance": plan.config.balance_tolerance,
        "balance_jitter_seconds": plan.config.balance_jitter_seconds,
        "dedup_mode": getattr(args, "dedup_mode", "fuzzy"),
        "vad_model": None,
        "vad_segments": None,
        "chunks": 1,
        "forced_chunks": 0,
        "dedup": [],
    }
    batching_stats: dict[str, object] = {
        "requested_size": args.batch_size,
        "auto": bool(args.batch_auto),
        "limit_size": 1,
        "applied_size": 1,
        "tolerance": args.batch_tolerance,
        "max_batch": args.batch_max,
        "max_driver_mb": args.batch_max_driver_mb,
        "driver_mb_before": None,
        "groups": [[0]],
        "fallbacks": [],
    }
    segments: list[Segment] = []
    paths: dict[int, Path] = {}
    # vad/window 模式会填充；full 模式保持为空（整段偏移为 0）
    pieces: list[PiecePlan] = []
    t_infer = time.perf_counter()
    # GPU 状态采样：定位显存溢出/降频导致的耗时波动（CPU/MPS 自动禁用）
    gpu_sampler = GpuSampler(args.device)
    gpu_sampler.start()
    # 临时工作目录：存放归一化 PCM 与切块 WAV，进程退出自动清理；
    # 32 分钟音频 16 kHz PCM16 约 60 MB，不留在项目目录里。
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        # 输入已是 16 kHz 单声道 WAV 时直接返回原路径，不产生副本
        pcm = prepare_pcm16k(audio, work / "input_16k.wav")
        # 交叉验证与主推理并行：CPU FunASR 的固定开销隐藏在识别期间，
        # 主推理结束后通常只需几毫秒等待（长音频可省 40–50s）
        cross_job = start_cross_model(args, pcm, work)
        if plan.mode == "vad":
            # VAD 主路径：语音段检测 → （可选）自适应策略 → 切分计划 → 分块识别
            vad_segments_raw = detect_speech(pcm, plan.vad_model)
            total_ms = int(round(audio_duration(pcm) * 1000))

            # ── 自适应策略流水线 ──
            # manual（默认）：完全按 CLI 参数切分；
            # adaptive：基于音频质量 + VAD 停顿密度特征自动选
            # silence/balanced/jitter/batch 组合，并覆写 args 的 batch 字段。
            strategy_decision: StrategyDecision | None = None
            active_config = plan.config
            vad_segments = vad_segments_raw
            if getattr(args, "strategy", "manual") == "adaptive":
                active_config, strategy_decision, vad_segments = \
                    adaptive_strategy_pipeline(pcm, vad_segments_raw, total_ms, plan.config)
                segment_stats["strategy_adaptive"] = True
                import dataclasses
                segment_stats["strategy_decision"] = dataclasses.asdict(strategy_decision)
                # 应用 batch 决策
                if not strategy_decision.use_batch:
                    args.batch_auto = False
                    args.batch_max = 1
                else:
                    args.batch_auto = True
                    args.batch_max = strategy_decision.batch_max
                print(f"[qwen3-asr] 自适应策略: {strategy_decision.reason}", flush=True)

            # 切分策略二选一：silence=贪心找最长静音（原策略）；
            # balanced=先均分块数再把切点吸附到附近静音（长音频块长更均匀，
            # batch 利用率更高）。切点确定后 build_chunks 负责加 overlap 与毫秒换算。
            if active_config.chunk_strategy == "balanced":
                planner = plan_balanced_cuts
            elif active_config.chunk_strategy == "dp":
                planner = plan_dp_cuts  # 全局最优切分（长音频）
            else:
                planner = plan_cuts
            cuts = planner(vad_segments, total_ms, active_config)
            chunks = build_chunks(cuts, total_ms, active_config)
            forced = sum(1 for c in cuts if c.forced)
            segment_stats.update(vad_model=str(plan.vad_model),
                                 vad_segments=len(vad_segments),
                                 chunks=len(chunks), forced_chunks=forced,
                                 chunk_strategy=active_config.chunk_strategy,
                                 balance_jitter_seconds=active_config.balance_jitter_seconds,
                                 balance_tolerance=active_config.balance_tolerance)
            lengths = [c.end_ms - c.start_ms for c in cuts]
            print(f"[qwen3-asr] VAD {len(vad_segments)} 段 → 切分 {len(chunks)} 块"
                  f"（{active_config.chunk_strategy}，强制 {forced} 块，"
                  f"块长 {min(lengths)/1000:.0f}~{max(lengths)/1000:.0f}s）", flush=True)
            _emit_event("asr_vad_planned", vad_segments=len(vad_segments),
                        chunks=len(chunks), forced_chunks=forced)
            pieces = [
                PiecePlan(ch.index, ch.core_start_ms / 1000, ch.core_end_ms / 1000,
                          ch.audio_start_ms, ch.audio_end_ms, ch.cut_silence_ms)
                for ch in chunks
            ]
            segments, text, batching_stats, dedup, paths = decode_pieces(
                processor, model, args, language, work, pcm, pieces, merge=True)
            segment_stats["dedup"] = dedup
            for piece, seg in zip(pieces, segments):
                print(f"[qwen3-asr] {piece.index + 1}/{len(pieces)} "
                      f"[{seg.start_s:.1f}-{seg.end_s:.1f}s]: {seg.result.text[:40]}...",
                      flush=True)
        elif plan.mode == "window":
            # 已弃用的固定等长切分：无 VAD、无重叠、不合并（换行拼接），
            # 保留仅为兼容旧参数；新调用一律走 vad 或 full。
            print("[qwen3-asr] 警告: --window 已弃用，请改用 --segmentation vad",
                  file=sys.stderr, flush=True)
            count = max(1, math.ceil(duration / args.window))
            bounds = [(i * args.window, min((i + 1) * args.window, duration))
                      for i in range(count)]
            segment_stats["chunks"] = len(bounds)
            pieces = [
                PiecePlan(index, start_s, end_s, int(start_s * 1000), int(end_s * 1000))
                for index, (start_s, end_s) in enumerate(bounds)
            ]
            segments, text, batching_stats, _, paths = decode_pieces(
                processor, model, args, language, work, pcm, pieces, merge=False)
            for piece, seg in zip(pieces, segments):
                print(f"[qwen3-asr] {piece.index + 1}/{len(pieces)}: "
                      f"{seg.result.text[:40]}...", flush=True)
        else:
            # full 模式：短音频（≤ full_max_seconds）整段一次识别。
            # flag_empty_transcript：长音频整段推理可能触发模型单次推理
            # 长度上限、只生成 EOS 返回空文本——空结果必须标记 degraded，
            # 不能当成「音频里没人说话」。
            budget = resolve_max_new_tokens(args.max_new_tokens, duration, chunked=False)
            result = decode_window(processor, model, str(pcm), language, args.device,
                                   budget, args.prompt)
            result = flag_empty_transcript(result, duration)
            segments = [Segment(result, 0.0, duration, None, budget)]
            text = result.text
            paths = {0: pcm}

        # P2 可选信号：采样一致性 / 强制对齐（默认均关闭）。
        # 放在临时目录内，复用切好的块 WAV，避免重复切音频。
        t_aux = time.perf_counter()
        sample_texts, align_feats, char_times = collect_aux_signals(
            processor, model, args, language, paths, segments)
        aux_s = time.perf_counter() - t_aux
        # 可选：第二模型交叉验证（FunASR Paraformer）。失败只告警不阻塞。
        cross_flags = None
        cross_info = None
        if getattr(args, "cross_check", None):
            t_cross = time.perf_counter()
            cross_wall = None
            if cross_job is not None:
                other_text, cross_wall = finish_cross_model(cross_job)
            else:
                other_text = resolve_cross_model_text(args, pcm, work)
            if other_text:
                try:
                    cross_flags = build_cross_flags(
                        text, segments, segment_stats["dedup"], other_text)
                    mismatch = sum(1 for fs in cross_flags.values()
                                   for flag in fs if flag)
                    cross_info = {
                        "enabled": True,
                        "source": args.cross_check,
                        "device": "cpu",
                        "mismatch_chars": mismatch,
                        # seconds=收到结果后的映射耗时；wall_seconds=后台任务
                        # 从启动到可用的总时长（与主推理重叠）
                        "seconds": round(time.perf_counter() - t_cross, 2),
                        "wall_seconds": (round(cross_wall, 2)
                                         if cross_wall is not None else None),
                    }
                    print(f"[qwen3-asr] 交叉验证: 不一致 {mismatch} 字"
                          f"（映射 {cross_info['seconds']}s，"
                          f"后台总计 {cross_info['wall_seconds']}s）", flush=True)
                except Exception as exc:  # 可选信号不能拖垮主流程
                    print(f"[qwen3-asr] 交叉验证映射失败，忽略: {exc}",
                          file=sys.stderr, flush=True)
    # 各块在全局音频中的起点毫秒：对齐时间以块 WAV 为原点，需换算。
    audio_offsets = {p.index: p.audio_start_ms for p in pieces}
    infer_s = time.perf_counter() - t_infer
    gpu_summary = gpu_sampler.stop()

    # 聚合每段的生成状态：任一段触顶/空转写 → 整体 degraded。
    # 逐段发 asr_generation_incomplete 事件，编排层据此在 CLI 里提示复核。
    results = [s.result for s in segments]
    status = aggregate_status(results)
    for i, seg in enumerate(segments):
        if seg.result.status == "completed":
            continue
        _emit_event("asr_generation_incomplete", index=i, status=seg.result.status,
                    generated_tokens=seg.result.generated_tokens,
                    max_new_tokens=seg.max_new_tokens)
        print(f"[qwen3-asr] 警告: 第 {i} 段生成不完整 ({seg.result.status})，"
              f"tokens={seg.result.generated_tokens}/{seg.max_new_tokens}", flush=True)

    # 发布顺序：raw（不可变事实底稿）先落盘，stats 后写。
    # immutable=True 让重跑同内容幂等、不同内容拒绝覆盖——raw 是后处理
    # 与 CER 评估的唯一输入，被悄悄改写会污染整条链路的可比性。
    raw_text = text
    publish_text(raw_path, raw_text, immutable=True)
    if not same_path(raw_path, out_path):
        publish_text(out_path, raw_text)
    print(f"[qwen3-asr] 原始文本检查点: {raw_path} ({len(text)} 字)", flush=True)

    rss_mb = peak_rss_mb()
    rtf = infer_s / duration if duration else float("nan")
    payloads, seg_signals = _build_segment_payloads(
        segments, processor, args, sample_texts, align_feats,
        char_times, audio_offsets, cross_flags)
    review_order = [s.seq for s in sorted(
        seg_signals, key=lambda x: x.risk_score, reverse=True)]
    marked_chars = sum(e - s for sig in seg_signals for s, e in sig.spans)
    stats = {
        "schema_version": 4,
        "stage": "asr",
        "status": status,
        "raw_path": str(raw_path),
        "raw_sha256": text_hash(raw_text),
        "model_path": str(Path(args.model).resolve()),
        "model_config_sha256": model_config_hash,
        "model_manifest": manifest,
        "elapsed_seconds": time.perf_counter() - started,
        "elapsed_scope": "asr_current_invocation_before_stats_publish",
        "parameters": {"language": args.language, "window": args.window,
                       "segmentation": args.segmentation,
                       "max_new_tokens": args.max_new_tokens or "auto",
                       "prompt_sha256": text_hash(args.prompt or "")},
        "memory_scope": "current_process_lifetime",
        "model": Path(args.model).name,
        "device": args.device,
        "audio_seconds": round(duration, 2),
        "load_seconds": round(load_s, 2),
        "infer_seconds": round(infer_s, 2),
        "first_segment_seconds": None,  # 整段自回归解码，无逐段输出
        "rtf": round(rtf, 4),
        "peak_rss_mb": round(rss_mb, 0),
        "chars": len(text),
        "generated_tokens": sum(r.generated_tokens for r in results),
        "hit_generation_limit": any(r.hit_limit for r in results),
        "hit_eos": all(r.hit_eos for r in results) if results else False,
        "segmentation": segment_stats,
        "batching": batching_stats,
        "aux_seconds": round(aux_s, 2),
        # 推理期间的 GPU 采样摘要（无 nvidia-smi 时 enabled=false）
        "gpu": gpu_summary,
        "align_timestamps": bool(char_times),
        "low_conf": {
            "mode": getattr(args, "low_conf_mode", "balanced"),
            "max_ratio": getattr(args, "low_conf_ratio", DEFAULT_MAX_RATIO),
            "score_floor": getattr(args, "low_conf_floor", LOW_CONF_SCORE_FLOOR),
            "marked_chars": marked_chars,
            "beam_size": getattr(args, "beam_size", 1),
            "sample_check": getattr(args, "sample_check", 0),
            "forced_aligner": bool(getattr(args, "forced_aligner", None)),
            "cross_check": cross_info,
        },
        # 按段级风险从高到低，供工作台「下一处可疑」导航
        "review_order": review_order,
        "segments": payloads,
    }
    print(f"[qwen3-asr] 音频 {duration:.1f}s，推理 {infer_s:.1f}s，RTF={rtf:.4f}，"
          f"峰值内存 {rss_mb:.0f}MB，字符 {len(text)}，状态 {status}", flush=True)
    publish_text(out_path.with_suffix(".stats.json"), json.dumps(stats, ensure_ascii=False, indent=2))
    _emit_event("asr_completed", elapsed_seconds=round(time.perf_counter() - started, 3),
                chars=len(text), raw_path=str(raw_path), status=status)
    print(f"[qwen3-asr] 完成，输出: {out_path} ({len(text)} 字)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())