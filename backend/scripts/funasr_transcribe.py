#!/usr/bin/env python3
"""使用 FunASR 转录音频（中文识别候选模型）。

支持的候选模型（均从本地目录加载，不触发联网下载）：
    paraformer-zh   SeACo-Paraformer + fsmn-vad + ct-punc（工业级基线，带标点与时间戳）
    sensevoice      SenseVoiceSmall + fsmn-vad（速度型，自带标点/ITN）
    fun-asr-nano    Fun-ASR-Nano-2512 + fsmn-vad（2025-12 端到端大模型，自带标点）

用法:
    ./.venv-funasr/bin/python scripts/funasr_transcribe.py \
        --model sensevoice --audio /private/tmp/测试音频_16k.wav \
        --output output/测试音频_funasr-sensevoice.txt --device mps
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resolve_model(path_str: str) -> Path:
    """模型目录兼容两种布局：backend/models 与仓库根 models。

    下载脚本与 Qwen 引擎把模型放在仓库根 ``models/``，早期 FunASR 脚本
    按 ``backend/models`` 解析；两处都查，且优先返回「真正有模型文件」的
    目录（backend/models 下可能残留只有 snapshots 的空壳，会让 FunASR 报
    model not registered）。
    """
    candidates = [ROOT / path_str, ROOT.parent / path_str]
    markers = ("config.yaml", "configuration.json")
    for cand in candidates:
        if any((cand / m).is_file() for m in markers):
            return cand
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]
sys.path.insert(0, str(ROOT / "src"))

# 共享工具：音频时长探测与峰值内存统计（跨平台正确处理 ru_maxrss 单位）
from local_asr.audio import audio_duration  # noqa: E402
from local_asr.postprocess.artifacts import peak_rss_mb  # noqa: E402

# 每个候选模型的本地目录与配套子模型
MODEL_PRESETS: dict[str, dict] = {
    "paraformer-zh": {
        "asr": "models/funasr-paraformer-zh",
        "vad": "models/funasr-fsmn-vad",
        "punc": "models/funasr-ct-punc",
        "native_punc": False,
    },
    "sensevoice": {
        "asr": "models/funasr-sensevoice-small",
        "vad": "models/funasr-fsmn-vad",
        "punc": None,
        "native_punc": True,
    },
    "fun-asr-nano": {
        "asr": "models/funasr-fun-asr-nano-2512",
        "vad": "models/funasr-fsmn-vad",
        "punc": None,
        "native_punc": True,
    },
}

_TAG_RE = re.compile(r"<\|[^|]*\|>")


def clean_text(text: str) -> str:
    """去掉 SenseVoice 等模型输出的特殊标签（<|zh|><|NEUTRAL|><|Speech|>...）。"""
    return _TAG_RE.sub("", text).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_PRESETS),
                    help="候选模型: " + " / ".join(sorted(MODEL_PRESETS)))
    ap.add_argument("--audio", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="mps", help="mps / cpu / cuda:0")
    ap.add_argument("--batch-size-s", type=int, default=300,
                    help="动态批处理音频总时长（秒）")
    ap.add_argument("--language", default="auto", help="语言提示: auto / zh / en")
    ap.add_argument("--itn", default=True, action=argparse.BooleanOptionalAction,
                    help="对支持 ITN 的模型（SenseVoice）启用逆文本规范化")
    ap.add_argument("--hotword", default=None, help="热词，逗号分隔（仅 paraformer-zh 生效）")
    ap.add_argument("--json-output", default=None, help="可选：保存含时间戳的完整结果 JSON")
    args = ap.parse_args()

    preset = MODEL_PRESETS[args.model]
    asr_dir = resolve_model(preset["asr"])
    if not asr_dir.exists():
        print(f"[funasr] 模型目录不存在: {asr_dir}", file=sys.stderr)
        return 2

    audio = Path(args.audio).resolve()
    if not audio.exists():
        print(f"[funasr] 音频不存在: {audio}", file=sys.stderr)
        return 2

    from funasr import AutoModel

    kwargs: dict = {
        "model": str(asr_dir),
        "device": args.device,
        "disable_update": True,
        "disable_pbar": True,
    }
    if preset["vad"]:
        kwargs["vad_model"] = str(resolve_model(preset["vad"]))
        kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
    if preset["punc"]:
        kwargs["punc_model"] = str(resolve_model(preset["punc"]))

    print(f"[funasr] 模型: {args.model} ({asr_dir.name}) device={args.device}", flush=True)
    t0 = time.perf_counter()
    model = AutoModel(**kwargs)
    load_s = time.perf_counter() - t0
    print(f"[funasr] 加载完成 耗时={load_s:.1f}s", flush=True)

    gen_kwargs: dict = {
        "input": str(audio),
        "batch_size_s": args.batch_size_s,
        "language": args.language,
    }
    if args.model == "sensevoice":
        gen_kwargs["use_itn"] = args.itn
    if args.hotword and args.model == "paraformer-zh":
        gen_kwargs["hotword"] = args.hotword

    duration = audio_duration(audio)
    t1 = time.perf_counter()
    result = model.generate(**gen_kwargs)
    infer_s = time.perf_counter() - t1

    texts: list[str] = []
    for item in result or []:
        if isinstance(item, dict):
            texts.append(clean_text(str(item.get("text", ""))))
        elif isinstance(item, str):
            texts.append(clean_text(item))
    text = "\n".join(t for t in texts if t)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    peak_mb = peak_rss_mb()
    rtf = infer_s / duration if duration else float("nan")
    stats = {
        "model": args.model,
        "device": args.device,
        "audio_seconds": round(duration, 2),
        "load_seconds": round(load_s, 2),
        "infer_seconds": round(infer_s, 2),
        "rtf": round(rtf, 4),
        "peak_rss_mb": round(peak_mb, 0),
        "chars": len(text),
    }
    print(f"[funasr] 音频 {duration:.1f}s，推理 {infer_s:.1f}s，RTF={rtf:.4f}，"
          f"峰值内存 {peak_mb:.0f}MB，字符 {len(text)}", flush=True)
    print(f"[funasr] 输出: {out_path}", flush=True)
    (out_path.with_suffix(".stats.json")).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json_output:
        json_path = Path(args.json_output).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[funasr] 完整结果: {json_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())