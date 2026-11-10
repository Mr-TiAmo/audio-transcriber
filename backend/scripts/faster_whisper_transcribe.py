#!/usr/bin/env python3
"""使用 faster-whisper 转录音频，并记录性能指标（与 FunASR/Qwen3-ASR 测评脚本同口径）。

用法:
    ./.venv/bin/python scripts/faster_whisper_transcribe.py \
        --model models/faster-whisper-large-v3 \
        --audio /private/tmp/测试音频_16k.wav \
        --output output/测试音频_large-v3.txt

说明:
- 严格离线：仅从本地目录加载模型
- 逐段迭代 segments，额外记录"首段延迟"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 与主转写脚本共用音频探测与内存统计（纯标准库，不引入额外依赖）
from local_asr.audio import audio_duration  # noqa: E402
from local_asr.postprocess.artifacts import peak_rss_mb  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="本地模型目录或模型名")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cpu", help="cpu / cuda（CTranslate2 不支持 mps）")
    ap.add_argument("--compute-type", default="int8", help="int8 / int8_float16 / float16 / float32")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--no-vad", action="store_true", help="关闭 VAD 过滤（默认开启以对齐其他后端）")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    audio = Path(args.audio).resolve()
    out_path = Path(args.output).resolve()

    print(f"[faster-whisper] 模型: {args.model} device={args.device} compute={args.compute_type}", flush=True)
    t0 = time.perf_counter()
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        download_root=str(ROOT / "models"),
        local_files_only=True,
    )
    load_s = time.perf_counter() - t0
    print(f"[faster-whisper] 加载完成 耗时={load_s:.1f}s", flush=True)

    duration = audio_duration(audio)
    t1 = time.perf_counter()
    segments_iter, info = model.transcribe(
        str(audio),
        language=args.language,
        task="transcribe",
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=False,
    )

    texts: list[str] = []
    first_segment_s = None
    for seg in segments_iter:
        if first_segment_s is None:
            first_segment_s = time.perf_counter() - t1
        texts.append(seg.text.strip())
    infer_s = time.perf_counter() - t1
    text = "\n".join(t for t in texts if t)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    rtf = infer_s / duration if duration else float("nan")
    stats = {
        "model": Path(args.model).name,
        "device": args.device,
        "compute_type": args.compute_type,
        "language": info.language,
        "audio_seconds": round(duration, 2),
        "load_seconds": round(load_s, 2),
        "infer_seconds": round(infer_s, 2),
        "first_segment_seconds": round(first_segment_s, 2) if first_segment_s else None,
        "rtf": round(rtf, 4),
        "peak_rss_mb": round(peak_rss_mb(), 0),
        "chars": len(text),
        "segments": len(texts),
    }
    print(f"[faster-whisper] 音频 {duration:.1f}s，推理 {infer_s:.1f}s，RTF={rtf:.4f}，"
          f"首段 {stats['first_segment_seconds']}s，峰值内存 {stats['peak_rss_mb']:.0f}MB，字符 {len(text)}", flush=True)
    print(f"[faster-whisper] 输出: {out_path}", flush=True)
    out_path.with_suffix(".stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())