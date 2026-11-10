#!/usr/bin/env python3
"""Qwen3-ASR 输出文本的两阶段后处理入口。

阶段：raw → 语义纠错（可选）→ 修改校验 → 数字规范化。

用法:
    # A 基线：只做数字规范化
    ./.venv/bin/python scripts/qwen3_asr_postprocess.py \
        --input output/bench/qwen3-asr-1.7b_raw.txt \
        --output-dir output/postprocess --stem qwen3-1.7b

    # B 候选：MLX-LM 本地纠错 + 校验 + 数字规范化
    ./.venv/bin/python scripts/qwen3_asr_postprocess.py \
        --input output/bench/qwen3-asr-1.7b_raw.txt \
        --output-dir output/postprocess --stem qwen3-1.7b-cec4 \
        --corrector mlx --corrector-model models/ChineseErrorCorrector4-4B-mlx-8bit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from local_asr.postprocess import PostprocessPipeline, build_corrector  # noqa: E402
from local_asr.postprocess.artifacts import read_text_exact  # noqa: E402
from local_asr.protocol import emit_event as _emit_event  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Qwen3-ASR 原始输出文本（事实底稿）")
    ap.add_argument("--output-dir", default="output/postprocess")
    ap.add_argument("--stem", default=None, help="产物文件前缀，默认取输入文件名")
    ap.add_argument("--corrector", default="none", choices=["none", "mlx"],
                    help="语义纠错器：none（A 基线）/ mlx")
    ap.add_argument("--corrector-model", default=None, help="本地纠错模型目录（mlx 必填）")
    ap.add_argument("--corrector-name", default=None, help="纠错模型展示名")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="单段输出预算；官方 README 用 512")
    ap.add_argument("--target-chars", type=int, default=300)
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--repetition-penalty", type=float, default=1.1,
                    help="重复惩罚；官方 README 要求 1.1，低于 1.0 会触发复读")
    ap.add_argument("--context-sentences", type=int, default=0)
    ap.add_argument("--input-stage", choices=["raw"], default="raw",
                    help="输入阶段声明；必须提供未经后处理的 ASR 原文")
    ap.add_argument("--max-edit-ratio", type=float, default=0.10)
    ap.add_argument("--no-itn", action="store_true",
                    help="跳过数字规范化（用于评估\"仅纠错\"变体）")
    ap.add_argument("--glossary", default=None, help="实体词典 JSON：{\"科创五零\":\"科创50\"}")
    args = ap.parse_args()

    raw_path = Path(args.input).resolve()
    if not raw_path.is_file():
        print(f"[postprocess] 输入不存在: {raw_path}", file=sys.stderr)
        return 2

    try:
        glossary = json.loads(read_text_exact(Path(args.glossary))) if args.glossary else {}
        corrector = build_corrector(
            args.corrector, args.corrector_model, model_name=args.corrector_name,
            max_tokens=args.max_tokens, target_chars=args.target_chars,
            max_input_tokens=args.max_input_tokens, context_sentences=args.context_sentences,
            repetition_penalty=args.repetition_penalty,
        )
        pipeline = PostprocessPipeline(
            corrector=corrector, glossary=glossary,
            max_edit_ratio=args.max_edit_ratio, enable_itn=not args.no_itn,
        )
        stem = args.stem or raw_path.stem.removesuffix("_raw")
        if args.glossary:
            from local_asr.postprocess.artifacts import artifact_paths, preflight_paths
            preflight_paths(artifact_paths(Path(args.output_dir), stem), inputs=(Path(args.glossary),))
        _emit_event("postprocess_started", corrector=args.corrector,
                    itn=not args.no_itn, input=str(raw_path))
        _emit_event("correction_started", corrector=args.corrector)
        result = pipeline.run(
            read_text_exact(raw_path), output_dir=Path(args.output_dir), stem=stem,
            input_path=raw_path, input_stage=args.input_stage,
        )
    except Exception as exc:
        _emit_event("postprocess_failed", stage="postprocess", error=str(exc))
        print(f"[postprocess] 失败: {exc}", file=sys.stderr)
        return 2
    elapsed = result.stats["elapsed_seconds"]

    # 后处理内部是一次调用完成三段，事件按 stats 的分项耗时补发。
    stages = result.stats
    _emit_event("correction_completed",
                elapsed_seconds=stages.get("corrector_load_seconds", 0)
                + stages.get("corrector_generate_seconds", 0),
                edits_total=stages.get("edits_total", 0),
                failed_segments=stages.get("corrector_failed_segments", 0))
    _emit_event("verify_completed",
                elapsed_seconds=stages.get("review_seconds", 0),
                edits_accepted=stages.get("edits_accepted", 0),
                edits_review=stages.get("edits_review", 0))
    if stages.get("itn_enabled"):
        _emit_event("itn_completed", elapsed_seconds=stages.get("itn_seconds", 0),
                    notes=stages.get("itn_notes", 0), edits=stages.get("itn_edits", 0))
    _emit_event("postprocess_completed" if stages.get("status") == "complete"
                else "postprocess_degraded",
                elapsed_seconds=elapsed, status=stages.get("status"),
                final_path=result.paths.get("final"),
                raw_chars=stages.get("raw_chars", 0),
                final_chars=stages.get("final_chars", 0))

    # 运行指标：耗时与峰值内存（含模型加载），写入 stats 便于横向对比
    runtime = result.stats
    print(f"[postprocess] 状态: {runtime['status']}")
    for reason in runtime["degradations"]:
        print(f"[postprocess] 降级: {reason}", file=sys.stderr)

    print(f"[postprocess] 纠错器: {result.stats['corrector']}")
    print(f"[postprocess] 耗时 {elapsed:.1f}s，峰值内存 {runtime['peak_rss_mb']:.0f} MB")
    print(f"[postprocess] 编辑 {result.stats['edits_total']} 条："
          f"接受 {result.stats['edits_accepted']}，待复核 {result.stats['edits_review']}")
    print(f"[postprocess] 数字复核项 {result.stats['itn_notes']}")
    print(f"[postprocess] 字符：raw {result.stats['raw_chars']} → "
          f"corrected {result.stats['corrected_chars']} → final {result.stats['final_chars']}")
    for key, path in result.paths.items():
        print(f"[postprocess] {key}: {path}")
    return 3 if result.stats["status"] == "degraded" else 0


if __name__ == "__main__":
    sys.exit(main())