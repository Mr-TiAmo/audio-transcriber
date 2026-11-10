"""两阶段后处理编排：语义纠错 → 修改校验 → 数字规范化。

产物（同一 stem）：
- ``*_raw.txt``         ASR 原始输出，事实底稿，不可被覆盖
- ``*_corrected.txt``   语义纠错 + 校验后的文本（保留数字原写法）
- ``*_final.txt``       数字规范化后的阅读版本
- ``*_edits.jsonl``     每条修改的定位、前后文本、接受/复核与原因
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from local_asr.postprocess.artifacts import (
    artifact_paths, json_hash, model_manifest, peak_rss_mb, preflight_paths, publish_text,
    runtime_versions, source_fingerprint, text_hash,
)
from local_asr.postprocess.corrector import (
    BaseCorrector, CorrectionSegment, PassthroughCorrector, require_module,
)
from local_asr.postprocess.itn import (
    ItnNote, normalize_numbers_segmented, preflight_itn,
)
from local_asr.postprocess.verify import Edit, numeric_spans, replay_edits, review_corrections

__all__ = ["PostprocessPipeline", "PostprocessResult"]


def _percentiles(values: list[int], points: tuple[int, ...] = (50, 90, 95, 99)) -> dict:
    """最近秩百分位；样本很少时只用于观察分布，不作为统计结论。"""
    if not values:
        return {}
    ordered = sorted(values)
    result = {}
    for point in points:
        index = min(len(ordered) - 1, max(0, math.ceil(point / 100 * len(ordered)) - 1))
        result[f"p{point}"] = ordered[index]
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class PostprocessResult:
    """后处理结果。"""

    raw_text: str
    corrected_text: str
    final_text: str
    edits: list[Edit] = field(default_factory=list)
    itn_notes: list[ItnNote] = field(default_factory=list)
    itn_edits: list[Edit] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)

    @property
    def accepted_count(self) -> int:
        return sum(1 for e in self.edits if e.decision == "accepted")

    @property
    def review_count(self) -> int:
        return sum(1 for e in self.edits if e.decision == "review")


class PostprocessPipeline:
    """raw → corrected → final 的编排器。"""

    def __init__(
        self,
        corrector: BaseCorrector | None = None,
        glossary: dict[str, str] | None = None,
        max_edit_ratio: float = 0.10,
        enable_itn: bool = True,
    ) -> None:
        self.corrector = corrector or PassthroughCorrector()
        self.glossary = glossary if glossary is not None else {}
        if not isinstance(self.glossary, dict) or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in self.glossary.items()
        ):
            raise ValueError("实体词典必须是非空字符串到非空字符串的映射")
        if not 0 <= max_edit_ratio <= 1:
            raise ValueError("max_edit_ratio 必须在 0 到 1 之间")
        self.max_edit_ratio = max_edit_ratio
        self.enable_itn = enable_itn

    def preflight(self) -> None:
        self.corrector.preflight()
        if self.enable_itn:
            require_module("wetext", "请使用已安装 wetext 的解释器，或显式关闭 ITN。")
            preflight_itn()

    def run(
        self,
        raw_text: str,
        output_dir: str | Path | None = None,
        stem: str = "postprocess",
        write_files: bool = True,
        input_path: str | Path | None = None,
        input_stage: str = "raw",
    ) -> PostprocessResult:
        """执行两阶段后处理。"""
        if input_stage != "raw":
            raise ValueError("后处理只接受 raw 输入，不能重复规范化 corrected/final")
        started = time.perf_counter()
        paths = artifact_paths(Path(output_dir), stem) if write_files and output_dir is not None else {}
        if paths:
            preflight_paths(paths, inputs=(Path(input_path),) if input_path else (),
                            raw_text=raw_text, allow_input_raw=True)
        self.preflight()
        model_path = getattr(self.corrector, "model_path", None)
        manifest = model_manifest(Path(model_path)) if model_path else None
        preflight_seconds = time.perf_counter() - started
        if paths:
            publish_text(paths["raw"], raw_text, immutable=True)

        # 阶段 1：语义纠错
        load_start = time.perf_counter()
        self.corrector.prepare()
        load_seconds = time.perf_counter() - load_start
        correction = self.corrector.correct(raw_text)

        # 阶段 2：修改校验（危险改动不放行）
        review_start = time.perf_counter()
        segments = correction.segment_results
        if not segments:
            if correction.segments > 1:
                raise ValueError("多段纠错必须返回原文范围，禁止退回全文比例校验")
            segments = [CorrectionSegment(
                0, len(raw_text), raw_text, correction.text,
                "failed" if correction.failed_segments or correction.truncated else "ok",
                "纠错器报告失败或截断" if correction.failed_segments or correction.truncated else "",
            )]
        # 失败段只保留原文；完整正文数值范围补足分段边界处的单位或符号。
        segments = [s if s.status == "ok" else replace(s, candidate=s.source) for s in segments]
        raw_numeric = numeric_spans(raw_text)
        candidate_numeric = numeric_spans("".join(s.candidate for s in segments))
        cursor = output_offset = candidate_offset = 0
        review_fallback_segments = review_flagged_segments = 0
        pieces, edits = [], []
        for segment in segments:
            if (segment.start != cursor or segment.end < cursor or
                    segment.end > len(raw_text) or raw_text[cursor:segment.end] != segment.source):
                raise ValueError("纠错段原文范围不连续或与原文不一致")
            adopted, local_edits = review_corrections(
                segment.source, segment.candidate, self.max_edit_ratio,
                protected_raw_spans=[(a - cursor, b - cursor) for a, b in raw_numeric
                                     if a <= segment.end and b >= cursor],
                protected_candidate_spans=[(a - candidate_offset, b - candidate_offset)
                                           for a, b in candidate_numeric
                                           if a <= candidate_offset + len(segment.candidate) and b >= candidate_offset],
            )
            if any(e.decision == "review" for e in local_edits):
                review_flagged_segments += 1
                review_fallback_segments += adopted == segment.source
            edits.extend(replace(
                e, start=e.start + cursor, end=e.end + cursor,
                output_start=e.output_start + output_offset, output_end=e.output_end + output_offset,
                candidate_start=e.candidate_start + candidate_offset,
                candidate_end=e.candidate_end + candidate_offset,
            ) for e in local_edits)
            pieces.append(adopted)
            cursor = segment.end
            output_offset += len(adopted)
            candidate_offset += len(segment.candidate)
        if cursor != len(raw_text):
            raise ValueError("纠错段未覆盖完整原文")
        corrected_text = "".join(pieces)
        if replay_edits(raw_text, edits) != corrected_text:
            raise ValueError("语义编辑无法重放")
        review_seconds = time.perf_counter() - review_start

        # 阶段 3：数字规范化（放在最后，避免错误边界提前固化）
        # enable_itn=False 时用于评估"仅纠错"变体
        itn_start = time.perf_counter()
        degradations = [f"纠错段 {s.start}:{s.end}: {s.reason}" for s in segments if s.status != "ok"]
        itn_edits = []
        final_text, itn_notes = corrected_text, []
        if self.enable_itn:
            try:
                # 长文本走「预筛 + 多进程并行」：只规范化含数字候选的连续句段，
                # 未命中段落原样保留（与整篇结果逐字符一致，实测 10min 12s→1.4s）
                itn = normalize_numbers_segmented(corrected_text, glossary=self.glossary)
                final_text, itn_notes = itn.text, list(itn.notes)
                itn_edits = list(itn.edits)
                if replay_edits(corrected_text, itn_edits) != final_text:
                    raise ValueError("ITN 编辑无法重放")
                if itn.degraded:
                    degradations.append("ITN 校验失败，已保留上一版文本")
            except Exception as exc:
                final_text, itn_edits = corrected_text, []
                itn_notes = [ItnNote("itn_failure", "", f"ITN 失败: {type(exc).__name__}，保留纠错文本")]
                degradations.append(itn_notes[0].reason)
        itn_seconds = time.perf_counter() - itn_start

        stats = {
            "schema_version": 2,
            "status": "degraded" if degradations else "complete",
            "degradations": degradations,
            "input_stage": input_stage,
            "input_sha256": text_hash(raw_text),
            "corrected_sha256": text_hash(corrected_text),
            "final_sha256": text_hash(final_text),
            "glossary_sha256": json_hash(self.glossary),
            "rules_sha256": source_fingerprint(),
            "runtime_versions": runtime_versions(),
            "max_edit_ratio": self.max_edit_ratio,
            "corrector_model_path": str(getattr(self.corrector, "model_path", "")) or None,
            "model_config_sha256": json_hash(getattr(self.corrector, "model_metadata", {})),
            "model_manifest": manifest,
            "generation_parameters": {k: getattr(self.corrector, k, None) for k in (
                "max_tokens", "max_input_tokens", "target_chars", "context_sentences",
                "repetition_penalty", "repetition_context_size")},
            "segments": [{k: v for k, v in asdict(s).items() if k not in ("source", "candidate")}
                         for s in segments],
            "prompt_tokens": sum(s.prompt_tokens for s in segments),
            "generation_tokens": sum(s.generation_tokens for s in segments),
            "token_count_scope": "observed_stream_responses_including_think_and_stop",
            "correction_tokens": {
                "prompt_per_segment": _percentiles([s.prompt_tokens for s in segments]),
                "generation_per_segment": _percentiles(
                    [s.generation_tokens for s in segments]),
                "prompt_tps_mean": round(_mean(
                    [s.prompt_tps for s in segments if s.prompt_tps > 0]), 1),
                "generation_tps_mean": round(_mean(
                    [s.generation_tps for s in segments if s.generation_tps > 0]), 1),
                # 由 tps 反推的分项耗时，用于判断 prompt cache 是否值得引入
                "prefill_seconds_estimate": round(sum(
                    s.prompt_tokens / s.prompt_tps for s in segments if s.prompt_tps > 0), 3),
                "generation_seconds_estimate": round(sum(
                    s.generation_tokens / s.generation_tps
                    for s in segments if s.generation_tps > 0), 3),
                "segment_count": len(segments),
            },
            "preflight_seconds": preflight_seconds,
            "corrector_load_seconds": load_seconds + correction.load_seconds,
            "corrector_generate_seconds": correction.generate_seconds,
            "review_seconds": review_seconds,
            "itn_seconds": itn_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "elapsed_scope": "postprocess_in_memory",
            "derived_publish_seconds": 0.0,
            "peak_rss_mb": peak_rss_mb(),
            "memory_scope": "current_process_lifetime",
            "mlx_peak_bytes": correction.mlx_peak_bytes,
            "mlx_memory_scope": "mlx_allocator_lifetime",
            "glossary_entries": len(self.glossary),
            "itn_edits": len(itn_edits),
            "corrector": correction.model,
            "corrector_segments": len(segments),
            "segment_count_scope": "validation_ranges_including_passthrough",
            "corrector_successful_segments": sum(s.status == "ok" for s in segments),
            "corrector_failed_segments": sum(s.status != "ok" for s in segments),
            "review_flagged_segments": review_flagged_segments,
            "review_fallback_segments": review_fallback_segments,
            "corrector_truncated": correction.truncated,
            "edits_total": len(edits),
            "edits_accepted": sum(1 for e in edits if e.decision == "accepted"),
            "edits_review": sum(1 for e in edits if e.decision == "review"),
            "itn_notes": len(itn_notes),
            "itn_enabled": self.enable_itn,
            "raw_chars": len(raw_text),
            "corrected_chars": len(corrected_text),
            "final_chars": len(final_text),
        }

        result = PostprocessResult(
            raw_text=raw_text,
            corrected_text=corrected_text,
            final_text=final_text,
            edits=edits,
            itn_notes=itn_notes,
            itn_edits=itn_edits,
            stats=stats,
        )

        if write_files and output_dir is not None:
            result.paths = self._write(result, Path(output_dir), stem, started)
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _write(result: PostprocessResult, out_dir: Path, stem: str, started: float) -> dict[str, str]:
        publish_started = time.perf_counter()
        paths = artifact_paths(out_dir, stem)
        publish_text(paths["raw"], result.raw_text, immutable=True)
        publish_text(paths["corrected"], result.corrected_text)
        publish_text(paths["final"], result.final_text)
        records = [asdict(e) for e in result.edits + result.itn_edits]
        records.extend({"op": "itn_note", "stage": "itn", "decision": "review", **asdict(n)}
                       for n in result.itn_notes)
        publish_text(paths["edits"], "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        result.stats["derived_publish_seconds"] = time.perf_counter() - publish_started
        result.stats["elapsed_seconds"] = time.perf_counter() - started
        result.stats["elapsed_scope"] = "postprocess_before_stats_publish"
        result.stats["peak_rss_mb"] = peak_rss_mb()
        # 最后发布完成标记；此前任一写入失败都不得出现成功 stats。
        publish_text(paths["stats"], json.dumps(result.stats, ensure_ascii=False, indent=2))
        return {k: str(v) for k, v in paths.items()}