"""Qwen3-ASR 文本后处理：两阶段语义纠错与数字规范化。

对外入口：
- ``normalize_numbers``：wetext ITN + 定向修正规则（A 基线）
- ``MLXCorrector`` / ``PassthroughCorrector``：语义纠错适配器
- ``review_corrections``：修改校验
- ``PostprocessPipeline``：编排 raw → corrected → final + edits.jsonl
"""

from local_asr.postprocess.corrector import (
    CorrectionResult,
    MLXCorrector,
    PassthroughCorrector,
    build_corrector,
)
from local_asr.postprocess.itn import ItnNote, ItnResult, normalize_numbers
from local_asr.postprocess.pipeline import PostprocessPipeline, PostprocessResult
from local_asr.postprocess.verify import Edit, review_corrections

__all__ = [
    "CorrectionResult",
    "Edit",
    "ItnNote",
    "ItnResult",
    "MLXCorrector",
    "PassthroughCorrector",
    "PostprocessPipeline",
    "PostprocessResult",
    "build_corrector",
    "normalize_numbers",
    "review_corrections",
]