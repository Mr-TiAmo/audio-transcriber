"""ASR 识别阶段的纯逻辑组件（生成完整性判定、切分规划等）。

本包不导入 torch / transformers，保证 `.venv` 下可直接单测。
"""

from local_asr.asr.generation import (
    COMPLETED,
    DEGRADED,
    TRUNCATED,
    UNEXPECTED_STOP,
    GenerationResult,
    aggregate_status,
    classify_generation,
    eos_ids_from,
    segment_payload,
)

__all__ = [
    "COMPLETED",
    "DEGRADED",
    "TRUNCATED",
    "UNEXPECTED_STOP",
    "GenerationResult",
    "aggregate_status",
    "classify_generation",
    "eos_ids_from",
    "segment_payload",
]