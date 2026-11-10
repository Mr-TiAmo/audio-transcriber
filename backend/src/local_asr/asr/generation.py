"""ASR 生成结果的完整性判定。

只依赖标准库：本模块在 `.venv-funasr` 的识别脚本和 `.venv` 的单元测试中
都会被导入，因此不得引入 torch / transformers。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

__all__ = [
    "COMPLETED",
    "DEGRADED",
    "EMPTY_TRANSCRIPT_MIN_SECONDS",
    "TRUNCATED",
    "UNEXPECTED_STOP",
    "GenerationResult",
    "aggregate_status",
    "classify_generation",
    "eos_ids_from",
    "flag_empty_transcript",
    "segment_payload",
]

COMPLETED = "completed"
TRUNCATED = "truncated"
UNEXPECTED_STOP = "unexpected_stop"
DEGRADED = "degraded"

# 长于此秒数的音频若转写为空，按异常处理：正常语音不可能整段没有字。
# 2026-11-05 实测：32 分钟音频整段识别只生成 1 个 EOS token、文本为空，
# 旧逻辑因 hit_eos 判为 completed，会发布空转写。
EMPTY_TRANSCRIPT_MIN_SECONDS = 30.0


@dataclass(frozen=True)
class GenerationResult:
    """一次 ``generate()`` 的输出与结束状态。

    ``token_features`` 为逐 token 的熵/对数概率等（tuple of dict），
    每个 dict 键：entropy / neg_logprob / neg_margin。
    未启用 output_scores 时为 None（默认）。
    """

    text: str
    generated_tokens: int
    hit_eos: bool
    hit_limit: bool
    degraded: bool = False
    token_features: tuple | None = None

    @property
    def status(self) -> str:
        # 批内重跑仍失败的块单独标记，不影响其余块的产物。
        if self.degraded:
            return DEGRADED
        # 有 EOS 即视为正常结束：即使长度恰好等于上限，也说明模型主动收尾。
        if self.hit_eos:
            return COMPLETED
        return TRUNCATED if self.hit_limit else UNEXPECTED_STOP


def eos_ids_from(generation_config: object) -> set[int]:
    """从 generation_config 读取 EOS 集合；兼容 None / int / list。"""
    value = getattr(generation_config, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(v) for v in value}


def classify_generation(
    *,
    last_token_id: int,
    generated_tokens: int,
    max_new_tokens: int,
    eos_ids: Iterable[int],
) -> tuple[bool, bool]:
    """返回 ``(hit_eos, hit_limit)``。

    ``generated_tokens`` 为 0 时没有任何 token 可判断结束符，按异常终止处理。
    """
    if generated_tokens <= 0:
        return False, False
    hit_eos = last_token_id in set(eos_ids)
    hit_limit = generated_tokens >= max_new_tokens
    return hit_eos, hit_limit


def flag_empty_transcript(result: GenerationResult, audio_seconds: float) -> GenerationResult:
    """长音频转写为空时标记 degraded，避免把空结果当成功发布。

    短音频（< ``EMPTY_TRANSCRIPT_MIN_SECONDS``）允许空文本（静音/极短片段），
    长音频出现空文本说明模型或内存异常，必须让调用方看见。
    """
    if result.degraded or result.text.strip():
        return result
    if audio_seconds < EMPTY_TRANSCRIPT_MIN_SECONDS:
        return result
    return replace(result, degraded=True)


def aggregate_status(results: Sequence[GenerationResult]) -> str:
    """多段结果的整体状态：只要有一段不完整，整体就不是 completed。"""
    statuses = [r.status for r in results]
    if not statuses:
        return UNEXPECTED_STOP
    if DEGRADED in statuses:
        return DEGRADED
    if TRUNCATED in statuses:
        return TRUNCATED
    if UNEXPECTED_STOP in statuses:
        return UNEXPECTED_STOP
    return COMPLETED


def segment_payload(
    result: GenerationResult,
    *,
    index: int,
    start_s: float,
    end_s: float,
    cut_silence_ms: int | None = None,
    include_text: bool = False,
    low_conf_spans: list | None = None,
) -> dict:
    """单个分段的 stats 记录。

    ``include_text`` 为 True 时附带该分段文本（供 API 层按 chunk 重建
    片段；V3.0 §5.6：无 Aligner 方案下片段边界即 chunk 边界）。
    ``low_conf_spans`` 为低置信度字符区间列表 [[start, end], ...]，
    用于前端标亮渲染。
    """
    payload = {
        "index": index,
        "start_s": round(start_s, 2),
        "end_s": round(end_s, 2),
        "cut_silence_ms": cut_silence_ms,
        "generated_tokens": result.generated_tokens,
        "hit_eos": result.hit_eos,
        "hit_limit": result.hit_limit,
        "status": result.status,
        "degraded": result.degraded,
        "chars": len(result.text),
    }
    if include_text:
        payload["text"] = result.text
    if low_conf_spans:
        payload["low_conf_spans"] = low_conf_spans
    return payload