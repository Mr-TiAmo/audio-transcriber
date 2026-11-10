"""核心数据模型 — 任务事件的统一数据结构。

本模块只包含跨解释器编排链路（orchestration）使用的事件模型：

- :class:`TaskEvent`  —— 单条任务事件，CLI/GUI 统一消费；
- :class:`EventType`  —— 事件类型常量表；
- ``TERMINAL_EVENTS`` —— 终止态事件集合，发布后任务不得再追加事件。

设计约束：
- ``TaskEvent`` 可无损序列化为 JSONL（orchestration 的进程间通信格式），
  因此字段只允许 JSON 原生类型（str / int / dict）；
- ``sequence`` 由 :class:`~local_asr.orchestration.task_state.TaskState`
  统一分配，单任务内严格递增，供消费端检测丢事件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class TaskEvent:
    """任务事件，CLI/GUI 统一消费。

    Attributes:
        task_id: 所属任务 ID，同一任务的所有事件共享。
        event_type: 事件类型，取值为 :class:`EventType` 中的常量。
        sequence: 单任务内严格递增的序号，用于消费端检测乱序/丢失。
        timestamp: UTC ISO-8601 时间戳，事件创建时自动生成。
        payload: 事件附带的结构化数据（必须是 JSON 可序列化的 dict）。
    """

    task_id: str
    event_type: str
    sequence: int
    timestamp: str
    payload: dict = field(default_factory=dict)

    @classmethod
    def create(cls, task_id: str, event_type: str, sequence: int, **payload: object) -> TaskEvent:
        """构造事件：自动打 UTC 时间戳，关键字参数收进 ``payload``。"""
        return cls(
            task_id=task_id,
            event_type=event_type,
            sequence=sequence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=dict(payload),
        )


class EventType:
    """事件类型常量表。

    分为两组：

    1. 任务生命周期事件（TASK_* / PHASE_* / PROGRESS_*）——任何任务都会出现；
    2. 编排阶段事件（ASR_* / CORRECTION_* / ITN_* 等）——对应统一编排链路
       「识别 → 纠错 → 校验 → 数字规范化」的各阶段。每个阶段由不同 venv 的
       子进程执行，事件用于 CLI/GUI 观察进度，不表示同一进程内的调用栈。
    """

    # ---- 任务生命周期 ----
    TASK_STARTED = "TASK_STARTED"
    PHASE_CHANGED = "PHASE_CHANGED"
    PROGRESS_UPDATED = "PROGRESS_UPDATED"
    SEGMENT_COMMITTED = "SEGMENT_COMMITTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_CANCELLED = "TASK_CANCELLED"
    TASK_FAILED = "TASK_FAILED"

    # ---- 统一编排（Qwen3-ASR 跨解释器链路）----
    ASR_STARTED = "ASR_STARTED"
    ASR_COMPLETED = "ASR_COMPLETED"
    CORRECTION_STARTED = "CORRECTION_STARTED"
    CORRECTION_COMPLETED = "CORRECTION_COMPLETED"
    VERIFY_COMPLETED = "VERIFY_COMPLETED"
    ITN_STARTED = "ITN_STARTED"
    ITN_COMPLETED = "ITN_COMPLETED"
    TASK_DEGRADED = "TASK_DEGRADED"


# 终止态事件：发布后不得再追加任何事件（TaskState 会拒绝后续写入）
TERMINAL_EVENTS = frozenset({
    EventType.TASK_COMPLETED,
    EventType.TASK_CANCELLED,
    EventType.TASK_FAILED,
    EventType.TASK_DEGRADED,
})
