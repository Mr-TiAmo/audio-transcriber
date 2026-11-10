"""任务状态与事件发布。

编排层是**唯一**知道完整链路的地方：ASR → 纠错 → 校验 → ITN。
CLI / GUI 只订阅这里发出的事件，不需要理解三个 venv 的内部实现。

约束：
- 事件序号在同一任务内严格递增。
- 进入终止态（COMPLETED / DEGRADED / FAILED / CANCELLED）后不得再发布事件。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from local_asr.core.events import EventBus
from local_asr.core.models import TERMINAL_EVENTS, EventType, TaskEvent

__all__ = ["TaskState", "make_task_id"]

# 阶段权重，用于把多进程链路折算成单一进度百分比。
# ASR 占大头是因为实测 331.9s 音频下识别 65.5s、纠错 115s、ITN 1.6s；
# 这里按"最坏情况"分配，实际进度以子进程事件为准。
_PHASE_ORDER = (
    EventType.ASR_STARTED,
    EventType.ASR_COMPLETED,
    EventType.CORRECTION_STARTED,
    EventType.CORRECTION_COMPLETED,
    EventType.VERIFY_COMPLETED,
    EventType.ITN_STARTED,
    EventType.ITN_COMPLETED,
)


def make_task_id() -> str:
    import uuid
    return uuid.uuid4().hex[:12]


@dataclass
class TaskState:
    """持有任务事件序号与终止状态。"""

    task_id: str
    event_bus: EventBus | None = None
    clock: Callable[[], str] | None = None
    _sequence: int = field(default=0, init=False)
    _terminal: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _events: list[TaskEvent] = field(default_factory=list, init=False)

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    @property
    def events(self) -> tuple[TaskEvent, ...]:
        return tuple(self._events)

    def emit(self, event_type: str, **payload: object) -> TaskEvent:
        """发布一个事件；进入终止态后再发布会抛错。"""
        with self._lock:
            if self._terminal:
                raise RuntimeError(
                    f"任务 {self.task_id} 已进入终止态，不能再发布 {event_type}"
                )
            self._sequence += 1
            event = TaskEvent.create(self.task_id, event_type, self._sequence, **payload)
            self._events.append(event)
            if event_type in TERMINAL_EVENTS:
                self._terminal = True
        if self.event_bus is not None:
            self.event_bus.publish(event)
        return event

    def phase(self, name: str, message: str) -> TaskEvent:
        """进度语义的阶段切换，兼容既有 CLI 渲染。"""
        return self.emit(EventType.PHASE_CHANGED, phase=name, message=message)

    def completed(self, **payload: object) -> TaskEvent:
        return self.emit(EventType.TASK_COMPLETED, **payload)

    def degraded(self, reasons: list[str], **payload: object) -> TaskEvent:
        return self.emit(EventType.TASK_DEGRADED, reasons=list(reasons), **payload)

    def failed(self, error: str, stage: str, **payload: object) -> TaskEvent:
        return self.emit(EventType.TASK_FAILED, error=error, stage=stage, **payload)


def phase_order() -> tuple[str, ...]:
    """暴露阶段顺序，便于 GUI 侧做进度条映射。"""
    return _PHASE_ORDER