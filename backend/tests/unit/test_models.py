"""测试核心事件模型（TaskEvent / EventType / TERMINAL_EVENTS）。"""

import pytest

from local_asr.core.models import TERMINAL_EVENTS, EventType, TaskEvent


class TestTaskEvent:
    def test_create(self):
        event = TaskEvent.create("task-1", EventType.TASK_STARTED, 1, key="val")
        assert event.task_id == "task-1"
        assert event.event_type == EventType.TASK_STARTED
        assert event.sequence == 1
        assert event.payload == {"key": "val"}

    def test_create_fills_timestamp(self):
        event = TaskEvent.create("task-1", "X", 1)
        assert event.timestamp  # ISO-8601 UTC 时间戳非空
        assert "T" in event.timestamp

    def test_payload_defaults_to_empty_dict(self):
        event = TaskEvent.create("task-1", "X", 1)
        assert event.payload == {}

    def test_sequence_independent(self):
        e1 = TaskEvent.create("t1", "X", 1)
        e2 = TaskEvent.create("t1", "X", 2)
        assert e2.sequence > e1.sequence

    def test_immutable(self):
        event = TaskEvent.create("t1", "X", 1)
        with pytest.raises(Exception):
            event.sequence = 2  # type: ignore[misc]


class TestEventType:
    def test_terminal_events_are_known_types(self):
        for event_type in TERMINAL_EVENTS:
            assert isinstance(event_type, str)

    def test_terminal_events_cover_all_endings(self):
        assert TERMINAL_EVENTS == frozenset({
            EventType.TASK_COMPLETED,
            EventType.TASK_CANCELLED,
            EventType.TASK_FAILED,
            EventType.TASK_DEGRADED,
        })

    def test_non_terminal_excluded(self):
        assert EventType.TASK_STARTED not in TERMINAL_EVENTS
        assert EventType.ASR_COMPLETED not in TERMINAL_EVENTS
