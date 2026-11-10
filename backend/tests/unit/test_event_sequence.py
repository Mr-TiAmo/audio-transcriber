"""测试事件序列一致性。

验证 sequence 递增、事件按时间顺序提交等约束。
"""

from local_asr.core.models import EventType, TaskEvent


class TestEventSequence:
    """事件序列的顺序约束测试。"""

    def test_sequence_monotonically_increases(self):
        """sequence 在单任务内必须严格递增。"""
        events: list[TaskEvent] = []
        for i in range(10):
            events.append(TaskEvent.create("task-1", EventType.PROGRESS_UPDATED, i + 1))

        for i in range(1, len(events)):
            assert events[i].sequence > events[i - 1].sequence, (
                f"sequence 应严格递增: {events[i].sequence} <= {events[i-1].sequence}"
            )

    def test_different_tasks_independent_sequences(self):
        """不同任务的 sequence 互不影响。"""
        e1 = TaskEvent.create("task-1", EventType.TASK_STARTED, 1)
        e2 = TaskEvent.create("task-2", EventType.TASK_STARTED, 1)
        assert e1.sequence == 1
        assert e2.sequence == 1
        assert e1.task_id != e2.task_id

    def test_event_types_are_valid(self):
        """所有事件类型常量应存在。"""
        expected = {
            # 基础任务事件
            "TASK_STARTED", "PHASE_CHANGED", "PROGRESS_UPDATED", "SEGMENT_COMMITTED",
            "TASK_COMPLETED", "TASK_CANCELLED", "TASK_FAILED",
            # 统一编排阶段事件（ELECTRA gate 已删除，见重构决策）
            "ASR_STARTED", "ASR_COMPLETED",
            "CORRECTION_STARTED", "CORRECTION_COMPLETED", "VERIFY_COMPLETED",
            "ITN_STARTED", "ITN_COMPLETED",
            "TASK_DEGRADED",
        }
        actual = {v for k, v in vars(EventType).items() if not k.startswith("_")}
        assert actual == expected

    def test_terminal_events_cover_all_end_states(self):
        """终止态集合必须包含所有"任务已结束"的事件。"""
        from local_asr.core.models import TERMINAL_EVENTS
        assert TERMINAL_EVENTS == {
            EventType.TASK_COMPLETED, EventType.TASK_CANCELLED,
            EventType.TASK_FAILED, EventType.TASK_DEGRADED,
        }


class TestProgressMonotonicity:
    """进度单调性测试。"""

    def _make_progress_event(self, task_id: str, seq: int, percent: float, processed: float, total: float) -> TaskEvent:
        return TaskEvent.create(
            task_id, EventType.PROGRESS_UPDATED, seq,
            percent=percent,
            processed_seconds=processed,
            total_seconds=total,
        )

    def test_progress_never_decreases(self):
        """进度百分比不应倒退。"""
        events = [
            self._make_progress_event("t1", 2, 10.0, 10.0, 100.0),
            self._make_progress_event("t1", 3, 25.0, 25.0, 100.0),
            self._make_progress_event("t1", 4, 50.0, 50.0, 100.0),
            self._make_progress_event("t1", 5, 99.0, 99.0, 100.0),
            self._make_progress_event("t1", 6, 100.0, 100.0, 100.0),
        ]
        for i in range(1, len(events)):
            prev_pct = events[i - 1].payload["percent"]
            curr_pct = events[i].payload["percent"]
            assert curr_pct >= prev_pct, f"进度倒退: {curr_pct} < {prev_pct}"

    def test_processed_seconds_never_decreases(self):
        """已处理时长不应倒退。"""
        events = [
            self._make_progress_event("t1", 2, 10.0, 10.0, 100.0),
            self._make_progress_event("t1", 3, 20.0, 20.0, 100.0),
        ]
        for i in range(1, len(events)):
            prev = events[i - 1].payload["processed_seconds"]
            curr = events[i].payload["processed_seconds"]
            assert curr >= prev, f"已处理时长倒退: {curr} < {prev}"

    def test_percent_not_exceed_100(self):
        """进度不应超过 100%。"""
        e = self._make_progress_event("t1", 1, 100.0, 100.0, 100.0)
        assert e.payload["percent"] <= 100.0

    def test_progress_range(self):
        """进度范围应为 0-100。"""
        for pct in [0.0, 50.0, 99.0, 100.0]:
            e = self._make_progress_event("t1", 1, pct, pct, 100.0)
            assert 0.0 <= e.payload["percent"] <= 100.0