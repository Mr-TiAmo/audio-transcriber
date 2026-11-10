"""事件系统 — 进程内事件发布/订阅。

编排链路（orchestration）用 :class:`EventBus` 把 :class:`TaskEvent`
从 runner 推送给 CLI 渲染器；每个任务创建独立的总线实例，不使用全局单例，
避免多任务/测试之间互相串扰。

线程安全说明：监听器列表的增删和快照复制在锁内完成，回调本身在锁外调用，
因此监听器内部可以再安全地发布事件（不会死锁）。
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from local_asr.core.models import TaskEvent

# 事件回调类型：接收一条 TaskEvent，无返回值
EventCallback = Callable[[TaskEvent], None]


class EventBus:
    """简单的进程内事件总线，线程安全。

    典型用法::

        bus = EventBus()
        bus.subscribe(lambda ev: print(ev.event_type))
        bus.publish(TaskEvent.create(task_id, EventType.TASK_STARTED, 1))
    """

    def __init__(self) -> None:
        self._listeners: list[EventCallback] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: EventCallback) -> None:
        """注册事件回调；重复注册同一回调会被调用多次。"""
        with self._lock:
            self._listeners.append(callback)

    def unsubscribe(self, callback: EventCallback) -> None:
        """取消注册；回调不存在时静默忽略。"""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def publish(self, event: TaskEvent) -> None:
        """发布事件给所有监听器。

        单个监听器抛异常不影响其余监听器（渲染失败不应中断编排主流程），
        异常被静默吞掉——这是有意的取舍：事件总线只用于观察，不用于控制。
        """
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                pass

    def clear(self) -> None:
        """清除所有监听器（任务结束时释放引用）。"""
        with self._lock:
            self._listeners.clear()
