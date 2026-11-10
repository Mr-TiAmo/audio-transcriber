"""常驻 ASR worker 执行器测试。

用临时假 worker 脚本验证协议与生命周期，不加载 torch：
- 同一 worker 跨任务复用（pid 不变）；
- 事件行转发、普通日志进 tail；
- 非零退出码透传且 worker 存活；
- 崩溃 / 取消 / 超时后实例被丢弃，下次任务自动重启。
"""

from __future__ import annotations

import sys
import textwrap
import threading
from pathlib import Path

import pytest

from local_asr.orchestration.asr_worker import AsrWorkerExecutor
from local_asr.orchestration.process_manager import ProcessCancelled, ProcessError

_FAKE_WORKER = """
import json
import os
import sys
import time


def emit(kind, **payload):
    print("@@EVENT@@ " + json.dumps({"kind": kind, **payload}), flush=True)


emit("worker_ready", pid=os.getpid())
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    if request["kind"] == "shutdown":
        break
    argv = request["argv"]
    if "--crash" in argv:
        sys.exit(7)
    if "--hang" in argv:
        time.sleep(60)
    print("普通日志一行", flush=True)
    emit("task_event", pid=os.getpid(), argv=argv)
    emit("worker_task_done", returncode=5 if "--fail" in argv else 0)
"""


@pytest.fixture()
def executor(tmp_path: Path):
    script = tmp_path / "fake_worker.py"
    script.write_text(textwrap.dedent(_FAKE_WORKER), encoding="utf-8")
    ex = AsrWorkerExecutor(script)
    yield ex
    ex.close()


def _argv(*extra: str) -> list[str]:
    return ["qwen3_asr_transcribe.py", "--audio", "a.wav", *extra]


def test_worker_reuse_across_tasks(executor):
    events: list[dict] = []
    first = executor(Path(sys.executable), _argv(), on_event=events.append)
    second = executor(Path(sys.executable), _argv(), on_event=events.append)

    assert first.ok and second.ok
    assert first.name == "asr" and first.returncode == 0
    assert "普通日志一行" in first.stdout_tail
    # 两个任务由同一个常驻进程处理
    assert events[0]["pid"] == events[-1]["pid"]
    assert all(e["kind"] == "task_event" for e in events)


def test_nonzero_exit_is_returned_and_worker_survives(executor):
    first = executor(Path(sys.executable), _argv("--fail"))
    assert first.returncode == 5
    # 失败不会杀掉 worker，下一次任务复用同一进程
    second = executor(Path(sys.executable), _argv())
    assert second.ok


def test_crash_discards_worker_and_next_task_restarts(executor):
    crashed = executor(Path(sys.executable), _argv("--crash"))
    assert not crashed.ok
    events: list[dict] = []
    again = executor(Path(sys.executable), _argv(), on_event=events.append)
    assert again.ok
    assert events and events[-1]["kind"] == "task_event"


def test_cancel_raises_and_terminates(executor):
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    with pytest.raises(ProcessCancelled):
        executor(Path(sys.executable), _argv("--hang"), cancel=cancel)


def test_timeout_raises(executor):
    with pytest.raises(ProcessError, match="未完成"):
        executor(Path(sys.executable), _argv("--hang"), timeout=0.5)


def test_close_rejects_further_tasks(executor):
    executor.close()
    with pytest.raises(ProcessError, match="已关闭"):
        executor(Path(sys.executable), _argv())