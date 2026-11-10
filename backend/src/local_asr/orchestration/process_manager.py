"""子进程执行与结构化事件分流。

子进程（ASR / 后处理）在关键节点向 stdout 写一行::

    @@EVENT@@ {"kind": "asr_completed", "elapsed_seconds": 65.5}

父进程在这里把带前缀的行解析成事件交给回调，其余行按日志保留尾部。
选择 stdout 前缀而不是 JSONL 文件，是为了避免额外的文件轮询与写入竞态。

约定：
- 非零退出码**不**在此抛异常，调用方按业务语义判断（后处理用 3 表示降级）。
- 超时、无法启动属于编排层故障，直接抛 ``ProcessError``。
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from local_asr.protocol import EVENT_PREFIX, parse_event_line

# EVENT_PREFIX 从协议模块再导出：调用方（测试/CLI）习惯从这里引用
__all__ = ["EVENT_PREFIX", "ProcessCancelled", "ProcessError", "StepResult", "run_step"]

# 保留的日志尾部行数，用于失败时给出上下文
_TAIL_LINES = 80


class ProcessError(RuntimeError):
    """子进程无法启动或超时。"""


class ProcessCancelled(RuntimeError):
    """子进程被用户取消（协作式取消：监视线程终止进程后抛出）。"""


@dataclass(frozen=True)
class StepResult:
    """一次子进程调用的结果。"""

    name: str
    returncode: int
    events: tuple[dict, ...] = ()
    stdout_tail: str = ""
    stderr_tail: str = ""
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def last_event(self, kind: str) -> dict | None:
        for event in reversed(self.events):
            if event.get("kind") == kind:
                return event
        return None


def _drain(stream, sink: deque) -> None:
    try:
        for line in stream:
            sink.append(line.rstrip("\n"))
    finally:
        stream.close()


def run_step(
    python: Path,
    argv: Sequence[str],
    *,
    name: str,
    on_event: Callable[[dict], None] | None = None,
    timeout: float | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    cancel: "threading.Event | None" = None,
) -> StepResult:
    """执行一个子进程并收集其结构化事件。

    Args:
        python: 目标解释器。
        argv: 传给该解释器的脚本与参数。
        name: 步骤名，用于日志与错误信息。
        on_event: 每解析到一个事件调用一次。
        timeout: 秒；超时后终止进程并抛 ``ProcessError``。
        cwd: 工作目录，默认沿用当前进程。
        env: 额外环境变量，会合并到当前环境。
        cancel: 协作式取消事件（§8.3）：触发后终止子进程并抛
            ``ProcessCancelled``。推理期间子进程可能长时间无输出，因此
            用监视线程轮询该事件，而不是只在读到输出行时检查。
    """
    command = [str(python), *[str(a) for a in argv]]
    stdout_tail: deque = deque(maxlen=_TAIL_LINES)
    stderr_tail: deque = deque(maxlen=_TAIL_LINES)
    events: list[dict] = []
    started = time.perf_counter()

    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
        )
    except OSError as exc:
        raise ProcessError(f"{name} 无法启动: {exc}") from exc

    stderr_reader = threading.Thread(target=_drain, args=(proc.stderr, stderr_tail), daemon=True)
    stderr_reader.start()

    # 超时必须覆盖整个流式读取阶段：子进程长时间不产出时，for 循环本身会阻塞，
    # 所以用定时器 kill，而不是只在 wait() 上设超时。
    timed_out = threading.Event()
    timer: threading.Timer | None = None
    if timeout is not None:
        def _expire() -> None:
            timed_out.set()
            proc.kill()

        timer = threading.Timer(timeout, _expire)
        timer.daemon = True
        timer.start()

    # 协作式取消：监视线程轮询 cancel 事件，触发即 kill；
    # done 用于步骤正常结束后让监视线程退出（daemon 线程不悬挂）
    cancelled = threading.Event()
    done = threading.Event()
    watcher: threading.Thread | None = None
    if cancel is not None:
        def _watch() -> None:
            while not done.is_set():
                if cancel.wait(timeout=0.5):
                    if done.is_set():
                        return
                    cancelled.set()
                    proc.kill()
                    return

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()

    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            event = parse_event_line(line)
            if event is None:
                stdout_tail.append(line)
                continue
            events.append(event)
            if on_event is not None:
                try:
                    on_event(event)
                except Exception:
                    # 监听器异常不应中断子进程，也不应吞掉已产生的事件
                    pass
        proc.wait()
    finally:
        if timer is not None:
            timer.cancel()
        done.set()
        if watcher is not None:
            watcher.join(timeout=2)
        if proc.stdout is not None:
            proc.stdout.close()
        stderr_reader.join(timeout=5)

    if cancelled.is_set() or (cancel is not None and cancel.is_set()):
        raise ProcessCancelled(f"{name} 已被用户取消")
    if timed_out.is_set():
        raise ProcessError(f"{name} 超过 {timeout:.0f}s 未完成，已终止")

    return StepResult(
        name=name,
        returncode=proc.returncode,
        events=tuple(events),
        stdout_tail="\n".join(stdout_tail),
        stderr_tail="\n".join(stderr_tail),
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )