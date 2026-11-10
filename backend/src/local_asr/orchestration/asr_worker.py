"""ASR 常驻 worker 的父进程执行器。

与 :func:`~local_asr.orchestration.process_manager.run_step` 的差别：
worker 进程不随任务退出，模型权重（ASR / ForcedAligner / VAD）跨任务
复用，因此第二次任务起省掉 5.5–10.5s 模型加载、2–9s 对齐模型加载和
1–3s VAD 加载。对外签名与 ``run_step`` 一致，可直接注入
:func:`local_asr.orchestration.runner.run` 的 ``asr_executor``。

协议（见 ``scripts/qwen3_asr_worker.py``）：

- 父 → 子：stdin 一行 JSON ``{"kind": "transcribe", "argv": [...]}``
- 子 → 父：stdout 照旧输出 ``@@EVENT@@`` 事件行；任务结束追加
  ``{"kind": "worker_task_done", "returncode": N}``

失败语义与 ``run_step`` 对齐：取消/超时终止 worker 并抛
``ProcessCancelled`` / ``ProcessError``；worker 崩溃（stdout EOF 而没有
``worker_task_done``）返回非零 ``StepResult``，由编排层判断失败，
下一次任务自动重启 worker（首个任务重新加载模型）。
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

from local_asr.orchestration.process_manager import (
    ProcessCancelled, ProcessError, StepResult,
)
from local_asr.protocol import parse_event_line

__all__ = ["AsrWorkerExecutor"]

# 保留的日志尾部行数，与 run_step 一致
_TAIL_LINES = 80
# 首次启动要 import torch / transformers / qwen_asr，慢盘上可能十几秒
_READY_TIMEOUT = 180.0
_SHUTDOWN_TIMEOUT = 5.0


class _WorkerProcess:
    """一个常驻 worker 进程及其 stdout/stderr 泵。"""

    def __init__(self, python: Path | str, script: Path, cwd: Path | None,
                 env: dict[str, str] | None) -> None:
        command = [str(python), str(script)]
        try:
            self.proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
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
            raise ProcessError(f"ASR worker 无法启动: {exc}") from exc
        # None 是 EOF 哨兵：泵线程退出时投递，读取方据此结束等待
        self.lines: "queue.Queue[str | None]" = queue.Queue()
        self.stdout_tail: deque[str] = deque(maxlen=_TAIL_LINES)
        self.stderr_tail: deque[str] = deque(maxlen=_TAIL_LINES)
        self._threads = [
            threading.Thread(target=self._pump_stdout, daemon=True),
            threading.Thread(target=self._pump_stderr, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def _pump_stdout(self) -> None:
        try:
            assert self.proc.stdout is not None
            for raw in self.proc.stdout:
                self.lines.put(raw.rstrip("\n"))
        finally:
            self.lines.put(None)

    def _pump_stderr(self) -> None:
        try:
            assert self.proc.stderr is not None
            for raw in self.proc.stderr:
                self.stderr_tail.append(raw.rstrip("\n"))
        finally:
            pass

    def _note(self, line: str) -> None:
        self.stdout_tail.append(line)

    def wait_ready(self, timeout: float = _READY_TIMEOUT) -> None:
        """等待 worker 完成 import（``worker_ready`` 事件）。"""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.terminate()
                raise ProcessError(
                    f"ASR worker 启动超过 {timeout:.0f}s 未就绪: "
                    f"{self._stderr_text()[-500:]}")
            try:
                line = self.lines.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            if line is None:
                raise ProcessError(
                    f"ASR worker 启动即退出（code={self.proc.poll()}）: "
                    f"{self._stderr_text()[-800:]}")
            event = parse_event_line(line)
            if event is not None and event.get("kind") == "worker_ready":
                return
            self._note(line)

    def _stderr_text(self) -> str:
        return "\n".join(self.stderr_tail)

    def transcribe(
        self,
        argv: Sequence[str],
        *,
        name: str,
        on_event: Callable[[dict], None] | None,
        timeout: float | None,
        cancel: "threading.Event | None",
    ) -> StepResult:
        started = time.perf_counter()
        payload = json.dumps(
            {"kind": "transcribe", "argv": [str(a) for a in argv]}, ensure_ascii=False)
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(payload + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise ProcessError(f"ASR worker 已不可写: {exc}") from exc

        events: list[dict] = []
        stdout_tail: deque[str] = deque(maxlen=_TAIL_LINES)
        timed_out = threading.Event()
        cancelled = threading.Event()
        done = threading.Event()

        timer: threading.Timer | None = None
        if timeout is not None:
            def _expire() -> None:
                timed_out.set()
                self.terminate()

            timer = threading.Timer(timeout, _expire)
            timer.daemon = True
            timer.start()

        watcher: threading.Thread | None = None
        if cancel is not None:
            def _watch() -> None:
                while not done.is_set():
                    if cancel.wait(timeout=0.5):
                        if done.is_set():
                            return
                        cancelled.set()
                        self.terminate()
                        return

            watcher = threading.Thread(target=_watch, daemon=True)
            watcher.start()

        returncode: int | None = None
        try:
            while True:
                try:
                    line = self.lines.get(timeout=0.2)
                except queue.Empty:
                    # 进程已退出但 EOF 哨兵尚未投递：继续等（哨兵一定会来）
                    continue
                if line is None:
                    break
                event = parse_event_line(line)
                if event is None:
                    stdout_tail.append(line)
                    continue
                if event.get("kind") == "worker_task_done":
                    returncode = int(event.get("returncode", -1))
                    break
                events.append(event)
                if on_event is not None:
                    try:
                        on_event(event)
                    except Exception:
                        # 监听器异常不应中断任务，也不应吞掉已产生的事件
                        pass
        finally:
            if timer is not None:
                timer.cancel()
            done.set()
            if watcher is not None:
                watcher.join(timeout=2)

        elapsed = round(time.perf_counter() - started, 3)
        if cancelled.is_set() or (cancel is not None and cancel.is_set()):
            raise ProcessCancelled(f"{name} 已被用户取消")
        if timed_out.is_set():
            raise ProcessError(f"{name} 超过 {timeout:.0f}s 未完成，已终止")
        if returncode is None:
            # stdout 已 EOF 却没有完成事件：worker 崩溃/被杀。
            # wait 让退出码落地（EOF 后进程通常已结束，这里几乎不阻塞）。
            try:
                exit_code: int | None = self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                exit_code = self.proc.poll()
            returncode = exit_code if exit_code not in (None, 0) else -1
        return StepResult(
            name=name,
            returncode=returncode,
            events=tuple(events),
            stdout_tail="\n".join(stdout_tail),
            stderr_tail=self._stderr_text(),
            elapsed_seconds=elapsed,
        )

    def terminate(self) -> None:
        try:
            self.proc.terminate()
        except OSError:
            pass

    def shutdown(self, timeout: float = _SHUTDOWN_TIMEOUT) -> None:
        """请求 worker 正常退出；超时则强制终止。"""
        try:
            if self.proc.stdin is not None and self.proc.poll() is None:
                self.proc.stdin.write('{"kind": "shutdown"}\n')
                self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass


class AsrWorkerExecutor:
    """把 ``run_step`` 调用转成「常驻 worker 的一次任务」。

    Args:
        worker_script: worker 脚本路径；默认取仓库 ``scripts/qwen3_asr_worker.py``。
    """

    def __init__(self, worker_script: Path | str | None = None) -> None:
        self._worker_script = (
            Path(worker_script) if worker_script is not None else
            Path(__file__).resolve().parents[3] / "scripts" / "qwen3_asr_worker.py")
        self._lock = threading.Lock()
        self._worker: _WorkerProcess | None = None
        self._signature: tuple | None = None
        self._closed = False

    def __call__(
        self,
        python: Path,
        argv: Sequence[str],
        *,
        name: str = "asr",
        on_event: Callable[[dict], None] | None = None,
        timeout: float | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        cancel: "threading.Event | None" = None,
    ) -> StepResult:
        """签名与 ``process_manager.run_step`` 保持一致；``argv[0]`` 是脚本路径，
        由 worker 内部固定导入，不随请求下发。"""
        with self._lock:
            if self._closed:
                raise ProcessError("ASR worker 已关闭")
            if cancel is not None and cancel.is_set():
                raise ProcessCancelled(f"{name} 已被用户取消")
            worker = self._ensure(python, cwd, env)
            args = [str(a) for a in argv]
            # run_step 约定 argv[0] 是脚本路径；worker 内部固定 import，
            # 不下发。调用方直接给参数列表（不以 .py 开头）时原样发送。
            if args and args[0].lower().endswith(".py"):
                args = args[1:]
            try:
                result = worker.transcribe(
                    args, name=name, on_event=on_event,
                    timeout=timeout, cancel=cancel)
            except (ProcessCancelled, ProcessError):
                self._discard()
                raise
            if not worker.alive:
                # 任务内崩溃：丢弃实例，下次任务重新加载模型
                self._discard()
            return result

    def _ensure(self, python: Path, cwd: Path | None,
                env: dict[str, str] | None) -> _WorkerProcess:
        signature = (str(python), str(cwd) if cwd is not None else None,
                     tuple(sorted((env or {}).items())))
        worker = self._worker
        if worker is not None and worker.alive and signature == self._signature:
            return worker
        self._discard()
        worker = _WorkerProcess(python, self._worker_script, cwd, env)
        worker.wait_ready()
        self._worker, self._signature = worker, signature
        return worker

    def _discard(self) -> None:
        worker, self._worker = self._worker, None
        self._signature = None
        if worker is not None:
            worker.shutdown()

    def close(self) -> None:
        """应用退出时停止常驻 worker（不等待正在执行的任务）。"""
        self._closed = True
        self._discard()