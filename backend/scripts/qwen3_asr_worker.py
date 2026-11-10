#!/usr/bin/env python
"""常驻 ASR worker：同一进程连续处理多个任务，复用已加载的权重。

启动后从 stdin 逐行读取 JSON 请求::

    {"kind": "transcribe", "argv": ["--model", "...", "--audio", "..."]}

执行 ``qwen3_asr_transcribe.main(argv)``；stdout 照旧输出 ``@@EVENT@@``
行（父进程原样转发），任务结束追加一条::

    @@EVENT@@ {"kind": "worker_task_done", "returncode": 0}

``{"kind": "shutdown"}`` 让 worker 正常退出。

为什么独立成脚本而不是把模型加载进 API 进程：
- 保持「API 不加载 torch、推理在子进程」的架构边界；
- worker 崩溃只影响当前任务，父进程下次重启即可，不会拖垮服务；
- 模型缓存（``_MODEL_CACHE`` 等）只在此进程内生效。

父进程侧见 :class:`local_asr.orchestration.asr_worker.AsrWorkerExecutor`。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# src 放最前、scripts 追加到最后：scripts/ 下有一个与包同名的
# ``local_asr.py`` 命令行包装，若把 scripts 放在 sys.path[0]，
# ``import local_asr`` 会命中该模块并报 "not a package"。
_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPTS_DIR.parent / "src"
# 必须确保 src 位于 sys.path 最前：PYTHONPATH 里可能已有相对路径形式的
# ``src``（排在脚本目录之后），不置顶就会先命中 scripts/local_asr.py。
_src = str(_SRC_DIR)
while _src in sys.path:
    sys.path.remove(_src)
sys.path.insert(0, _src)
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(_SCRIPTS_DIR))

from local_asr.protocol import emit_event  # noqa: E402


def _release_temp_vram() -> None:
    """归还任务临时张量占用的显存缓存块（不影响已常驻的模型权重）。

    常驻 worker 连续跑多个任务时，torch 缓存分配器会保留峰值缓存；
    每次任务后清一次可避免长时间运行后碎片累积。
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - 释放失败不影响任务结果
        pass


def _handle_transcribe(asr_mod, request: dict) -> int:
    argv = request.get("argv")
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        print("[qwen3-asr-worker] 请求缺少合法的 argv", file=sys.stderr, flush=True)
        return 2
    try:
        return int(asr_mod.main(argv))
    except SystemExit as exc:  # argparse 参数错误
        return int(exc.code) if isinstance(exc.code, int) else 2
    except Exception:  # noqa: BLE001 - 单任务失败不应杀死常驻进程
        traceback.print_exc()
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="常驻 ASR worker")
    parser.add_argument("--name", default="asr", help="日志前缀（父进程用于区分实例）")
    parser.parse_args(argv)

    # 协议与文本路径都用 UTF-8：Windows 控制台默认编码会让中文日志乱码/报错
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    import qwen3_asr_transcribe as asr_mod

    emit_event("worker_ready", pid=__import__("os").getpid())
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            print(f"[qwen3-asr-worker] 无法解析请求: {line[:200]}",
                  file=sys.stderr, flush=True)
            emit_event("worker_task_done", returncode=2)
            continue
        if not isinstance(request, dict):
            emit_event("worker_task_done", returncode=2)
            continue
        kind = request.get("kind")
        if kind == "shutdown":
            break
        if kind != "transcribe":
            print(f"[qwen3-asr-worker] 未知请求 kind={kind!r}", file=sys.stderr, flush=True)
            emit_event("worker_task_done", returncode=2)
            continue
        code = _handle_transcribe(asr_mod, request)
        _release_temp_vram()
        emit_event("worker_task_done", returncode=code)
    return 0


if __name__ == "__main__":
    sys.exit(main())