"""父子进程结构化事件协议（编排层 ↔ 子进程脚本的唯一契约）。

子进程（ASR 识别 / 文本后处理）在关键节点向 stdout 写一行::

    @@EVENT@@ {"kind": "asr_completed", "elapsed_seconds": 65.5}

编排层 :mod:`local_asr.orchestration.process_manager` 逐行读取 stdout，
把带前缀的行解析成事件 dict 交给回调，其余行按普通日志保留尾部。

选择 stdout 前缀而不是 JSONL 文件轮询：无写入竞态、无额外 IO、
事件到达即解析（进度实时）。前缀 ``@@EVENT@@`` 足够罕见，
不会与人类可读日志冲突。

本模块只依赖标准库 json，两端（.venv 的编排进程、.venv-funasr /
.venv-mlx 的子进程）都可以直接导入；协议定义单点维护，
避免发射端与解析端各自硬编码前缀导致漂移。
"""

from __future__ import annotations

import json

__all__ = ["EVENT_PREFIX", "emit_event", "parse_event_line"]

EVENT_PREFIX = "@@EVENT@@"


def emit_event(kind: str, **payload: object) -> None:
    """向 stdout 发布一条结构化事件（子进程侧）。

    ``kind`` 是事件名（如 ``asr_started``），其余关键字参数原样进入
    JSON payload。立即 flush：父进程逐行读取，缓冲会让进度事件延迟
    到进程结束才可见。
    """
    print(f"{EVENT_PREFIX} {json.dumps({'kind': kind, **payload}, ensure_ascii=False)}",
          flush=True)


def parse_event_line(line: str) -> dict | None:
    """解析一行 stdout；不是事件行或 JSON 损坏时返回 None（父进程侧）。

    宽容策略是刻意的：事件行损坏只损失一条进度通知，
    不应该让整个编排任务失败；非事件行本来就应返回 None 走日志通道。
    """
    if not line.startswith(EVENT_PREFIX):
        return None
    payload = line[len(EVENT_PREFIX):].strip()
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
