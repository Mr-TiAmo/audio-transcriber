"""local_asr.protocol 事件协议的单测。

协议两端：子进程 emit_event 写 stdout，父进程 parse_event_line 逐行解析。
这里覆盖 roundtrip 与宽容解析（损坏行只丢事件、不抛异常）。
"""

from __future__ import annotations

from local_asr.protocol import EVENT_PREFIX, emit_event, parse_event_line


def test_emit_parse_roundtrip(capsys):
    """emit 一行 → parse 还原 dict，中文不转义（ensure_ascii=False）。"""
    emit_event("asr_completed", elapsed_seconds=65.5, note="完成")
    line = capsys.readouterr().out.rstrip("\n")
    assert line.startswith(f"{EVENT_PREFIX} ")
    parsed = parse_event_line(line)
    assert parsed == {"kind": "asr_completed", "elapsed_seconds": 65.5, "note": "完成"}


def test_parse_rejects_plain_log_line():
    """普通日志行没有前缀，返回 None 走日志通道。"""
    assert parse_event_line("[asr] 正在加载模型...") is None
    assert parse_event_line("") is None


def test_parse_rejects_bare_prefix():
    """只有前缀没有 payload：损坏行，宽容返回 None。"""
    assert parse_event_line(EVENT_PREFIX) is None
    assert parse_event_line(f"{EVENT_PREFIX}   ") is None


def test_parse_rejects_broken_json():
    assert parse_event_line(f'{EVENT_PREFIX} {{"kind": "trunc') is None


def test_parse_rejects_non_dict_json():
    """JSON 合法但不是对象（数组/标量）同样拒绝，保证回调拿到 dict。"""
    assert parse_event_line(f"{EVENT_PREFIX} [1, 2]") is None
    assert parse_event_line(f"{EVENT_PREFIX} 42") is None
