"""第二模型交叉验证的字符坐标映射（ASR 脚本层）。

``build_cross_flags`` 把「合并全文坐标」的模型不一致标志映射回各 chunk
原文坐标（``CharSignal.index`` 的口径）。去重会裁掉下一段开头/上一段
结尾的重复字符，映射必须跟着平移，否则加分落到错误的字上。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_asr.asr.segmentation import merge_texts

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "qwen3_asr_transcribe.py"


def _load_script():
    name = "qwen3_asr_transcribe_for_cross_check"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclass 处理字符串注解时会查 sys.modules[cls.__module__]，
    # 动态加载的模块必须先注册，否则 AttributeError。
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def asr_script():
    return _load_script()


def _seg(text: str) -> SimpleNamespace:
    return SimpleNamespace(result=SimpleNamespace(text=text))


def test_build_cross_flags_without_dedup(asr_script):
    segs = [_seg("我们讨论游泳"), _seg("比赛马上开始")]
    merged = "我们讨论游泳比赛马上开始"
    other = "我们讨论蝶泳比赛马上开始"
    flags = asr_script.build_cross_flags(merged, segs, [], other)

    assert flags[0] == [False, False, False, False, True, False]
    assert flags[1] == [False] * 6


def test_build_cross_flags_shifts_after_dedup(asr_script):
    texts = ["我们讨论游泳运动", "游泳运动马上开始"]
    merged, dedup = merge_texts(texts)
    assert merged == "我们讨论游泳运动马上开始"
    assert dedup and dedup[0]["dropped_chars"] == 4  # 裁掉重叠的「游泳运动」

    segs = [_seg(texts[0]), _seg(texts[1])]
    other = "我们讨论游泳运动马上开始"
    flags = asr_script.build_cross_flags(merged, segs, dedup, other)

    assert not any(flags[0])
    # 第二段原文前 4 字被去重裁掉，后面全部与第二模型一致
    assert not any(flags[1])
    assert len(flags[1]) == len(texts[1])


def test_build_cross_flags_maps_mismatch_inside_trimmed_segment(asr_script):
    """不一致落在去重后保留的部分时，原文坐标要包含裁掉的头部长度。"""
    texts = ["我们讨论游泳运动", "游泳运动马上开始"]
    merged, dedup = merge_texts(texts)
    segs = [_seg(texts[0]), _seg(texts[1])]
    other = "我们讨论游泳运动马上开始"  # 一致，先确认基线
    assert not any(asr_script.build_cross_flags(merged, segs, dedup, other)[1])

    other2 = "我们讨论游泳运动立刻开始"  # 「马上」→「立刻」
    flags = asr_script.build_cross_flags(merged, segs, dedup, other2)
    # 第二段原文 = 游泳运动马上开始（「马」在原文下标 4、5）
    assert flags[1][4] and flags[1][5]
    assert sum(flags[1]) == 2