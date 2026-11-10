"""分桶与批大小决策的单元测试。"""

from __future__ import annotations

import pytest

from local_asr.asr.batching import (
    BatchConfig,
    group_budget,
    plan_buckets,
    select_batch_size,
    split_groups,
)


def _cfg(**kwargs) -> BatchConfig:
    return BatchConfig(**kwargs)


# ---------------------------------------------------------------------------
# plan_buckets
# ---------------------------------------------------------------------------

def test_plan_buckets_groups_equal_lengths() -> None:
    groups = plan_buckets({0: 120.0, 1: 121.0, 2: 119.0}, _cfg())
    assert groups == [[2, 0, 1]]  # 组内按长度升序


def test_plan_buckets_splits_when_tolerance_exceeded() -> None:
    # (180-134)/180 ≈ 0.255 > 0.2，必须拆成两组
    groups = plan_buckets({0: 134.0, 1: 180.0}, _cfg())
    assert groups == [[0], [1]]


def test_plan_buckets_accepts_exact_tolerance_boundary() -> None:
    # (125-100)/125 = 0.2，允许同组
    assert plan_buckets({0: 100.0, 1: 125.0}, _cfg()) == [[0, 1]]


def test_plan_buckets_respects_max_batch() -> None:
    groups = plan_buckets({i: 100.0 + i for i in range(5)}, _cfg(max_batch=2))
    assert [len(g) for g in groups] == [2, 2, 1]


def test_plan_buckets_keeps_index_order_within_group() -> None:
    groups = plan_buckets({7: 90.0, 3: 95.0, 5: 100.0}, _cfg())
    assert groups == [[7, 3, 5]]


def test_plan_buckets_empty_input() -> None:
    assert plan_buckets({}, _cfg()) == []


def test_plan_buckets_handles_zero_length() -> None:
    assert plan_buckets({0: 0.0, 1: 0.0}, _cfg()) == [[0, 1]]


# ---------------------------------------------------------------------------
# split_groups
# ---------------------------------------------------------------------------

def test_split_groups_respects_limit() -> None:
    assert split_groups([[0, 1, 2], [3]], 2) == [[0, 1], [2], [3]]


def test_split_groups_limit_one_is_identity() -> None:
    assert split_groups([[0, 1], [2]], 1) == [[0], [1], [2]]


def test_split_groups_rejects_bad_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        split_groups([[0]], 0)


# ---------------------------------------------------------------------------
# group_budget
# ---------------------------------------------------------------------------

def test_group_budget_uses_longest_chunk() -> None:
    seconds = {0: 10.0, 1: 200.0}
    assert group_budget([0], seconds) == 512
    assert group_budget([0, 1], seconds) == group_budget([1], seconds)


# ---------------------------------------------------------------------------
# select_batch_size
# ---------------------------------------------------------------------------

def test_select_batch_size_keeps_requested_without_auto() -> None:
    assert select_batch_size(requested=1, auto=False, driver_mb=5000,
                             longest_seconds=120, config=_cfg()) == 1
    assert select_batch_size(requested=3, auto=False, driver_mb=5000,
                             longest_seconds=120, config=_cfg()) == 3


def test_select_batch_size_caps_at_max_batch() -> None:
    assert select_batch_size(requested=8, auto=False, driver_mb=5000,
                             longest_seconds=120, config=_cfg()) == 3


def test_select_batch_size_auto_uses_max_when_memory_allows() -> None:
    # 5500 + 800*2 = 7100 <= 8000
    assert select_batch_size(requested=1, auto=True, driver_mb=5000,
                             longest_seconds=120, config=_cfg()) == 3


def test_select_batch_size_auto_drops_when_current_driver_high() -> None:
    # 当前已占 7500：batch3 预计 9100 超限，batch2 预计 8300 仍超限 -> 1
    assert select_batch_size(requested=1, auto=True, driver_mb=7500,
                             longest_seconds=120, config=_cfg()) == 1


def test_select_batch_size_auto_reduces_for_long_chunks() -> None:
    assert select_batch_size(requested=1, auto=True, driver_mb=5000,
                             longest_seconds=200, config=_cfg()) == 2


def test_select_batch_size_never_below_one() -> None:
    assert select_batch_size(requested=1, auto=True, driver_mb=20000,
                             longest_seconds=300, config=_cfg()) == 1


# ---------------------------------------------------------------------------
# 配置校验
# ---------------------------------------------------------------------------

def test_config_rejects_bad_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        _cfg(tolerance=1.0)


def test_config_rejects_bad_batch_size() -> None:
    with pytest.raises(ValueError, match="max_batch"):
        _cfg(max_batch=0)


def test_config_rejects_bad_memory() -> None:
    with pytest.raises(ValueError, match="内存"):
        _cfg(per_item_mb=0)