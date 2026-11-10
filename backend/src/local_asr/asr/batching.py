"""按块长分桶与 batch 规模决策（纯标准库）。

实测依据（1.7B / MPS fp16）：

- 等长块 batch ×3 相对逐条串行 **1.9x**；
- 块长差 1.4 倍时只有 **1.15x** —— padding 到最长块并按最长序列同步解码，
  长度差越大收益越低；
- 每多一路约 +0.8 GB 显存；
- 线程并发 **0.94x**（MPS 的 Metal command queue 在设备层串行化），因此不做并发。

所以 P3 只做「分桶 + batch」：先把长度相近的块分到一组，再决定批大小。
本模块不触碰 torch，内存采样由调用方注入，便于在 `.venv` 下直接单测。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from local_asr.asr.segmentation import estimate_max_new_tokens

__all__ = [
    "BatchConfig",
    "group_budget",
    "plan_buckets",
    "select_batch_size",
    "split_groups",
]


@dataclass(frozen=True)
class BatchConfig:
    """分桶与内存护栏参数。"""

    tolerance: float = 0.2          # 组内 (最长-最短)/最长 的上限
    max_batch: int = 3              # 16 GB 机器实测上限
    per_item_mb: int = 800          # 每多一路的显存增量估算
    base_driver_mb: int = 5500      # 单条推理的 driver 内存基线
    max_driver_mb: int = 8000       # 超过则降档
    long_chunk_seconds: float = 180.0  # 超长块一律降一档

    def __post_init__(self) -> None:
        if not 0 <= self.tolerance < 1:
            raise ValueError("tolerance 必须满足 0 <= tolerance < 1")
        if self.max_batch < 1:
            raise ValueError("max_batch 必须 >= 1")
        if self.per_item_mb <= 0 or self.base_driver_mb < 0 or self.max_driver_mb <= 0:
            raise ValueError("内存参数必须为正数")


def plan_buckets(
    lengths: Mapping[int, float],
    config: BatchConfig = BatchConfig(),
) -> list[list[int]]:
    """按块长把块分到若干桶，返回索引分组（组内按长度升序）。

    贪心：按长度升序排列，依次塞进当前组；一旦加入会让
    ``(最长 - 最短) / 最长`` 超过 ``tolerance``，或组已满 ``max_batch``，
    就开新组。长度相近才拼批，这正是 1.9x 与 1.15x 的分界。
    """
    if not lengths:
        return []
    ordered = sorted(lengths.items(), key=lambda item: (item[1], item[0]))
    groups: list[list[int]] = []
    current: list[int] = []
    shortest = 0.0
    for index, seconds in ordered:
        if not current:
            current, shortest = [index], seconds
            continue
        # 升序遍历，seconds 即加入后的最长块；用加入后的跨度判断
        fits = len(current) < config.max_batch and (
            seconds <= 0 or (seconds - shortest) <= config.tolerance * seconds
        )
        if fits:
            current.append(index)
        else:
            groups.append(current)
            current, shortest = [index], seconds
    groups.append(current)
    return groups


def split_groups(groups: Sequence[Sequence[int]], limit: int) -> list[list[int]]:
    """把每个桶按 ``limit`` 再切小，保持桶内顺序。"""
    if limit < 1:
        raise ValueError("limit 必须 >= 1")
    result: list[list[int]] = []
    for group in groups:
        for start in range(0, len(group), limit):
            result.append(list(group[start:start + limit]))
    return result


def group_budget(group: Sequence[int], seconds: Mapping[int, float]) -> int:
    """一组共用的生成预算：按组内最长块估算（generate 同步解码）。"""
    return max((estimate_max_new_tokens(seconds[g]) for g in group), default=512)


def select_batch_size(
    *,
    requested: int,
    auto: bool,
    driver_mb: float,
    longest_seconds: float,
    config: BatchConfig = BatchConfig(),
) -> int:
    """决定实际批大小。

    - ``auto=False``：直接用 ``min(requested, max_batch)``；
    - ``auto=True``：先取 ``max_batch``，遇到超长块或内存不足逐档下调；
    - 任何情况下都不低于 1。
    """
    if requested < 1:
        raise ValueError("requested 必须 >= 1")
    size = min(config.max_batch if auto else requested, config.max_batch)
    if auto and longest_seconds > config.long_chunk_seconds:
        size -= 1
    while size > 1:
        baseline = max(float(driver_mb), float(config.base_driver_mb))
        predicted = baseline + config.per_item_mb * (size - 1)
        if predicted <= config.max_driver_mb:
            break
        size -= 1
    return max(1, size)