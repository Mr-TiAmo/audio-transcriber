"""长音频切分规划与边界去重。

只依赖标准库：VAD 推理由调用方注入语音段列表，本模块不做任何模型加载，
因此可以在 `.venv` 下直接单测。

切分原则（来自实测）：
- 切点取「目标长度区间内最长静音的中点」，而不是第一个超长边界；
  切在 100 ms 短停顿上会导致下一块开头复述上一句。
- 每块两侧各扩展 ``overlap_seconds``，用途是给边界词提供声学上下文；
  重叠区通常是静音，模型不会复述，因此去重是惰性兜底而非必做步骤。
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "BoundaryOverlap",
    "Chunk",
    "Cut",
    "SegmentationConfig",
    "VadSegment",
    "align_boundary",
    "build_chunks",
    "decide_mode",
    "drop_prefix_by_chars",
    "drop_suffix_by_chars",
    "effective_len",
    "estimate_max_new_tokens",
    "keep_char",
    "merge_texts",
    "merge_short_gaps",
    "overlap_prefix_len",
    "plan_balanced_cuts",
    "plan_cuts",
]


@dataclass(frozen=True)
class VadSegment:
    """VAD 检测到的一段语音区间（毫秒）。"""

    start_ms: int
    end_ms: int


def merge_short_gaps(
    segments: Sequence[VadSegment],
    *,
    max_gap_ms: int = 800,
    min_speech_ms: int = 300,
) -> list[VadSegment]:
    """合并 VAD 输出的短间隙，保持语义连续性。

    FSMN-VAD 有时会把一句话中的短暂停顿当作段边界切碎。
    合并条件：两段语音之间的停顿 ≤ ``max_gap_ms``。

    同时丢弃过短的孤立语音段（< ``min_speech_ms``），
    因为 200-300ms 的"语音"通常是咳嗽/点击声等非语义声音。
    """
    if not segments:
        return []
    merged: list[VadSegment] = []
    cur_start = segments[0].start_ms
    cur_end = segments[0].end_ms
    for i in range(1, len(segments)):
        gap = segments[i].start_ms - cur_end
        if gap <= max_gap_ms and gap >= 0:
            cur_end = segments[i].end_ms
        else:
            if cur_end - cur_start >= min_speech_ms:
                merged.append(VadSegment(cur_start, cur_end))
            cur_start = segments[i].start_ms
            cur_end = segments[i].end_ms
    if cur_end - cur_start >= min_speech_ms:
        merged.append(VadSegment(cur_start, cur_end))
    return merged


@dataclass(frozen=True)
class Cut:
    """一个切分点及其右侧判断依据。"""

    index: int
    start_ms: int
    end_ms: int
    cut_silence_ms: int | None = None
    forced: bool = False


@dataclass(frozen=True)
class Chunk:
    """实际送去识别的音频区间（含两侧扩展）。"""

    index: int
    core_start_ms: int
    core_end_ms: int
    audio_start_ms: int
    audio_end_ms: int
    cut_silence_ms: int | None = None
    forced: bool = False


@dataclass(frozen=True)
class SegmentationConfig:
    """切分策略参数，与 CLI 默认值保持一致。"""

    mode: str = "auto"                 # full | vad | auto
    full_max_seconds: float = 300.0
    hard_max_seconds: float = 1200.0
    target_chunk_seconds: float = 120.0
    chunk_slack: tuple[float, float] = (0.7, 1.3)
    min_silence_ms: int = 200
    max_chunk_seconds: float = 180.0
    overlap_seconds: float = 0.5
    chunk_strategy: str = "silence"     # silence | balanced | dp
    balance_tolerance: float = 0.15     # balanced：切点偏离理想位置的容差（占目标段长比例）
    balance_min_silence_ms: int = 40    # balanced：可吸附的最小静音（连续口播里 200 ms 太稀缺）
    balance_jitter_seconds: float = 0.0 # balanced：理想位置±该秒数内找停顿，避开词边界
    # dp：硬切代价（相对「块长偏离」项的量级）。60 分钟真实音频实测：
    # 0.3 → 吸附率 63.6%、块长标准差 8.53；0.6 → 70.4%、13.82。
    # 默认取 0.3，在静音吸附与块长均匀（batch 效率）之间折中。
    dp_forced_penalty: float = 0.3
    dp_gap_bonus: float = 0.15          # dp：切在静音上的奖励（gap 按 1 秒封顶归一）

    def __post_init__(self) -> None:
        if self.mode not in ("full", "vad", "auto"):
            raise ValueError(f"未知切分模式: {self.mode}")
        if self.chunk_strategy not in ("silence", "balanced", "dp"):
            raise ValueError(f"未知切片策略: {self.chunk_strategy}")
        if self.target_chunk_seconds <= 0 or self.max_chunk_seconds <= 0:
            raise ValueError("切分长度必须为正数")
        if not (0 < self.chunk_slack[0] < 1 < self.chunk_slack[1]):
            raise ValueError("chunk_slack 必须满足 0 < lo < 1 < hi")
        if not 0 <= self.balance_tolerance < 0.5:
            raise ValueError("balance_tolerance 必须满足 0 <= tol < 0.5")
        if self.overlap_seconds < 0 or self.min_silence_ms < 0 or self.balance_min_silence_ms < 0:
            raise ValueError("overlap_seconds / min_silence_ms 不能为负")
        if self.balance_jitter_seconds < 0:
            raise ValueError("balance_jitter_seconds 不能为负")
        if self.dp_forced_penalty < 0 or self.dp_gap_bonus < 0:
            raise ValueError("dp 代价权重不能为负")


def decide_mode(duration_seconds: float, config: SegmentationConfig) -> str:
    """按音频时长决定 full / vad。"""
    if config.mode in ("full", "vad"):
        return config.mode
    if duration_seconds > config.hard_max_seconds:
        return "vad"
    return "full" if duration_seconds <= config.full_max_seconds else "vad"


def estimate_max_new_tokens(chunk_seconds: float, *, cap: int = 8192) -> int:
    """按实测 token 增长率（约 3.05 tokens/秒）估算生成预算。

    1.25 倍余量用于覆盖语速快、标点密的片段；下限 512 与模型卡默认值对齐。
    """
    estimate = math.ceil(max(chunk_seconds, 0.0) * 3.05 * 1.25)
    return max(512, min(cap, estimate))


def plan_cuts(
    segments: Sequence[VadSegment],
    total_ms: int,
    config: SegmentationConfig,
) -> list[Cut]:
    """把 VAD 语音段规划为若干切分区间。

    每一刀都落在目标长度 ±slack 区间内**最长静音**的中点；
    若该区间内没有达到 ``min_silence_ms`` 的静音，则退化为按
    ``max_chunk_seconds`` 硬切，并标记 ``forced``。
    """
    if total_ms <= 0:
        return [Cut(index=0, start_ms=0, end_ms=0)]

    target = config.target_chunk_seconds * 1000.0
    lo_ratio, hi_ratio = config.chunk_slack
    gaps = [
        (segments[i + 1].start_ms - segments[i].end_ms, i)
        for i in range(len(segments) - 1)
    ]

    cuts: list[Cut] = []
    start = 0
    index = 0
    max_chunk_ms = int(config.max_chunk_seconds * 1000)

    while total_ms - start > target * hi_ratio:
        lo = start + target * lo_ratio
        hi = start + target * hi_ratio
        candidates = [
            (gap, i) for gap, i in gaps
            if gap > 0 and lo <= segments[i].end_ms <= hi
        ]
        usable = [(gap, i) for gap, i in candidates if gap >= config.min_silence_ms]

        if usable:
            # 最长静音优先；长度相同取更靠后的边界，让块更接近目标长度
            gap, i = max(usable, key=lambda item: (item[0], segments[item[1]].end_ms))
            cut_ms = (segments[i].end_ms + segments[i + 1].start_ms) // 2
            cuts.append(Cut(index=index, start_ms=start, end_ms=cut_ms,
                            cut_silence_ms=gap, forced=False))
        else:
            cut_ms = min(start + max_chunk_ms, total_ms)
            cuts.append(Cut(index=index, start_ms=start, end_ms=cut_ms,
                            cut_silence_ms=None, forced=True))
            if cut_ms >= total_ms:
                start = cut_ms
                break
        start = cut_ms
        index += 1

    if start < total_ms or not cuts:
        cuts.append(Cut(index=index, start_ms=start, end_ms=total_ms))
    return cuts


def plan_balanced_cuts(
    segments: Sequence[VadSegment],
    total_ms: int,
    config: SegmentationConfig,
) -> list[Cut]:
    """均衡动态切分：先按总时长均分，再把每个理论切点吸附到附近的静音。

    与 :func:`plan_cuts` 的区别是「长度优先」而非「静音优先」：

    - ``plan_cuts`` 从当前块出发找目标区间里最长的静音，切完一块再决定下一块，
      块长会在 0.7~1.3 倍目标间漂移，连续口播里还会出现 180 s 硬切；
    - ``plan_balanced_cuts`` 先算出块数 ``n = round(total / target)``，
      再让每个切点落在理想位置 ±tolerance 内的静音上，块长标准差小得多，
      分桶批处理的 padding 浪费随之下降（实测 32 分钟音频：块长标准差 37 s → 9 s）。

    吸附优先级：静音更长优先，其次离理想位置更近；窗口内没有可用静音时
    退回理想位置硬切并标记 ``forced``。

    ``balance_jitter_seconds`` 在 ``balance_tolerance`` 之外再给理想位置一个
    固定的微调窗口（秒），用于在严格等长（tolerance=0）时仍能避开语音内部：
    例如 ±0.25 s 只在理想切点附近找停顿，块长偏差不超过 ±0.25 s，
    批内 padding 代价约 1%。停顿太少时可放宽到 1~2 s，代价是块长差异变大。
    """
    if total_ms <= 0:
        return [Cut(index=0, start_ms=0, end_ms=0)]
    target_ms = config.target_chunk_seconds * 1000.0
    n = max(1, round(total_ms / target_ms))
    span = total_ms / n
    tol = span * config.balance_tolerance + config.balance_jitter_seconds * 1000.0
    min_len = span * (1 - config.balance_tolerance) / 2

    candidates = []
    for i in range(len(segments) - 1):
        gap = segments[i + 1].start_ms - segments[i].end_ms
        if gap >= config.balance_min_silence_ms:
            candidates.append(((segments[i].end_ms + segments[i + 1].start_ms) // 2, gap))

    bounds: list[int] = [0]
    silences: list[int | None] = []
    forced_flags: list[bool] = []
    for i in range(1, n):
        ideal = total_ms * i / n
        prev = bounds[-1]
        lo = max(prev + min_len, ideal - tol)
        hi = ideal + tol
        window = [c for c in candidates if lo <= c[0] <= hi]
        if window:
            cut, gap = max(window, key=lambda c: (c[1], -abs(c[0] - ideal)))
            silences.append(gap)
            forced_flags.append(False)
        else:
            cut = int(round(min(max(ideal, prev + min_len), total_ms)))
            silences.append(None)
            forced_flags.append(True)
        bounds.append(int(cut))
    bounds.append(total_ms)

    return [
        Cut(index=i, start_ms=bounds[i], end_ms=bounds[i + 1],
            cut_silence_ms=None if i == n - 1 else silences[i],
            forced=False if i == n - 1 else forced_flags[i])
        for i in range(n)
    ]


def plan_dp_cuts(
    segments: Sequence[VadSegment],
    total_ms: int,
    config: SegmentationConfig,
) -> list[Cut]:
    """动态规划全局最优切分（长音频）。

    与 :func:`plan_balanced_cuts` 的区别：后者每个切点独立吸附静音，是
    局部贪心——前面的点偏晚会让后续窗口变窄、吸附失败。本函数把整条音频的
    切点序列作为一个整体求最优：

    - 节点：全局所有 VAD 停顿中点 + 每个理论边界的硬切兜底点；
    - 状态：``dp[j]`` = 到达候选节点 j 的最小累计代价；
    - 转移代价：``((块长-span)/span)²`` + 硬切惩罚 − 静音长度奖励；
    - 约束：块长限制在 ``[0.5*span, 1.8*span]`` 且不超过 ``max_chunk``。

    与 balanced 的本质差异：balanced 要求切点必须落在理想位置 ±tolerance
    内，找不到就硬切；DP 允许在块长约束内借用相邻停顿点，用代价函数全局
    权衡「块长均匀」与「切在静音上」。

    复杂度 O(n·m)（m 为候选节点数），60 分钟音频通常在毫秒级。
    无可行路径时回退 :func:`plan_balanced_cuts`。
    """
    if total_ms <= 0:
        return [Cut(index=0, start_ms=0, end_ms=0)]
    target_ms = config.target_chunk_seconds * 1000.0
    n = max(1, round(total_ms / target_ms))
    if n == 1:
        return [Cut(index=0, start_ms=0, end_ms=total_ms)]

    span = total_ms / n
    min_len = max(1000.0, span * 0.5)
    max_len = min(config.max_chunk_seconds * 1000.0, span * 1.8)

    # 候选节点：全局所有 VAD 停顿中点，不与「理论切点±tol」绑定——
    # 这正是 DP 相对 balanced 的价值：理想位置附近没有停顿时，可以借用
    # 相邻的停顿点，由块长约束与代价权衡，而不是直接硬切。
    candidates: dict[int, tuple[bool, int]] = {}
    for j in range(len(segments) - 1):
        gap = segments[j + 1].start_ms - segments[j].end_ms
        if gap < config.balance_min_silence_ms:
            continue
        pos = (segments[j].end_ms + segments[j + 1].start_ms) // 2
        if 0 < pos < total_ms:
            candidates[pos] = (True, gap)
    # 理想位置的硬切兜底：停顿过稀时保证有解，并给 DP 一个「均匀」参考点
    for i in range(1, n):
        candidates.setdefault(int(round(total_ms * i / n)), (False, 0))

    nodes = ([(0, True, 0)]
             + [(pos, sil, gap) for pos, (sil, gap) in sorted(candidates.items())]
             + [(total_ms, True, 0)])
    m = len(nodes)
    inf = float("inf")
    dp = [inf] * m
    back = [-1] * m
    dp[0] = 0.0

    for j in range(1, m):
        pos_j, silence_j, gap_j = nodes[j]
        for i in range(j - 1, -1, -1):
            length = pos_j - nodes[i][0]
            if length > max_len:
                break  # 节点按位置有序，i 继续前移只会更长
            if length < min_len and j != m - 1:
                continue
            if length < min_len * 0.5:
                continue
            step = ((length - span) / span) ** 2
            if j != m - 1:  # 末节点是总时长，不产生切点
                step += 0.0 if silence_j else config.dp_forced_penalty
                if silence_j:
                    step -= config.dp_gap_bonus * min(gap_j, 1000) / 1000.0
            if dp[i] + step < dp[j]:
                dp[j] = dp[i] + step
                back[j] = i

    if back[m - 1] < 0:
        # 约束过紧（如整段无停顿且块长跨度不够）：回退逐点吸附，保证可用
        return plan_balanced_cuts(segments, total_ms, config)

    path = [m - 1]
    while path[-1] != 0:
        path.append(back[path[-1]])
    path.reverse()

    cuts: list[Cut] = []
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        pos, silence, gap = nodes[b]
        is_last = b == m - 1
        cuts.append(Cut(
            index=k,
            start_ms=nodes[a][0],
            end_ms=pos,
            cut_silence_ms=(None if is_last or not silence else gap),
            forced=(not is_last and not silence),
        ))
    return cuts


def build_chunks(
    cuts: Sequence[Cut],
    total_ms: int,
    config: SegmentationConfig,
) -> list[Chunk]:
    """给每个切分区间加上两侧扩展，得到实际送去识别的音频区间。"""
    overlap_ms = int(config.overlap_seconds * 1000)
    return [
        Chunk(
            index=c.index,
            core_start_ms=c.start_ms,
            core_end_ms=c.end_ms,
            audio_start_ms=max(0, c.start_ms - overlap_ms),
            audio_end_ms=min(total_ms, c.end_ms + overlap_ms),
            cut_silence_ms=c.cut_silence_ms,
            forced=c.forced,
        )
        for c in cuts
    ]


# ---------------------------------------------------------------------------
# 边界去重（惰性触发）
# ---------------------------------------------------------------------------

def keep_char(char: str) -> bool:
    """参与边界比较的字符：字母、数字、汉字；标点与空白不参与。

    空串返回 False：token 展开越界时会出现，避免 unicodedata 抛错。
    """
    if not char:
        return False
    if char == ".":
        return True
    category = unicodedata.category(char)
    return category.startswith("L") or category.startswith("N")


def _kept_chars(text: str) -> list[tuple[str, int]]:
    return [(ch, i) for i, ch in enumerate(text) if keep_char(ch)]


def overlap_prefix_len(
    prev: str,
    nxt: str,
    *,
    window: int = 16,
    min_overlap: int = 4,
) -> int:
    """返回 ``nxt`` 开头应裁掉的**有效字符数**；无重叠返回 0。

    只比较前一文本尾部与后一文本头部各 ``window`` 个有效字符，
    避免正文自带重复（如「止盈…止盈」）被误删。
    """
    if window <= 0 or min_overlap <= 0:
        return 0
    tail = "".join(ch for ch, _ in _kept_chars(prev)[-window:])
    head = "".join(ch for ch, _ in _kept_chars(nxt)[:window])
    upper = min(len(tail), len(head))
    for k in range(upper, min_overlap - 1, -1):
        if tail.endswith(head[:k]):
            return k
    return 0


# 块边界可清理的标点：句末与停顿标点（不含括号，避免误删原文包裹符）
_BOUNDARY_PUNCT = frozenset("。！？；，、：!?;,:")
# 尾部额外清理右引号：块末引语被截断时会残留半个引号（「渴望。”」+「超过了」）
_TAIL_PUNCT = _BOUNDARY_PUNCT | frozenset("”’")


def strip_tail_punct(text: str) -> tuple[str, int]:
    """剥掉文本末尾的标点/空白，返回 ``(新文本, 去掉的字符数)``。

    用于块边界：上一块末尾的标点是 ASR 对**截断音频**补出来的，
    并不对应真实停顿。若保留，拼接后会得到「游泳运动概。」「栩栩如。」
    这类词中句号；去掉后下一块开头的真实标点正好接上。
    """
    end = len(text)
    while end > 0 and (text[end - 1] in _TAIL_PUNCT or text[end - 1].isspace()):
        end -= 1
    return text[:end], len(text) - end


def strip_head_punct(text: str) -> tuple[str, int]:
    """剥掉文本开头的标点/空白，返回 ``(新文本, 去掉的字符数)``。

    仅在上一块尾部确实存在块末伪标点时使用（见 ``strip_tail_punct``），
    避免段首出现「，在英国…」「。游泳是…」这类突兀标点。
    """
    start = 0
    while start < len(text) and (text[start] in _BOUNDARY_PUNCT or text[start].isspace()):
        start += 1
    return text[start:], start


def drop_prefix_by_chars(text: str, count: int) -> str:
    """按有效字符数裁掉开头；``count <= 0`` 原样返回。"""
    if count <= 0:
        return text
    seen = 0
    for i, char in enumerate(text):
        if not keep_char(char):
            continue
        seen += 1
        if seen >= count:
            return text[i + 1:]
    return ""


def drop_suffix_by_chars(text: str, count: int) -> str:
    """按有效字符数裁掉结尾（连同其后的标点）；``count <= 0`` 原样返回。"""
    if count <= 0:
        return text
    seen = 0
    for i in range(len(text) - 1, -1, -1):
        if not keep_char(text[i]):
            continue
        seen += 1
        if seen >= count:
            return text[:i]
    return ""


def effective_len(text: str) -> int:
    """参与边界比较的有效字符数（不含标点与空白）。"""
    return sum(1 for char in text if keep_char(char))


@dataclass(frozen=True)
class BoundaryOverlap:
    """相邻两块在拼接点的重叠判定结果。

    ``drop_prefix`` 是下一块开头与上一块重复、应裁掉的有效字符数；
    ``trim_prev`` 是上一块尾部转录与下一块不一致、应一并裁掉的有效字符数。
    两者都只在该边界生效，不跨段累积。
    """

    drop_prefix: int = 0
    trim_prev: int = 0
    method: str = "none"
    edit_distance: int = 0


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def align_boundary(
    prev: str,
    nxt: str,
    *,
    window: int = 16,
    min_overlap: int = 4,
    max_trim: int | None = None,
    expected: int | None = None,
    edge_slack: int = 3,
    max_edit_ratio: float = 0.25,
    fuzzy: bool = True,
) -> BoundaryOverlap:
    """判定拼接点重叠：先精确匹配尾部后缀，失败再做模糊对齐。

    硬切时同一段重叠音频在两侧可能被转写成不同字（如「岛民都是」vs
    「岛民啊都会」），精确后缀匹配会漏判、把重叠内容拼两遍。这里的模糊
    对齐允许少量错字，并把上一块尾部那段不一致的转录一并裁掉。

    约束（防止误删正文）：
    - 匹配块必须从下一块的第 0 个有效字符开始；
    - 匹配块在上一块里的位置必须贴近尾部（``edge_slack`` 个有效字符内）；
    - 模糊匹配至少 6 个有效字符；长度受 ``max_trim`` 限制；
    - 传入 ``expected``（重叠音频估算出的字符数）时，优先选长度最接近的
      候选，否则要求编辑距离不超过 ``max_edit_ratio``。
    """
    if window <= 0 or min_overlap <= 0:
        return BoundaryOverlap()
    tail = "".join(ch for ch, _ in _kept_chars(prev)[-window:])
    head = "".join(ch for ch, _ in _kept_chars(nxt)[:window])
    upper = min(len(tail), len(head))
    if upper < min_overlap:
        return BoundaryOverlap()

    # 1) 精确后缀匹配（原行为）：上一块结尾正好是下一块开头
    for k in range(upper, min_overlap - 1, -1):
        if tail.endswith(head[:k]):
            return BoundaryOverlap(k, 0, "exact", 0)

    # 2) 模糊对齐：匹配块可以落在尾部稍前的位置
    if not fuzzy:
        return BoundaryOverlap()
    if expected is not None and expected < 5:
        # 重叠音频太短，模糊匹配的证据不足，只保留精确后缀
        return BoundaryOverlap()
    cap = upper if max_trim is None else min(upper, max_trim)
    floor = max(min_overlap, 6)
    if cap < floor:
        return BoundaryOverlap()
    # 下一块开头可能先冒出几个属于重叠区、但被上一块放在更早位置的字
    # （如上一块尾部「…可以Google」、下一块开头「学的朋友呢可以Google」），
    # 允许匹配块从下一块头部偏移 max_offset 个字符开始。
    max_offset = min(6, max(0, len(head) - floor))
    ratio = 0.34 if expected is not None else max_edit_ratio
    exact: list[tuple[int, int, int]] = []      # (drop_prefix, start, 0)
    fuzzy: list[tuple[int, int, int]] = []      # (k, start, distance)
    for offset in range(max_offset + 1):
        for k in range(cap, floor - 1, -1):
            if offset + k > len(head):
                break
            limit = max(1, int(k * ratio))
            for start in range(max(0, len(tail) - k - edge_slack), len(tail) - k + 1):
                segment = tail[start:start + k]
                if len(segment) < k:
                    continue
                distance = _edit_distance(head[offset:offset + k], segment)
                if distance == 0:
                    exact.append((offset + k, start, k))
                elif distance <= limit:
                    fuzzy.append((k, start, distance))
    if exact:
        drop_prefix, start, k = max(exact, key=lambda c: c[0])
        distance = 0
    elif fuzzy:
        k, start, distance = max(fuzzy, key=lambda c: (c[0] - 2 * c[2], c[0]))
        drop_prefix = None
    else:
        return BoundaryOverlap()
    if drop_prefix is None:
        # 模糊命中时，按匹配块长度裁掉下一块前缀（偏移未知，按 0 处理）
        drop_prefix = k
    leftover = max(0, len(tail) - (start + k))
    return BoundaryOverlap(drop_prefix, min(leftover, edge_slack), "fuzzy", distance)


def merge_texts(
    texts: Sequence[str],
    *,
    window: int = 16,
    min_overlap: int = 4,
    expected_overlap: Sequence[int] | None = None,
    edge_slack: int = 3,
    fuzzy: bool = True,
) -> tuple[str, list[dict]]:
    """按顺序拼接分块文本，惰性去重。

    ``expected_overlap`` 是每个边界重叠音频大致对应的有效字符数
    （由调用方按重叠秒数与语速估算），用来给模糊对齐的裁剪长度设上限，
    避免把正文误当成重叠删掉。缺省时退化为 ``window``。

    返回 ``(合并文本, 去重记录)``；无重叠时不裁剪，记录为空。
    """
    if not texts:
        return "", []
    merged = texts[0]
    records: list[dict] = []
    expected = list(expected_overlap or [])
    for i in range(1, len(texts)):
        cap = window
        expected_value = None
        if i - 1 < len(expected):
            expected_value = expected[i - 1]
            cap = min(window, max(min_overlap, int(math.ceil(expected_value * 1.5)) + 1))
        match = align_boundary(
            texts[i - 1], texts[i], window=window, min_overlap=min_overlap,
            max_trim=cap, expected=expected_value, edge_slack=edge_slack,
            fuzzy=fuzzy)
        if not match.drop_prefix:
            merged += texts[i]
            continue
        piece = drop_prefix_by_chars(texts[i], match.drop_prefix)
        record = {
            "boundary": i - 1,
            "dropped_chars": match.drop_prefix,
            "dropped": texts[i][: len(texts[i]) - len(piece)],
            "method": match.method,
            "edit_distance": match.edit_distance,
        }
        if match.trim_prev:
            before = merged
            merged = drop_suffix_by_chars(merged, match.trim_prev)
            record["trimmed_prev_chars"] = match.trim_prev
            record["trimmed_prev"] = before[len(merged):]
        # 块边界标点：上一块在词中被截断时模型会补句号（「概述。」）。
        # 下一块开头若带标点，说明该处确有真实停顿 → 保留上一块的标点、
        # 去掉下一块重复标点；下一块开头直接接字 → 上一块的标点是伪影，
        # 去掉后拼接才连续（「概」+「述」）。
        before = merged
        stripped_prev, tail_cut = strip_tail_punct(merged)
        stripped_piece, head_cut = strip_head_punct(piece)
        if tail_cut and not head_cut:
            merged = stripped_prev
            record["trimmed_tail_punct"] = before[len(merged):]
        elif tail_cut and head_cut:
            piece = stripped_piece
            record["trimmed_head_punct"] = True
        records.append(record)
        merged += piece
    return merged, records


def apply_dedup_to_texts(
    texts: Sequence[str], records: Sequence[dict]
) -> list[str]:
    """把 ``merge_texts`` 的逐边界去重记录回写到各分块文本。

    去重此前只作用于合并全文（导出/复制），入库 segments 仍是各 chunk
    原文，导致页面句行在 chunk 边界出现拼接两遍的技术重复（页面≠导出）。
    回写规则与 merge_texts 内部等价：``dropped_chars`` 裁下一块开头，
    ``trimmed_prev_chars`` 裁上一块尾部；各边界独立、不跨边界累积
    （align_boundary 比较的正是各块原文，故逐边界回写不改变判定）。
    """
    out = list(texts)
    for rec in records:
        i = int(rec["boundary"])
        trimmed_prev = int(rec.get("trimmed_prev_chars") or 0)
        dropped = int(rec.get("dropped_chars") or 0)
        if trimmed_prev and 0 <= i < len(out):
            out[i] = drop_suffix_by_chars(out[i], trimmed_prev)
        if dropped and 0 <= i + 1 < len(out):
            out[i + 1] = drop_prefix_by_chars(out[i + 1], dropped)
            # 与 merge_texts 同一规则：下一块开头带标点则保留上一块标点、
            # 去掉下一块重复标点；否则上一块的标点是块末伪影，去掉。
            stripped_prev, tail_cut = strip_tail_punct(out[i])
            stripped_next, head_cut = strip_head_punct(out[i + 1])
            if tail_cut and not head_cut:
                out[i] = stripped_prev
            elif tail_cut and head_cut:
                out[i + 1] = stripped_next
    return out