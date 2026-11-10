"""字符级低置信标亮与段级复核优先级（P1/P2，2026-11 bench 实测）。

策略来源：``docs/asr-error-detection-research-2026-11.md`` 与
``backend/output/confidence/p1p2/report.md``。核心结论：

- 只有「单字 token」适合逐字标亮——多字 token 共享一个熵，标亮会整词误报；
- 标亮总量按全片信息字符的固定比例（默认 4%）截断，避免零错误段被强制标亮；
- 段内单字 token 最高熵低于门限时整段跳过；
- 附加信号按档位启用：
  - ``balanced``（默认）：按 entropy 排序，不依赖额外模型；
  - ``quality``：采样一致性参与打分，覆盖率低（约 1%）时精确率更高；
  - ``recall``：强制对齐时长异常参与打分，覆盖率宽（8%-10%）时召回更高。

bench_10min 实测（2197 字 / 57 错，重跑无 overlap 版本）：

    策略                         覆盖率   P       R
    entropy（单字，无预算）       3.5%    26.3%   35.1%
    entropy（单字，4% 预算）      4.0%    26.1%   40.4%
    entropy+采样（1% 预算）       1.0%    50.0%   19.3%
    entropy+对齐（10% 预算）     10.0%    19.5%   75.4%

注意：beam rank 在本模型上无增量信息（AUROC 0.500），默认不启用。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .segmentation import keep_char

# 单字 token 的熵下限（quality 档仍用它过滤极低熵字；balanced 档用统一下限）
LOW_CONF_MIN_ENTROPY = 0.2
# 全片标亮预算与候选最低分：低于该分的字即使预算未用满也不标。
# 2026-11-08 bench_10min 实测（真实错误率 1.49%，数字统一写法后）：
#   4% / 0.1 -> 标 87 字，P 23%、R 61%（误报过多，用户反馈"标的多是正确字"）
#   1% / 0.8 -> 标约 22 字，P 44%、R 30%
#   1% / 1.2 -> 标约 13 字，P 50%、R 20%
# 预算大于错误量时精确率有结构性上限（33 个真实错误、88 个预算 => P<=37%），
# 因此默认走「精度优先」：宁可少标，标出的更可信。
DEFAULT_MAX_RATIO = 0.01
LOW_CONF_SCORE_FLOOR = 0.8
# 段级门控：本段单字 token 的最高熵低于该值 → 整段不标
LOW_CONF_SEGMENT_MAX_ENTROPY = 0.7
# 文本侧风险的段级阈值：异常字符至少 2 个且占比 >= 1%
TEXT_ANOMALY_MIN_COUNT = 2
TEXT_ANOMALY_MIN_RATIO = 0.01

_PUNCT = set("，。！？；：、,.!?;:…—-()（）《》\"'“”‘’")
_LATIN_DIGIT = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


@dataclass
class CharSignal:
    """一个信息字符的多路置信信号；分数越高越可疑。"""

    index: int
    entropy: float = 0.0
    neg_logprob: float = 0.0
    neg_margin: float = 0.0
    single_token: bool = False
    token_len: int = 0
    beam_rank: float | None = None
    sample_agree: float | None = None
    align_short: bool = False
    align_long: bool = False
    align_abs_z: float = 0.0
    text_risk: bool = False
    # 与第二模型（FunASR）对齐后不一致：独立信号，bench 召回约为熵的 2 倍
    cross_mismatch: bool = False


@dataclass
class SegmentSignals:
    """一段的全部字符信号与段级汇总。"""

    seq: int
    text: str
    chars: list[CharSignal | None] = field(default_factory=list)
    max_single_entropy: float = 0.0
    mean_entropy: float = 0.0
    tail_max_entropy: float = 0.0
    text_anomaly_count: int = 0
    align_anomaly_count: int = 0
    spans: list[list[int]] = field(default_factory=list)
    risk_score: float = 0.0
    risk_flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# token → 字符映射
# ---------------------------------------------------------------------------

def align_ops(ref: str, hyp: str) -> list[tuple[str, int | None, int | None]]:
    """Levenshtein 回溯，返回逐操作对齐 (op, ref_idx, hyp_idx)。

    与 scripts/confidence_build_dataset.py 的实现同口径，复制到生产模块
    以避免运行时依赖 scripts 目录。
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri, row, prev = ref[i - 1], dp[i], dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
                0 if ref[i - 1] == hyp[j - 1] else 1):
            ops.append(("C" if ref[i - 1] == hyp[j - 1] else "S", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("D", i - 1, None))
            i -= 1
        else:
            ops.append(("I", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


# ---------------------------------------------------------------------------
# 第二模型交叉验证（两模型不一致 = 可疑）
# ---------------------------------------------------------------------------
# bench_10min 实测（Qwen3-ASR 0.6B vs FunASR Paraformer-zh）：
#   不一致           标 35 字  P=40.0%  R=42.4%
#   entropy 1%/0.8   标 16 字  P=43.8%  R=21.2%
# 即召回翻倍、精确率仅降约 4 个百分点；融合排序（entropy + 1.0×不一致）
# 在 2%/3% 预算下 P/R 双升。因此不一致给固定加分，由预算统一截断。
CROSS_BONUS = 1.0
# 交叉验证字符作为独立第二档：不受段门控与分数下限约束，但有安全上限。
# bench 实测「不一致」42 字 P=52%/R=58%，而叠加 1% 预算截断后只剩
# 25 字 P=52%/R=34%——预算会浪费召回，故单独成档。
CROSS_MAX_RATIO = 0.05

# 归一化：仅保留 CJK / 字母 / 数字（与 eval_cer、confidence_build_dataset 同口径）
_KEEP = re.compile(r"[^一-鿿㐀-䶿a-zA-Z0-9]")
# 语气词等价集：互换/增删不算错误（标签规则 v2）
FILLER = set("啊嘛哈呢吧哦呀嗯呃")

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_ARABIC = set("0123456789")
_NUM_CHARS = _ARABIC | set(_CN_DIGITS) | set(_CN_UNITS) | {"万"}


def _parse_cn_number(s: str) -> int | None:
    if not s:
        return None
    if any(ch in _CN_UNITS or ch == "万" for ch in s):
        total = section = num = 0
        for ch in s:
            if ch in _CN_DIGITS:
                num = _CN_DIGITS[ch]
            elif ch in _CN_UNITS:
                section += (num or 1) * _CN_UNITS[ch]
                num = 0
            elif ch == "万":
                section = (section + num) * 10000
                total += section
                section = num = 0
        return total + section + num
    try:
        return int("".join(str(_CN_DIGITS[ch]) for ch in s))
    except KeyError:
        return None


def normalize_numbers(text: str) -> tuple[str, list[int]]:
    """中文数字串 -> 阿拉伯数字，返回 (新文本, 新字符 -> 原文本下标)。

    只转换长度 >= 2 的纯数字串（避免把「一项」「一同」里的「一」改写）；
    两模型对同一个数常写成「二零零八」与「2008」，不统一会产生伪不一致。
    """
    out: list[str] = []
    mapping: list[int] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _NUM_CHARS:
            j = i
            while j < n and text[j] in _NUM_CHARS:
                j += 1
            run = text[i:j]
            if len(run) >= 2 and all(c in _CN_DIGITS or c in _CN_UNITS or c == "万"
                                     for c in run):
                value = _parse_cn_number(run)
                if value is not None:
                    digits = str(value)
                    out.extend(digits)
                    mapping.extend([i] * len(digits))
                    i = j
                    continue
            out.extend(run)
            mapping.extend(range(i, j))
            i = j
            continue
        out.append(ch)
        mapping.append(i)
        i += 1
    return "".join(out), mapping


def conservative_norm(text: str) -> tuple[str, list[int]]:
    """保守归一化：剥离标点/空白，返回 (归一化文本, 原文本字符下标映射)。"""
    kept, idx_map = [], []
    for i, ch in enumerate(text):
        if _KEEP.search(ch) is None:
            kept.append(ch)
            idx_map.append(i)
    return "".join(kept), idx_map


def apply_filler_rules(ops: list[tuple], ref: str, hyp: str) -> tuple[list[tuple], int]:
    """语气词互换/插入/漏识别不计为错误（标签规则 v2）。"""
    adj, ignored = [], 0
    for op, ri, hi in ops:
        if op == "S" and ri is not None and hi is not None \
                and ref[ri] in FILLER and hyp[hi] in FILLER:
            adj.append(("C", ri, hi))
            ignored += 1
        elif op == "I" and hi is not None and hyp[hi] in FILLER:
            adj.append(("C", None, hi))
            ignored += 1
        elif op == "D" and ri is not None and ref[ri] in FILLER:
            ignored += 1
        else:
            adj.append((op, ri, hi))
    return adj, ignored


def cross_model_flags(hyp_text: str, other_text: str) -> list[bool]:
    """第二模型交叉验证：返回与 ``hyp_text`` 等长的「不一致」标志。

    ``hyp_text`` 是主模型文本，``other_text`` 是第二模型的整片文本。
    两侧先做数字归一化与标点/空白剥离，再用 Levenshtein 对齐；
    hyp 侧的替换（S）与插入（I）都算不一致（插入常是重复/幻觉），
    参考侧的删除（D）没有 hyp 字符可标。语气词差异按规则 v2 忽略。
    """
    hyp_num, num_map = normalize_numbers(hyp_text)
    other_num, _ = normalize_numbers(other_text)
    hyp_norm, keep = conservative_norm(hyp_num)
    other_norm, _ = conservative_norm(other_num)
    ops = align_ops(other_norm, hyp_norm)
    ops, _ = apply_filler_rules(ops, other_norm, hyp_norm)
    flags = [False] * len(hyp_text)
    for op, _ri, hi in ops:
        if hi is None or op == "C":
            continue
        # 两层映射：对齐索引在归一化文本上 → num 索引 → 原文本索引。
        # 数字串会被压缩（「一百二十三」→「123」），漏掉第一层会整体
        # 向右错位，标亮落到无关字符上。
        idx = keep[hi]
        start = num_map[idx]
        # 同一数字串的多个归一化字符映射到同一起点：无法定位到具体某位，
        # 整串一起标（到下一个映射目标之前）。
        end_idx = idx
        while end_idx + 1 < len(num_map) and num_map[end_idx + 1] == start:
            end_idx += 1
        end = num_map[end_idx + 1] if end_idx + 1 < len(num_map) else len(hyp_text)
        for k in range(start, min(end, len(hyp_text))):
            flags[k] = True
    return flags


def sample_agreement(sample_texts: list[str], text: str) -> list[float | None]:
    """主文本每个字符被采样候选支持的平均比例（仅比较信息字符）。"""
    kept = [i for i, ch in enumerate(text) if keep_char(ch)]
    hyp = "".join(text[i] for i in kept)
    if not hyp:
        return [None] * len(text)
    flags = [0.0] * len(text)
    valid = 0
    for cand in sample_texts:
        ckept = "".join(ch for ch in cand if keep_char(ch))
        if not ckept:
            continue
        hit = [False] * len(text)
        for op, ri, hi in align_ops(ckept, hyp):
            if op == "C" and hi is not None:
                hit[kept[hi]] = True
        for i in kept:
            flags[i] += 1.0 if hit[i] else 0.0
        valid += 1
    if not valid:
        return [None] * len(text)
    return [float(x / valid) if text[i] and keep_char(text[i]) else None
            for i, x in enumerate(flags)]


def _align_items(aligner, audio_path: str, text: str,
                 language: str = "Chinese",
                 max_seconds: float | None = None) -> list[tuple[int, float, float]] | None:
    """调用强制对齐，返回 ``[(text 码点下标, start_s, end_s), ...]``。

    对齐器只处理字母/数字（中文按字），标点不参与；items 多字符时
    其内部每个字符共享该 item 的起止时间。

    按 V3.0 §5.5 的硬性规则丢弃无效单元：起止逆序、负时间、超出对齐窗口
    （``max_seconds`` + 0.5 s 容差）。被丢弃的字符不产生时间，前端回退插值。
    失败返回 None（调用方降级）。
    """
    try:
        res = aligner.align(str(audio_path), text, language)[0]
    except Exception:  # noqa: BLE001 - 对齐是可选增强
        return None
    kept_pos = [i for i, ch in enumerate(text) if _aligner_kept(ch)]
    seq: list[tuple[int, float, float]] = []
    cursor = 0  # 已消耗的信息字符数：无效 item 也要推进，避免后续字符错位
    for item in res.items:
        n_chars = len(item.text)
        positions = kept_pos[cursor:cursor + n_chars]
        cursor += n_chars
        start, end = float(item.start_time), float(item.end_time)
        if start < 0 or end < start:
            continue
        if max_seconds is not None and end > max_seconds + 0.5:
            continue
        seq.extend((pos, start, end) for pos in positions)
    return seq


def align_char_signals(
    aligner, audio_path: str, text: str, language: str = "Chinese",
    max_seconds: float | None = None,
) -> tuple[list[dict | None], list[tuple[float, float] | None]]:
    """一次对齐同时产出字符时长特征与起止时间（避免重复推理）。

    返回 ``(features, times)``，均与 ``text`` 等长，标点/未覆盖字符为 None。
    ``max_seconds`` 为音频时长，用于丢弃越界时间戳（V3.0 §5.5）。
    对齐失败返回两个全 None 列表，调用方降级为按字数比例插值。
    """
    feats: list[dict | None] = [None] * len(text)
    times: list[tuple[float, float] | None] = [None] * len(text)
    seq = _align_items(aligner, audio_path, text, language, max_seconds)
    if not seq:
        return feats, times
    for pos, start, end in seq:
        times[pos] = (start, end)
    durs = [max(0.0, e - s) for _, s, e in seq]
    ordered = sorted(d for d in durs if d > 0)
    med = ordered[len(ordered) // 2] if ordered else 0.0
    mad = sorted(abs(d - med) for d in durs)[len(durs) // 2] if durs else 0.0
    prev_end: float | None = None
    for pos, start, end in seq:
        dur = max(0.0, end - start)
        feats[pos] = {
            "align_dur": dur,
            "align_dur_z": (dur - med) / (mad + 1e-6) if mad else 0.0,
            "align_overlap": max(0.0, prev_end - start) if prev_end is not None else 0.0,
            "align_short": dur < 0.06 or (med > 0 and dur < 0.25 * med),
            "align_long": dur > 1.2 or (med > 0 and dur > 3.5 * med),
        }
        prev_end = end
    return feats, times


def align_char_times(aligner, audio_path: str, text: str,
                     language: str = "Chinese",
                     max_seconds: float | None = None) -> list[tuple[float, float] | None]:
    """强制对齐的字符起止时间（秒，相对 ``audio_path``），供句级跳转。"""
    return align_char_signals(aligner, audio_path, text, language, max_seconds)[1]


def align_char_features(aligner, audio_path: str, text: str,
                        language: str = "Chinese",
                        max_seconds: float | None = None) -> list[dict | None]:
    """强制对齐的字符时长特征；失败时返回全 None，不中断转写。"""
    return align_char_signals(aligner, audio_path, text, language, max_seconds)[0]


def _aligner_kept(ch: str) -> bool:
    """与 Qwen3-ForcedAligner 的文本切分口径一致：字母/数字/撇号。"""
    import unicodedata
    return ch == "'" or unicodedata.category(ch)[0] in ("L", "N")


def merge_positions(sorted_pos: list[int]) -> list[list[int]]:
    """把有序字符位置合并为 [start, end) 连续区间。"""
    merged: list[list[int]] = []
    for pos in sorted_pos:
        if merged and pos == merged[-1][1]:
            merged[-1][1] = pos + 1
        else:
            merged.append([pos, pos + 1])
    return merged


def token_char_spans(token_features, text: str, tokenizer) -> list[tuple[int, int, dict]]:
    """把逐 token 特征映射为 text 上的 (start, end, features) 跨度。

    重新 tokenize 转写文本，用逐 token 解码长度累加得到字符区间，
    保证与 ASR 实际生成的 token 序列一一对应。
    """
    if not token_features or not text.strip() or tokenizer is None:
        return []
    try:
        tids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:  # noqa: BLE001 - tokenizer 异常不应中断转写
        return []
    if not tids:
        return []
    n = min(len(tids), len(token_features))
    spans: list[tuple[int, int, dict]] = []
    cumulative = 0
    for idx in range(n):
        token_text = tokenizer.decode([tids[idx]], clean_up_tokenization_spaces=False)
        if not token_text:
            continue
        start = min(cumulative, len(text))
        cumulative += len(token_text)
        end = min(cumulative, len(text))
        if end <= start:
            continue  # 逐 token 解码可能比原文本长，越界部分直接丢弃
        spans.append((start, end, token_features[idx]))
    return spans


def char_feature_series(token_features, text: str, tokenizer,
                        key: str) -> list[tuple[int, float]]:
    """返回每个信息字符的 (text 字符索引, 指定 token 特征值)。"""
    indexed: list[tuple[int, float]] = []
    for s, e, feats in token_char_spans(token_features, text, tokenizer):
        val = float(feats.get(key, 0.0))
        for ch_i in range(s, e):
            ch = text[ch_i] if ch_i < len(text) else ""
            if keep_char(ch):
                indexed.append((ch_i, val))
    return indexed


# ---------------------------------------------------------------------------
# 文本侧特征
# ---------------------------------------------------------------------------

def text_char_features(text: str) -> list[dict]:
    """文本侧启发式：重复片段、连续标点、长拉丁/数字串、段尾位置。

    只用于段级风险，不直接逐字标亮——bench 实测其字符级精确率低于 entropy。
    """
    n = len(text)
    out = [{"repeat": False, "punct_run": False,
            "latin_run": False, "tail": False} for _ in range(n)]
    for i in range(n):
        if n > 10 and i >= 0.9 * n:
            out[i]["tail"] = True
        if text[i] in _PUNCT:
            j = i
            while j + 1 < n and text[j + 1] in _PUNCT:
                j += 1
            if j > i:
                for k in range(i, j + 1):
                    out[k]["punct_run"] = True
        if text[i] in _LATIN_DIGIT:
            j = i
            while j + 1 < n and text[j + 1] in _LATIN_DIGIT:
                j += 1
            if j - i + 1 >= 4:
                for k in range(i, j + 1):
                    out[k]["latin_run"] = True
        for k in (1, 2, 3):
            if i + 2 * k <= n and text[i:i + k] == text[i + k:i + 2 * k]:
                for p in range(i, i + 2 * k):
                    out[p]["repeat"] = True
    return out


# ---------------------------------------------------------------------------
# 信号收集
# ---------------------------------------------------------------------------

def build_char_signals(token_features, text: str, tokenizer) -> list[CharSignal | None]:
    """把逐 token 特征展开到字符，生成 CharSignal 列表（与 text 等长）。"""
    signals: list[CharSignal | None] = [None] * len(text)
    if tokenizer is None:
        return signals
    for s, e, feats in token_char_spans(token_features, text, tokenizer):
        kept = [i for i in range(s, min(e, len(text))) if keep_char(text[i])]
        if not kept:
            continue
        for pos in kept:
            signals[pos] = CharSignal(
                index=pos,
                entropy=float(feats.get("entropy", 0.0)),
                neg_logprob=float(feats.get("neg_logprob", 0.0)),
                neg_margin=float(feats.get("neg_margin", 0.0)),
                single_token=len(kept) == 1,
                token_len=len(kept),
                beam_rank=(float(feats["beam_rank"])
                           if feats.get("beam_rank") is not None else None),
            )
    return signals


def collect_segment_signals(
    token_features,
    text: str,
    tokenizer,
    *,
    seq: int = 0,
    sample_agreement: list[float | None] | None = None,
    align_features: list[dict | None] | None = None,
) -> SegmentSignals:
    """构建一段的字符信号与段级汇总。"""
    signals = build_char_signals(token_features, text, tokenizer)
    tfeats = text_char_features(text)
    for i, (sig, tf) in enumerate(zip(signals, tfeats)):
        if sig is not None:
            sig.text_risk = bool(tf["repeat"] or tf["punct_run"] or tf["latin_run"])
    if sample_agreement:
        for i, agree in enumerate(sample_agreement):
            if i < len(signals) and signals[i] is not None and agree is not None:
                signals[i].sample_agree = float(agree)
    if align_features:
        for i, af in enumerate(align_features):
            if i < len(signals) and signals[i] is not None and af:
                signals[i].align_short = bool(af.get("align_short"))
                signals[i].align_long = bool(af.get("align_long"))
                signals[i].align_abs_z = abs(float(af.get("align_dur_z") or 0.0))

    seg = SegmentSignals(seq=seq, text=text, chars=signals)
    ents = [s.entropy for s in signals if s is not None]
    singles = [s.entropy for s in signals if s is not None and s.single_token]
    seg.max_single_entropy = max(singles or [0.0])
    seg.mean_entropy = sum(ents) / len(ents) if ents else 0.0
    tail_from = int(len(signals) * 0.9)
    tail = [s.entropy for s in signals[tail_from:]
            if s is not None and s.single_token]
    seg.tail_max_entropy = max(tail or [0.0])
    seg.text_anomaly_count = sum(1 for s in signals if s is not None and s.text_risk)
    seg.align_anomaly_count = sum(
        1 for s in signals if s is not None and (s.align_short or s.align_long))
    return seg


# ---------------------------------------------------------------------------
# 打分与全局预算
# ---------------------------------------------------------------------------

def score_signal(sig: CharSignal, mode: str = "balanced") -> float:
    """单字符风险分（越高越可疑）。各档位只在附加信号存在时加分。"""
    score = sig.entropy
    if mode == "quality" and sig.sample_agree is not None:
        # 采样不支持（agree<1）是强信号；1% 覆盖率下 bench P 50%
        score += 0.8 * (1.0 - sig.sample_agree)
    elif mode == "recall":
        score += 0.35 * min(sig.align_abs_z, 4.0)
        score += 0.3 * int(sig.align_short) + 0.3 * int(sig.align_long)
    if sig.cross_mismatch:
        # 与第二模型不一致：与档位无关的独立加分（bench 2% 预算 P 33→37%）
        score += CROSS_BONUS
    return score


def _gate_open(seg: SegmentSignals) -> bool:
    """段级门控：单字 token 最高熵不足时整段跳过。

    recall 档允许「对齐异常 >=2」的段绕过门控（吞字/错位的段熵可能不高）。
    """
    if seg.max_single_entropy >= LOW_CONF_SEGMENT_MAX_ENTROPY:
        return True
    return seg.align_anomaly_count >= 2


def select_low_conf(
    segments: list[SegmentSignals],
    *,
    mode: str = "balanced",
    max_ratio: float = DEFAULT_MAX_RATIO,
    score_floor: float = LOW_CONF_SCORE_FLOOR,
) -> int:
    """全局预算截断，把选中位置写回每段的 ``spans``。

    - 候选：单字 token（多字 token 会整词误报）；
    - 排除低于 ``score_floor`` 的极低熵字；
    - 按全片信息字符数的 ``max_ratio`` 取 top-K；
    - **第二档**：与第二模型不一致（``cross_mismatch``）的字符不受段门控
      与 ``score_floor`` 限制，按 ``CROSS_MAX_RATIO`` 上限追加（去重）。
      这类字符熵往往不高（模型很自信但两模型不同），走常规门控会漏掉。
    返回标亮字符总数。
    """
    total_info = sum(1 for seg in segments for ch in seg.text if keep_char(ch))
    budget = max(1, int(math.ceil(total_info * max(0.0, max_ratio))))
    cross_budget = max(1, int(math.ceil(total_info * CROSS_MAX_RATIO)))
    cands: list[tuple[float, int, CharSignal]] = []
    cross_cands: list[tuple[float, int, CharSignal]] = []
    for seg in segments:
        seg.spans = []
        gate = _gate_open(seg)
        for sig in seg.chars:
            if sig is None:
                continue
            if sig.cross_mismatch:
                # 交叉验证档只看融合分排序（含 CROSS_BONUS），不受门控/floor；
                # 且**不占常规预算**——否则加分后会把纯 entropy 的高价值候选
                # 挤出预算（bench 实测 P 52%→27%），第二档本身已覆盖它们。
                cross_cands.append((score_signal(sig, mode), seg.seq, sig))
                continue
            if not gate or not sig.single_token:
                continue
            score = score_signal(sig, mode)
            if score < score_floor:
                continue
            cands.append((score, seg.seq, sig))
    cands.sort(key=lambda x: x[0], reverse=True)
    picked = cands[:budget]
    by_seg: dict[int, list[int]] = {}
    seen: set[tuple[int, int]] = set()
    for _, seq, sig in picked:
        by_seg.setdefault(seq, []).append(sig.index)
        seen.add((seq, sig.index))
    cross_cands.sort(key=lambda x: x[0], reverse=True)
    added = 0
    for _, seq, sig in cross_cands:
        if added >= cross_budget:
            break
        if (seq, sig.index) in seen:
            continue
        by_seg.setdefault(seq, []).append(sig.index)
        seen.add((seq, sig.index))
        added += 1
    marked = 0
    for seg in segments:
        positions = sorted(by_seg.get(seg.seq, []))
        seg.spans = merge_positions(positions)
        marked += len(positions)
    return marked


def finalize_segment_risk(seg: SegmentSignals) -> None:
    """段级风险分与 risk_flags（供复核排序与前端提示）。"""
    spans_chars = sum(e - s for s, e in seg.spans)
    seg.risk_score = (
        seg.max_single_entropy
        + 0.3 * seg.tail_max_entropy
        + 0.1 * seg.mean_entropy
        + 0.2 * seg.text_anomaly_count
        + 0.15 * seg.align_anomaly_count
    )
    flags: list[str] = []
    if spans_chars:
        flags.append("low_conf_review")
    info_chars = sum(1 for ch in seg.text if keep_char(ch))
    if (seg.text_anomaly_count >= TEXT_ANOMALY_MIN_COUNT
            and seg.text_anomaly_count >= TEXT_ANOMALY_MIN_RATIO * max(info_chars, 1)):
        flags.append("text_anomaly")
    if seg.align_anomaly_count >= TEXT_ANOMALY_MIN_COUNT:
        flags.append("alignment_anomaly")
    if seg.tail_max_entropy >= LOW_CONF_SEGMENT_MAX_ENTROPY and seg.spans:
        flags.append("boundary_uncertain")
    seg.risk_flags = flags