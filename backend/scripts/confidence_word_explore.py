# 低置信标亮探索：误报归因、预算/阈值扫描、词级聚合方案对比。
#
# 输入是 ASR stats.json（需带 --dump-token-features，含 char_signals）
# 与 LiPS 参考文本。只读脚本，不改 DB/产物。
#
# 用法：
#   python backend/scripts/confidence_word_explore.py \
#       data/bench/conf_dump/raw.stats.json data/bench/bench_10min_reference.txt \
#       [第二模型文本.txt]
#
# 第三个参数可选：传入 FunASR 等第二模型的识别文本，追加「两模型不一致」
# 信号的 P/R 评估。stats 需带 --dump-token-features 才能做 token 分桶/
# 预算扫描/采样与对齐信号；缺 char_signals 时这些段落自动跳过。
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import jieba

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
# 注意顺序：src 必须在 scripts 之前——scripts/local_asr.py 是兼容垫片，
# 若 scripts 排在前面会把 local_asr 解析成单文件模块，导入 local_asr.asr 失败
sys.path.insert(0, str(BACKEND_ROOT / "scripts"))
sys.path.insert(0, str(BACKEND_ROOT / "src"))

from local_asr.asr.confidence import (  # noqa: E402
    align_ops, apply_filler_rules, conservative_norm, cross_model_flags,
    normalize_numbers,
)
from local_asr.asr.segmentation import apply_dedup_to_texts  # noqa: E402


def load_segments(stats_path: Path) -> list[dict]:
    """读 stats 段，并应用与入库一致的边界去重（stats 存的是 chunk 原文）。

    去重会在头/尾裁字符，因此 char_signals 的索引、low_conf_spans 都要
    同步平移，否则特征会错位到别的字上。
    """
    data = json.loads(stats_path.read_text(encoding="utf-8"))
    raw = data.get("segments") or []
    orig_texts = [s.get("text") or "" for s in raw]
    records = (data.get("segmentation") or {}).get("dedup") or []
    final_texts = apply_dedup_to_texts(orig_texts, records)
    out = []
    for i, s in enumerate(raw):
        orig = orig_texts[i]
        final = final_texts[i]
        lo = orig.find(final)
        if lo < 0:  # 兜底：映射失败时保持原文
            lo, final = 0, orig
        hi = lo + len(final)
        signals = {}
        for row in s.get("char_signals") or []:
            pos = int(row[0])
            if lo <= pos < hi:
                signals[pos - lo] = row
        spans = []
        for a, b in (s.get("low_conf_spans") or []):
            a2, b2 = max(int(a), lo) - lo, min(int(b), hi) - lo
            if b2 > a2:
                spans.append((a2, b2))
        out.append({
            "seq": i,
            "text": final,
            "spans": spans,
            "signals": signals,
        })
    return out


def build_hyp(segments: list[dict]):
    """拼接归一化文本；返回 hyp_norm、每归一化字符的原(段, 段内下标)、标亮 flag。"""
    parts: list[str] = []
    origin: list[tuple[int, int]] = []
    marked: list[bool] = []
    for seg in segments:
        norm_text, norm_map = normalize_numbers(seg["text"])
        norm, keep_idx = conservative_norm(norm_text)
        # 归一化字符 -> 原文本下标
        orig_of_norm = [norm_map[k] for k in keep_idx]
        spans = seg["spans"]
        flag_by_orig = [False] * len(seg["text"])
        for s, e in spans:
            for k in range(s, min(e, len(flag_by_orig))):
                flag_by_orig[k] = True
        parts.append(norm)
        origin.extend((seg["seq"], k) for k in orig_of_norm)
        marked.extend(flag_by_orig[k] for k in orig_of_norm)
    return "".join(parts), origin, marked


def labels_for(ref_text: str, hyp_norm: str) -> list[int | None]:
    ref_num, _ = normalize_numbers(ref_text)
    ref_norm, _ = conservative_norm(ref_num)
    ops = align_ops(ref_norm, hyp_norm)
    ops, _ = apply_filler_rules(ops, ref_norm, hyp_norm)
    labels: list[int | None] = [None] * len(hyp_norm)
    for op, ri, hi in ops:
        if hi is not None:
            labels[hi] = 1 if op == "C" else 0
    return labels


def pr(marked: list[bool], labels: list[int | None]) -> dict:
    labelled = [i for i, l in enumerate(labels) if l is not None]
    errs = [i for i in labelled if labels[i] == 0]
    marks = [i for i in labelled if marked[i]]
    tp = [i for i in marks if labels[i] == 0]
    return {
        "chars": len(labelled), "errors": len(errs), "marked": len(marks),
        "tp": len(tp),
        "p": len(tp) / len(marks) if marks else 0.0,
        "r": len(tp) / len(errs) if errs else 0.0,
    }


def bucket_table(segments: list[dict], origin, labels) -> None:
    """按字符 entropy 分桶看真实错误率（校准曲线）。"""
    rows = []
    for i, (seq, pos) in enumerate(origin):
        if labels[i] is None:
            continue
        sig = segments[seq]["signals"].get(pos)
        if sig is None:
            continue
        ent = sig[1]
        single = bool(sig[5])
        rows.append((ent, single, 1 if labels[i] == 0 else 0))
    if not rows:
        # stats 未带 char_signals（服务层未传 --dump-token-features）时无 token 特征
        print("\n--- entropy 分桶错误率：stats 无 char_signals，跳过 ---")
        return
    print("\n--- entropy 分桶错误率（全字符）---")
    edges = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 99]
    for lo, hi in zip(edges, edges[1:]):
        band = [e for ent, single, e in rows if lo <= ent < hi]
        if band:
            print(f"  [{lo:>4}, {hi:>4}): {len(band):>5} 字  错误率 {sum(band)/len(band):.2%}")
    print("\n--- 单字 token vs 多字 token ---")
    for label, want in (("单字 token", True), ("多字 token", False)):
        band = [e for ent, single, e in rows if single == want]
        print(f"  {label}: {len(band):>5} 字  错误率 {sum(band)/len(band):.2%}"
              if band else f"  {label}: 无")


def fusion_sweep(segments, labels, origin,
                 ratios=(0.01, 0.02, 0.03, 0.04)) -> None:
    """重放段级门控 + 全局预算，对比纯 entropy 与 quality（entropy+采样）融合。"""
    gate = {}
    for seg in segments:
        max_ent = max((float(s[1]) for s in seg["signals"].values() if s[5]),
                      default=0.0)
        gate[seg["seq"]] = max_ent >= 0.7
    total_info = sum(1 for seg in segments for ch in seg["text"] if not ch.isspace())
    rows: list[tuple[int, float, float]] = []  # (origin 下标, entropy 分, quality 分)
    for i, (seq, pos) in enumerate(origin):
        sig = segments[seq]["signals"].get(pos)
        if sig is None or not sig[5] or not gate.get(seq):
            continue
        ent = float(sig[1])
        agree = sig[6] if len(sig) > 6 else None
        qual = ent + (0.8 * (1.0 - float(agree)) if agree is not None else 0.0)
        rows.append((i, ent, qual))
    if not rows:
        return
    has_sample = any(len(s[6:]) > 0 and s[6] is not None
                     for seg in segments for s in seg["signals"].values())
    print("\n--- 段门控 + 预算重放（纯 entropy vs 采样融合）---")
    print("  ratio   纯 entropy        quality(融合)")
    for ratio in ratios:
        k = max(1, math.ceil(total_info * ratio))
        cells = []
        for col in (1, 2):
            picked = {i for i, _, _ in sorted(rows, key=lambda r: -r[col])[:k]}
            flags = [i in picked for i in range(len(origin))]
            m = pr(flags, labels)
            cells.append(f"P={m['p']:>4.0%} R={m['r']:>4.0%}")
        if not has_sample:
            cells[1] = "（未启用采样）"
        print(f"  {ratio:>4.0%}   {cells[0]}   {cells[1]}")


def budget_sweep(segments, labels, origin, floors, ratios) -> None:
    """重放 select_low_conf 的预算逻辑：候选=单字 token 且 entropy>=floor。"""
    total_info = sum(1 for seg in segments for ch in seg["text"] if not ch.isspace())
    cands = []
    for seg in segments:
        for pos, sig in seg["signals"].items():
            if not sig[5]:  # 只取单字 token
                continue
            cands.append((float(sig[1]), seg["seq"], pos))
    if not cands:
        print("\n--- 预算 × 分数下限扫描：无 token 特征（char_signals），跳过 ---")
        return
    print("\n--- 预算 × 分数下限扫描（字符级 P/R）---")
    print("  floor\\ratio " + "  ".join(f"{r:>4.0%}" for r in ratios))
    for floor in floors:
        pool = sorted((c for c in cands if c[0] >= floor), key=lambda x: -x[0])
        cells = []
        for ratio in ratios:
            k = max(1, math.ceil(total_info * ratio))
            picked = {(seq, pos) for _, seq, pos in pool[:k]}
            marked = [
                (segments[seq]["signals"].get(pos) is not None and (seq, pos) in picked
                 and not _is_punct(segments[seq]["text"][pos]))
                for (seq, pos) in origin
            ]
            m = pr(marked, labels)
            cells.append(f"{m['p']:>4.0%}")
        print(f"  {floor:>5.2f}      " + "  ".join(cells))


def _is_punct(ch: str) -> bool:
    import re
    return re.match(r"[^\w]", ch) is not None and not ch.isalnum()


# char_signals 列定义（与 qwen3_asr_transcribe 写入一致）：
#   [index, entropy, neg_logprob, neg_margin, token_len, single,
#    sample_agree, align_abs_z, align_short, align_long, cross_mismatch]
def signal_table(segments: list[dict], origin, labels) -> None:
    """独立信号评估：采样一致性（quality 档）与对齐异常（recall 档）。"""
    rows = []
    for i, (seq, pos) in enumerate(origin):
        if labels[i] is None:
            continue
        sig = segments[seq]["signals"].get(pos)
        if sig is None:
            continue
        # labels: 1=正确、0=错误；这里统一成 1=错误，便于算错误率
        rows.append((sig, 1 if labels[i] == 0 else 0))
    if not rows:
        print("\n--- 独立信号：无 char_signals，跳过 ---")
        return

    # --- 采样一致性：值与 1 的差越小越可信 ---
    agree_rows = [(float(s[6]), e) for s, e in rows if len(s) > 6 and s[6] is not None]
    if agree_rows:
        print("\n--- 采样一致性（sample_agree）分桶错误率 ---")
        for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 1.01)):
            band = [e for a, e in agree_rows if lo <= a < hi]
            if band:
                print(f"  [{lo:.2f}, {hi:.2f}): {len(band):>5} 字  "
                      f"错误率 {sum(band) / len(band):.2%}")
        print("\n--- 阈值扫描：标亮 sample_agree < t 的字符 ---")
        for t in (0.25, 0.5, 0.75, 1.0):
            flags = _flags_from_signals(segments, origin, lambda s, t=t: bool(
                len(s) > 6 and s[6] is not None and float(s[6]) < t))
            m = pr(flags, labels)
            print(f"  t={t:.2f}: 标亮 {m['marked']:>4} 字  "
                  f"P={m['p']:.1%}  R={m['r']:.1%}")
    else:
        print("\n--- 采样一致性：本 stats 未启用 --sample-check，跳过 ---")

    # --- 强制对齐异常：短/长时长偏离 ---
    align_rows = [s for s, _ in rows if len(s) > 9]
    if align_rows and any(s[8] or s[9] for s in align_rows):
        print("\n--- 强制对齐异常（align_short / align_long）---")
        for name, col in (("时长过短", 8), ("时长过长", 9)):
            flags = _flags_from_signals(segments, origin, lambda s, col=col: bool(
                len(s) > col and s[col]))
            m = pr(flags, labels)
            print(f"  {name}: 标亮 {m['marked']:>4} 字  "
                  f"P={m['p']:.1%}  R={m['r']:.1%}")


def _flags_from_signals(segments, origin, pred) -> list[bool]:
    """按字符信号生成标亮 flags；pred 接收该字符的 char_signals 行。"""
    flags = [False] * len(origin)
    for i, (seq, pos) in enumerate(origin):
        sig = segments[seq]["signals"].get(pos)
        if sig is not None and pred(sig):
            flags[i] = True
    return flags


def cross_model_table(labels, hyp_norm: str, other_text: str,
                      marked: list[bool] | None = None) -> None:
    """输出交叉验证一致性的 P/R（独立信号，不做预算）。

    传入 ``marked`` 时额外输出「第二档并集」：不一致字符全量追加到当前
    标亮（``select_low_conf`` 的 CROSS_MAX_RATIO 档），这是生产落地口径。
    """
    flags = cross_model_flags(hyp_norm, other_text)
    m = pr(flags, labels)
    print("\n--- 第二模型交叉验证（不一致 = 可疑）---")
    print(f"  标亮 {m['marked']:>4} 字  P={m['p']:.1%}  R={m['r']:.1%}")
    if marked is not None:
        union = [a or b for a, b in zip(marked, flags)]
        m2 = pr(union, labels)
        print(f"  + 当前标亮并集 {m2['marked']:>4} 字  "
              f"P={m2['p']:.1%}  R={m2['r']:.1%}")


def word_level(segments: list[dict], labels, origin, marked) -> None:
    """词级聚合：标亮扩展到整词，评估字符级与词级 P/R。"""
    # 归一化字符 -> 词语 id；用段原文分词，再映射到归一化字符
    word_of_char: list[int | None] = [None] * len(origin)
    words: dict[int, list[int]] = {}
    wid = 0
    for seg in segments:
        text = seg["text"]
        pieces = jieba.lcut(text)
        char_word: list[int] = []
        for piece in pieces:
            for _ in piece:
                char_word.append(wid)
            wid += 1
        for i, (seq, pos) in enumerate(origin):
            if seq == seg["seq"] and pos < len(char_word):
                word_of_char[i] = char_word[pos]
    for i, w in enumerate(word_of_char):
        if w is not None:
            words.setdefault(w, []).append(i)

    # 方案 A：当前字符标亮 -> 扩展到整词
    expand = list(marked)
    for w, idxs in words.items():
        if any(marked[i] for i in idxs):
            for i in idxs:
                expand[i] = True
    # 方案 B：词级平均熵 top-K（词内所有字符 entropy 均值）
    word_score: list[tuple[float, int, list[int]]] = []
    for w, idxs in words.items():
        ents = []
        for i in idxs:
            seq, pos = origin[i]
            sig = segments[seq]["signals"].get(pos)
            if sig is not None:
                ents.append(float(sig[1]))
        if ents:
            word_score.append((sum(ents) / len(ents), w, idxs))
    word_score.sort(key=lambda x: -x[0])

    print("\n--- 词级聚合 ---")
    for name, flags in (("字符标亮扩到整词", expand), ("保持字符级", marked)):
        m = pr(flags, labels)
        print(f"  {name:<14} 标亮 {m['marked']:>4} 字  P={m['p']:.1%}  R={m['r']:.1%}")

    if not word_score:
        # 无 token 特征时，词级平均熵不可算；扩词方案已在上表给出
        print("  词级平均熵：无 token 特征（char_signals），跳过")
        return
    total_words = sum(1 for w, idxs in words.items() if idxs)
    for ratio in (0.01, 0.02, 0.03):
        k = max(1, int(total_words * ratio))
        picked_words = {w for _, w, _ in word_score[:k]}
        if not picked_words:
            continue
        flags = [False] * len(origin)
        for w in picked_words:
            for i in words.get(w, []):
                flags[i] = True
        m = pr(flags, labels)
        # 词级：标亮词中至少 1 个错误字符的比例
        word_hit = sum(1 for w in picked_words
                       if any(labels[i] == 0 for i in words.get(w, []) if labels[i] is not None))
        print(f"  词级平均熵 top {ratio:.0%}（{len(picked_words)} 词）: "
              f"字级 P={m['p']:.1%} R={m['r']:.1%} | 词级命中率={word_hit/len(picked_words):.1%}")


def main() -> None:
    stats_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        REPO_ROOT / "data/bench/conf_dump/raw.stats.json"
    ref_path = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        REPO_ROOT / "data/bench/bench_10min_reference.txt"
    segments = load_segments(stats_path)
    hyp_norm, origin, marked = build_hyp(segments)
    labels = labels_for(ref_path.read_text(encoding="utf-8"), hyp_norm)
    # 可选第三个参数：第二模型的识别文本，用于交叉验证评估
    other_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    m = pr(marked, labels)
    print("=" * 64)
    print("低置信标亮探索（数字归一化后）")
    print("=" * 64)
    print(f"有效字符 {m['chars']}  真实错误 {m['errors']}（错误率 {m['errors']/m['chars']:.2%}）")
    print(f"当前标亮 {m['marked']} 字  TP={m['tp']}  P={m['p']:.1%}  R={m['r']:.1%}")
    bucket_table(segments, origin, labels)
    budget_sweep(segments, labels, origin,
                 floors=(0.1, 0.3, 0.5, 0.8, 1.2), ratios=(0.01, 0.02, 0.03, 0.04))
    signal_table(segments, origin, labels)
    fusion_sweep(segments, labels, origin)
    if other_path is not None and other_path.exists():
        cross_model_table(labels, hyp_norm,
                          other_path.read_text(encoding="utf-8"), marked)
    word_level(segments, labels, origin, marked)


if __name__ == "__main__":
    main()