#!/usr/bin/env python3
"""两阶段后处理 A/B 对比。

对同一份原始识别文本的不同后处理变体（A 基线 = 仅 ITN；B = 纠错 + ITN）
统一计算 CER、S/D/I、数值保真、修改统计与运行指标，输出 markdown 表。

用法:
    ./.venv/bin/python scripts/compare_postprocess.py \
        --ref output/测试音频_实际字幕版.txt \
        --candidate output/postprocess/qwen3-1.7b \
        --candidate output/postprocess/qwen3-1.7b-cec4 \
        --raw output/bench/qwen3-asr-1.7b_raw.txt

说明：修改准确率是基于文本对齐的启发式估计，不是人工逐条标注。
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from eval_cer import edit_stats, normalize  # noqa: E402
from check_numeric_fidelity import compare as numeric_compare  # noqa: E402
from local_asr.postprocess.artifacts import (
    json_hash, read_text_exact, runtime_versions, source_fingerprint, text_hash,
)
from local_asr.postprocess.itn import normalize_numbers


def load_variant(base: Path) -> dict | None:
    """读取一个候选变体的产物。"""
    stem = base.name
    directory = base.parent
    final = directory / f"{stem}_final.txt"
    if not final.exists():
        print(f"跳过（缺少 {final}）")
        return None
    corrected = directory / f"{stem}_corrected.txt"
    stats_path = directory / f"{stem}_postprocess.json"
    edits_path = directory / f"{stem}_edits.jsonl"
    if not stats_path.exists():
        print(f"跳过（缺少完成标记 {stats_path}）")
        return None
    stats = json.loads(read_text_exact(stats_path))
    if stats.get("status") not in ("complete", "degraded"):
        print(f"跳过（未确认完成状态，历史产物需单独核对）: {stem}")
        return None
    raw_path = directory / f"{stem}_raw.txt"
    if not raw_path.is_file():
        raise ValueError(f"缺少原始输入，不能做同源比较: {raw_path}")
    raw_text = read_text_exact(raw_path)
    if stats.get("input_sha256") != text_hash(raw_text):
        raise ValueError(f"原文指纹不符: {stem}")
    final_text = read_text_exact(final)
    corrected_text = read_text_exact(corrected)
    if (stats.get("final_sha256") != text_hash(final_text) or
            stats.get("corrected_sha256") != text_hash(corrected_text)):
        raise ValueError(f"产物指纹不符: {stem}")
    edits = []
    if edits_path.exists():
        for line in edits_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                edits.append(json.loads(line))
    return {
        "name": stem,
        "raw_text": raw_text,
        "final_text": final_text,
        "corrected_text": corrected_text,
        "stats": stats,
        "edits": edits,
    }


def edit_score(raw_text: str, corrected_text: str | None, ref_norm: str) -> tuple[int, int, int]:
    """启发式评估纠错改动：修复 / 伤害 / 中性。

    在归一化后的文本上比对改动片段，看改动后的写法是否更接近参考。
    """
    if not corrected_text:
        return 0, 0, 0
    a = normalize(raw_text)
    b = normalize(corrected_text)
    fixed = broken = neutral = 0
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        # 单字改动用前后各 6 字的上下文窗口判断，避免单字过于常见导致误判
        before = a[max(0, i1 - 6):min(len(a), i2 + 6)]
        after = b[max(0, j1 - 6):min(len(b), j2 + 6)]
        if len(before) < 2 or len(after) < 2:
            continue
        before_hit = before in ref_norm
        after_hit = after in ref_norm
        if after_hit and not before_hit:
            fixed += 1
        elif before_hit and not after_hit:
            broken += 1
        else:
            neutral += 1
    return fixed, broken, neutral


def evaluate_text(name: str, ref_text: str, hypothesis: str) -> dict:
    ref_norm = normalize(ref_text)
    sub, dele, ins, dist = edit_stats(ref_norm, normalize(hypothesis))
    literal_distance = edit_stats(ref_text, hypothesis)[3]
    num = numeric_compare(ref_text, hypothesis)
    return {
        "name": name,
        "cer": dist / len(ref_norm) * 100 if ref_norm else float("nan"),
        "literal_cer": literal_distance / len(ref_text) * 100 if ref_text else float("nan"),
        "sub": sub, "del": dele, "ins": ins,
        "numeric": num["rate"] * 100, "numeric_bag": num["multiset_rate"] * 100,
        "edits": "-", "elapsed": "-", "rss": "-",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--candidate", action="append", required=True,
                    help="产物路径前缀（不含 _final.txt 等后缀），可重复")
    ap.add_argument("--raw", required=True, help="固定原始识别文本；所有变体必须来自同一份 raw")
    ap.add_argument("--glossary", default=None, help="与候选一致的实体词典，用于仅 ITN 基线")
    args = ap.parse_args()

    ref_raw = read_text_exact(Path(args.ref))
    ref_norm = normalize(ref_raw)
    raw_text = read_text_exact(Path(args.raw))
    glossary = json.loads(read_text_exact(Path(args.glossary))) if args.glossary else {}
    rules_hash = source_fingerprint()
    wetext_version = runtime_versions()["wetext"]
    baseline = normalize_numbers(raw_text, glossary)
    if baseline.degraded:
        raise ValueError("仅 ITN 基线发生降级，不能标为正常对照")
    rows = [evaluate_text("raw（ASR 原始）", ref_raw, raw_text),
            evaluate_text("仅 ITN（同一 raw）", ref_raw, baseline.text)]

    for cand in args.candidate:
        v = load_variant(Path(cand))
        if v is None:
            continue
        st = v["stats"]
        if v["raw_text"] != raw_text or st.get("glossary_sha256") != json_hash(glossary):
            raise ValueError(f"候选 {v['name']} 的 raw 或词典与基线不同，拒绝比较")
        if (st.get("rules_sha256") != rules_hash or
                st.get("runtime_versions", {}).get("wetext") != wetext_version):
            raise ValueError(f"候选 {v['name']} 的规则或 wetext 版本与基线不同，拒绝比较")
        state = "（降级）" if st["status"] == "degraded" else ""
        rows.append(evaluate_text(v["name"] + " 仅纠错" + state, ref_raw, v["corrected_text"]))
        row = evaluate_text(v["name"] + " 最终版" + state, ref_raw, v["final_text"])
        row.update({
            "edits": f"{st.get('edits_accepted', 0)}/{st.get('edits_review', 0)}",
            "elapsed": round(st["elapsed_seconds"], 3),
            "rss": st.get("peak_rss_mb", "-"),
        })
        rows.append(row)
        if raw_text is not None and v["corrected_text"] is not None:
            fixed, broken, neutral = edit_score(raw_text, v["corrected_text"], ref_norm)
            print(f"[{v['name']}] 改动评估（启发式）：修复 {fixed}，伤害 {broken}，中性 {neutral}")

    print()
    print("归一化 CER 及 S/D/I 仅作历史对照；数字位置/频次为参考召回口径，不代表纠错准确率，仍需人工标注。")
    print("耗时仅覆盖后处理至 stats 发布前；RSS 为该进程生命周期峰值，不是独立后处理增量或跨进程全链路峰值。")
    print("| 变体 | 原文CER | 归一化CER | S | D | I | 数值位置命中 | 数值频次命中 | 改动(接受/复核) | 耗时(s) | 进程峰值RSS(MB) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['literal_cer']:.2f}% | {r['cer']:.2f}% | {r['sub']} | {r['del']} | {r['ins']} "
              f"| {r['numeric']:.1f}% | {r['numeric_bag']:.1f}% | {r['edits']} | {r['elapsed']} | {r['rss']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())