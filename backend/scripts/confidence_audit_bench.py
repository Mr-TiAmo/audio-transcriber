# 置信度标亮对账：把 bench_10min 实际入库的 low_conf_spans 与 LiPS 参考文本
# 逐字符对齐，量化「标亮是否真的落在错误字符上」。
#
# 只读脚本：不修改 DB / 产物，输出 JSON + 控制台报告。
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from confidence_build_dataset import conservative_norm, align_ops, apply_filler_rules  # noqa: E402

DB_PATH = Path(r"D:\work\workspace\audio-transcriber\data\database\app.db")
REF_PATH = Path(r"D:\work\workspace\audio-transcriber\data\bench\bench_10min_reference.txt")
RUN_ID = "run_dbf4e07a270c"
OUT_PATH = PROJECT_ROOT / "output" / "confidence" / "bench_audit.json"


def load_segments(db_path: Path, run_id: str) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT sequence_no, original_text, low_conf_spans FROM transcript_segments "
        "WHERE run_id=? ORDER BY sequence_no", (run_id,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        spans = json.loads(r["low_conf_spans"]) if r["low_conf_spans"] else []
        out.append({
            "seq": r["sequence_no"],
            "text": r["original_text"],
            "spans": [tuple(map(int, s)) for s in spans],
        })
    return out


def load_segments_from_stats(stats_path: Path) -> list[dict]:
    """从 ASR stats.json 直接读段文本与 low_conf_spans（不经过 DB）。"""
    data = json.loads(stats_path.read_text(encoding="utf-8"))
    out = []
    for i, s in enumerate(data.get("segments") or []):
        out.append({
            "seq": i,
            "text": s.get("text") or "",
            "spans": [tuple(map(int, x)) for x in (s.get("low_conf_spans") or [])],
        })
    return out


def build_hyp(segments: list[dict]) -> tuple[str, list[bool], list[tuple[int, int]], list[str]]:
    """拼接各段归一化文本，返回：
    hyp_norm、每字符是否标亮、每段在全局的 [start, end)、全局字符所属段序号。
    """
    parts: list[str] = []
    flags: list[bool] = []
    seg_ranges: list[tuple[int, int]] = []
    seg_of_char: list[str] = []
    cursor = 0
    for seg in segments:
        norm, idx_map = conservative_norm(seg["text"])
        # 原文索引 -> 归一化索引
        inv = {orig: ni for ni, orig in enumerate(idx_map)}
        highlight = [False] * len(norm)
        for s, e in seg["spans"]:
            for orig in range(s, e):
                ni = inv.get(orig)
                if ni is not None:
                    highlight[ni] = True
        parts.append(norm)
        flags.extend(highlight)
        seg_ranges.append((cursor, cursor + len(norm)))
        seg_of_char.extend([str(seg["seq"])] * len(norm))
        cursor += len(norm)
    return "".join(parts), flags, seg_ranges, seg_of_char


def labels_for(ref_text: str, hyp_norm: str) -> list[int | None]:
    ref_norm, _ = conservative_norm(ref_text)
    ops = align_ops(ref_norm, hyp_norm)
    ops, _ = apply_filler_rules(ops, ref_norm, hyp_norm)
    labels: list[int | None] = [None] * len(hyp_norm)
    for op, ri, hi in ops:
        if hi is None:
            continue
        labels[hi] = 1 if op == "C" else 0
    return labels


def context(text: str, i: int, lo: int = 10, hi: int = 10) -> str:
    a, b = max(0, i - lo), min(len(text), i + hi + 1)
    return text[a:i] + "【" + text[i] + "】" + text[i + 1:b]


def main() -> None:
    # 用法：confidence_audit_bench.py [stats.json] [输出.json]
    # 不带参数时读 DB 里既定 run 的入库结果；带参数时直接读 stats.json。
    cli = sys.argv[1:]
    if cli and cli[0].startswith("run_"):
        segments = load_segments(DB_PATH, cli[0])
        run_label = cli[0]
    elif cli:
        stats_path = Path(cli[0])
        segments = load_segments_from_stats(stats_path)
        run_label = stats_path.name
    else:
        segments = load_segments(DB_PATH, RUN_ID)
        run_label = RUN_ID
    out_path = Path(cli[1]) if len(cli) > 1 else OUT_PATH
    ref_text = REF_PATH.read_text(encoding="utf-8")
    hyp_norm, flags, seg_ranges, seg_of_char = build_hyp(segments)
    labels = labels_for(ref_text, hyp_norm)

    n = len(hyp_norm)
    labelled = [(i, flags[i], labels[i]) for i in range(n) if labels[i] is not None]
    errs = [(i, f) for i, f, l in labelled if l == 0]
    marked = [(i, l) for i, f, l in labelled if f]
    tp = [(i) for i, f, l in labelled if f and l == 0]
    fp = [(i) for i, f, l in labelled if f and l == 1]
    fn = [(i) for i, f, l in labelled if (not f) and l == 0]
    tn = [(i) for i, f, l in labelled if (not f) and l == 1]

    precision = len(tp) / len(marked) if marked else 0.0
    recall = len(tp) / len(errs) if errs else 0.0

    print("=" * 72)
    print("置信度标亮对账（bench_10min vs LiPS 参考）")
    print("=" * 72)
    print(f"归一化字符数       : {len(labelled)}（未计入 None={n - len(labelled)}）")
    print(f"真实错误字符数     : {len(errs)}  错误率 {len(errs)/max(len(labelled),1):.2%}")
    print(f"标亮字符数         : {len(marked)}")
    print()
    print(f"命中(TP)           : {len(tp)}")
    print(f"误报(FP)           : {len(fp)}")
    print(f"漏报(FN)           : {len(fn)}")
    print(f"正确且未标(TN)     : {len(tn)}")
    print()
    print(f"精确率 precision   : {precision:.1%}  （标亮里有多少是真错）")
    print(f"召回率 recall      : {recall:.1%}  （真错里标出了多少）")
    # 按段
    print()
    print("按段统计：")
    for (a, b) in seg_ranges:
        idxs = [i for i in range(a, b) if labels[i] is not None]
        if not idxs:
            continue
        seg_err = sum(1 for i in idxs if labels[i] == 0)
        seg_mark = sum(1 for i in idxs if flags[i])
        seg_tp = sum(1 for i in idxs if flags[i] and labels[i] == 0)
        print(f"  段{seg_of_char[a]:>2}: 字{len(idxs):>4} 错{seg_err:>3} 标{seg_mark:>3} 命中{seg_tp:>3}")

    print()
    print("--- 误报样例（标亮但正确，前 25）---")
    for i in fp[:25]:
        print(f"  [{i:>4}] {context(hyp_norm, i)}")
    print()
    print("--- 漏报样例（真错但未标，前 25）---")
    for i in fn[:25]:
        print(f"  [{i:>4}] {context(hyp_norm, i)}")
    print()
    print("--- 命中样例（前 25）---")
    for i in tp[:25]:
        print(f"  [{i:>4}] {context(hyp_norm, i)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "run_id": run_label,
        "labelled_chars": len(labelled),
        "error_chars": len(errs),
        "marked_chars": len(marked),
        "tp": len(tp), "fp": len(fp), "fn": len(fn), "tn": len(tn),
        "precision": precision, "recall": recall,
        "fp_samples": [{"idx": i, "ctx": context(hyp_norm, i)} for i in fp[:100]],
        "fn_samples": [{"idx": i, "ctx": context(hyp_norm, i)} for i in fn[:100]],
        "tp_samples": [{"idx": i, "ctx": context(hyp_norm, i)} for i in tp[:100]],
        "per_segment": [
            {"seg": seg_of_char[a], "chars": sum(1 for i in range(a, b) if labels[i] is not None),
             "errors": sum(1 for i in range(a, b) if labels[i] == 0),
             "marked": sum(1 for i in range(a, b) if flags[i]),
             "tp": sum(1 for i in range(a, b) if flags[i] and labels[i] == 0)}
            for (a, b) in seg_ranges
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"报告已写入: {OUT_PATH}")


if __name__ == "__main__":
    main()