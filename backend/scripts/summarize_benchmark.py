#!/usr/bin/env python3
"""汇总 output/bench/ 下各模型的转录结果与性能指标，生成对比表。

用法:
    ./.venv/bin/python scripts/summarize_benchmark.py \
        --ref output/测试音频_实际字幕版.txt \
        --bench-dir output/bench \
        --markdown output/最终测评报告.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_cer import edit_stats, normalize  # noqa: E402

# 展示顺序与名称
DISPLAY = {
    "qwen3-asr-1.7b": "Qwen3-ASR-1.7B（wetext ITN）",
    "qwen3-asr-0.6b": "Qwen3-ASR-0.6B（wetext ITN）",
    "turbo": "faster-whisper large-v3-turbo",
    "large-v3": "faster-whisper large-v3",
    "funasr-fun-asr-nano": "FunASR Fun-ASR-Nano-2512",
    "funasr-paraformer-zh": "FunASR paraformer-zh（SeACo）",
    "funasr-sensevoice": "FunASR SenseVoiceSmall（ITN）",
}


def load_stats(txt: Path) -> dict:
    p = txt.with_suffix(".stats.json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--bench-dir", default="output/bench")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    ref_raw = Path(args.ref).read_text(encoding="utf-8")
    ref = normalize(ref_raw)
    ref_n = len(ref)

    rows = []
    for txt in sorted(Path(args.bench_dir).glob("*.txt")):
        if txt.stem.endswith("_raw"):
            continue
        hyp_raw = txt.read_text(encoding="utf-8")
        hyp = normalize(hyp_raw)
        sub, dele, ins, dist = edit_stats(ref, hyp)
        st = load_stats(txt)
        rows.append({
            "key": txt.stem,
            "name": DISPLAY.get(txt.stem, txt.stem),
            "cer": dist / ref_n if ref_n else float("nan"),
            "s": sub, "d": dele, "i": ins,
            "chars": len(hyp),
            "raw_chars": len(hyp_raw),
            "load_s": st.get("load_seconds"),
            "infer_s": st.get("infer_seconds"),
            "first_s": st.get("first_segment_seconds"),
            "rtf": st.get("rtf"),
            "rss": st.get("peak_rss_mb"),
            "device": st.get("device", "?"),
            "compute": st.get("compute_type", ""),
        })

    order = list(DISPLAY)
    rows.sort(key=lambda r: (order.index(r["key"]) if r["key"] in order else 99, r["cer"]))

    lines = [
        f"参考文本：`{args.ref}`（归一化后 {ref_n} 字）",
        "",
        "| 模型 | CER | S / D / I | 推理 | RTF | 首段延迟 | 加载 | 峰值 RSS | 设备 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        dev = r["device"] + (f" {r['compute']}" if r["compute"] else "")
        first = f"{r['first_s']:.1f}s" if r["first_s"] is not None else "—"
        load = f"{r['load_s']:.1f}s" if r["load_s"] is not None else "—"
        infer = f"{r['infer_s']:.1f}s" if r["infer_s"] is not None else "—"
        rtf = f"{r['rtf']:.3f}" if r["rtf"] is not None else "—"
        rss = f"{r['rss']:.0f}MB" if r["rss"] is not None else "—"
        lines.append(
            f"| {r['name']} | {r['cer'] * 100:.2f}% | {r['s']} / {r['d']} / {r['i']} | "
            f"{infer} | {rtf} | {first} | {load} | {rss} | {dev} |"
        )

    table = "\n".join(lines)
    print(table)

    if args.markdown:
        p = Path(args.markdown)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(table + "\n", encoding="utf-8")
        print(f"\n已写入: {p}")

    best = min(rows, key=lambda r: r["cer"])
    fastest = min((r for r in rows if r["rtf"] is not None), key=lambda r: r["rtf"])
    print(f"\nCER 最低: {best['name']} {best['cer'] * 100:.2f}%")
    print(f"RTF 最低: {fastest['name']} {fastest['rtf']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())