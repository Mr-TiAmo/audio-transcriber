#!/usr/bin/env python3
"""从 Chinese-LiPS 测试集随机组装多段固定时长评测音频。

与 ``build_lips_eval.py`` 的区别：不取单一说话人的连续段，而是打乱全部
语句后随机抽取，说话人、话题和语速分布更接近真实录音；每条语句之间
插入固定静音，避免波形硬切导致吞字。

用法::

    python scripts/build_random_bench.py \
        --count 2 --minutes 5 --seed 20261107 \
        --out-dir ../data/bench/random5m

产物：
    random5m_1.wav / random5m_1.txt / ... + manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import wave
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets/chinese-lips"
GAP_MS = 300  # 语句间静音，降低拼接边界的吞字/粘连


def load_meta(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["ID"]: r.get("TEXT", "") for r in csv.DictReader(f) if r.get("ID")}


def scan(wav_dir: Path, meta: dict[str, str]) -> list[dict]:
    items = []
    for path in sorted(wav_dir.glob("*.wav")):
        with wave.open(str(path)) as w:
            dur = w.getnframes() / w.getframerate()
        items.append({
            "id": path.stem,
            "path": path,
            "seconds": dur,
            "text": meta.get(path.stem, ""),
            "speaker": path.stem.rsplit("_", 1)[0],
        })
    return items


def slice_until(items: list[dict], pos: int, target_s: float,
                gap_ms: int) -> tuple[list[dict], int]:
    """从 pos 起顺序取语句直到累计时长达到 target_s（含语句间静音）。"""
    picked, total = [], 0.0
    while pos < len(items) and total < target_s:
        item = items[pos]
        picked.append(item)
        total += item["seconds"] + gap_ms / 1000.0
        pos += 1
    return picked, pos


def write_concat(picked: list[dict], output: Path, gap_ms: int) -> None:
    with wave.open(str(picked[0]["path"])) as first:
        channels = first.getnchannels()
        width = first.getsampwidth()
        rate = first.getframerate()
    gap = b"\x00" * (width * channels * rate * GAP_MS // 1000)
    with wave.open(str(output), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        for i, item in enumerate(picked):
            with wave.open(str(item["path"])) as w:
                out.writeframes(w.readframes(w.getnframes()))
            if i + 1 < len(picked):
                out.writeframes(gap)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2, help="生成多少段音频")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/bench/random5m")
    ap.add_argument("--gap-ms", type=int, default=GAP_MS)
    args = ap.parse_args()

    meta = load_meta(DATASET / "meta_test.csv")
    items = scan(DATASET / "processed_test", meta)
    missing = sum(1 for x in items if not x["text"])
    items = [x for x in items if x["text"]]
    print(f"扫描 {len(items)} 条语句，{len({x['speaker'] for x in items})} 个说话人"
          f"（缺文本 {missing} 条已跳过）")

    random.Random(args.seed).shuffle(items)
    target_s = args.minutes * 60.0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"seed": args.seed, "minutes": args.minutes,
                "gap_ms": args.gap_ms, "created": datetime.now().isoformat(),
                "targets": {}}
    pos = 0
    for n in range(1, args.count + 1):
        if pos >= len(items):
            print(f"⚠ 语句不足，只生成 {n - 1} 段")
            break
        picked, pos = slice_until(items, pos, target_s, args.gap_ms)
        total = sum(x["seconds"] for x in picked) + args.gap_ms / 1000.0 * (len(picked) - 1)
        wav_out = args.out_dir / f"random{n}.wav"
        txt_out = args.out_dir / f"random{n}.txt"
        write_concat(picked, wav_out, args.gap_ms)
        txt_out.write_text("\n".join(x["text"] for x in picked), encoding="utf-8")

        manifest["targets"][f"random{n}"] = {
            "wav": str(wav_out),
            "ref_text": str(txt_out),
            "segments": len(picked),
            "seconds": round(total, 2),
            "speakers": sorted({x["speaker"] for x in picked}),
            "ids": [x["id"] for x in picked],
        }
        dur = wave.open(str(wav_out)).getnframes() / 16000
        print(f"random{n}: {len(picked)} 条，{dur:.1f}s，"
              f"{len({x['speaker'] for x in picked})} 个说话人 → {wav_out}")

    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())