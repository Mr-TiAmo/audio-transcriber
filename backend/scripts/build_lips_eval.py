#!/usr/bin/env python3
"""从 Chinese-LiPS 测试集合成 5/10/20 分钟单人连续讲解评测音频。

策略：用不同说话人的前 N 个连续段接近目标时长，偏差 <1s。

需要先下载并解压 processed_test.zip 到 datasets/chinese-lips/processed_test/，
并将 meta_test.csv 放在 datasets/chinese-lips/。

用法（项目根目录）:
    .venv/bin/python scripts/build_lips_eval.py

产物:
    tests/chinese-lips/lips_5min.wav  + .txt
    tests/chinese-lips/lips_10min.wav + .txt
    tests/chinese-lips/lips_20min.wav + .txt
    tests/chinese-lips/manifest.json
"""

from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WAV_DIR = ROOT / "datasets/chinese-lips/processed_test"
META_CSV = ROOT / "datasets/chinese-lips/meta_test.csv"
OUT_DIR = ROOT / "tests/chinese-lips"
TARGETS = (5 * 60, 10 * 60, 20 * 60)  # 秒


def load_meta() -> dict[str, str]:
    """ID → TEXT，自动处理 UTF-8 BOM。"""
    for enc in ("utf-8", "utf-8-sig"):
        try:
            with open(META_CSV, newline="", encoding=enc) as f:
                rows = list(csv.DictReader(f))
            return {r["ID"]: r["TEXT"] for r in rows if r.get("ID")}
        except (UnicodeDecodeError, csv.Error):
            continue
    raise RuntimeError(f"无法解析 {META_CSV}")


def scan_wavs() -> dict[str, list[dict]]:
    """返回 {speaker_prefix: [seg_dict_sorted_by_seg], ...}。"""
    meta = load_meta()
    by_spk: dict[str, list[dict]] = {}
    for f in sorted(WAV_DIR.glob("*.wav")):
        stem = f.stem
        parts = stem.rsplit("_", 1)
        spk = parts[0]
        seg = int(parts[1]) if len(parts) == 2 else 0
        wf = wave.open(str(f))
        frames, rate = wf.getnframes(), wf.getframerate()
        dur = frames / rate
        wf.close()
        by_spk.setdefault(spk, []).append({
            "id": stem, "path": str(f), "duration": dur,
            "speaker": spk, "segment": seg,
            "text": meta.get(stem, ""), "frames": frames, "rate": rate,
        })
    for segs in by_spk.values():
        segs.sort(key=lambda s: s["segment"])
    return by_spk


def pick_first_n(segs: list[dict], target_sec: float) -> tuple[int, float]:
    """返回 (前N段段数, 总时长) 使得总时长最接近 target_sec 且 ≥ target_sec。"""
    total, best_n, best_total, best_diff = 0.0, 0, 0.0, float("inf")
    for i, s in enumerate(segs):
        total += s["duration"]
        if total >= target_sec:
            diff = total - target_sec
            if diff < best_diff:
                best_n, best_total, best_diff = i + 1, total, diff
        # 最多超出 60s
        if total > target_sec + 60:
            break
    return best_n, best_total


def concat_wavs(segments: list[dict], output: Path) -> None:
    """PCM 直接拼接写入新 WAV。"""
    with wave.open(str(output), "wb") as out:
        first = wave.open(segments[0]["path"])
        out.setnchannels(first.getnchannels())
        out.setsampwidth(first.getsampwidth())
        out.setframerate(first.getframerate())
        first.close()
        total_frames = 0
        for s in segments:
            wf = wave.open(s["path"])
            data = wf.readframes(wf.getnframes())
            out.writeframes(data)
            total_frames += wf.getnframes()
            wf.close()
    wf = wave.open(str(output))
    assert wf.getnframes() == total_frames, f"{output}: 帧数不匹配"
    wf.close()


def main() -> int:
    by_spk = scan_wavs()
    total_segs = sum(len(v) for v in by_spk.values())
    print(f"扫描 {total_segs} 段，{len(by_spk)} 个说话人")

    # 对每个目标，找最接近的 (说话人, 前N段)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "targets": {},
        "build_date": __import__("datetime").datetime.now().isoformat(),
    }
    used_speakers: set[str] = set()

    for target_sec in sorted(TARGETS):  # 先短后长，短音频优先用时长更接近的说话人
        label = f"{target_sec // 60}min"

        best_spk, best_n, best_total, best_diff = "", 0, 0.0, float("inf")
        for spk, segs in sorted(by_spk.items()):
            if spk in used_speakers:
                continue
            n, total = pick_first_n(segs, target_sec)
            if n == 0:
                continue
            diff = abs(total - target_sec)
            if diff < best_diff:
                best_spk, best_n, best_total, best_diff = spk, n, total, diff

        if not best_spk:
            print(f"⚠ {label}: 未找到候选")
            continue

        used_speakers.add(best_spk)
        chosen = by_spk[best_spk][:best_n]

        wav_out = OUT_DIR / f"lips_{label}.wav"
        txt_out = OUT_DIR / f"lips_{label}.txt"
        concat_wavs(chosen, wav_out)
        ref = " ".join([s["text"] for s in chosen])
        txt_out.write_text(ref, encoding="utf-8")

        manifest["targets"][label] = {
            "speaker": best_spk,
            "segments": best_n,
            "seconds": round(best_total, 1),
            "deviation_s": round(best_diff, 1),
            "frame_rate": chosen[0]["rate"],
            "channels": 1,
            "segment_ids": [s["id"] for s in chosen],
            "first_seg": chosen[0]["segment"],
            "last_seg": chosen[-1]["segment"],
            "wav": str(wav_out.relative_to(ROOT)),
            "ref_text": str(txt_out.relative_to(ROOT)),
        }
        print(f"  {label}: {best_spk} 前 {best_n} 段 {best_total:.1f}s (偏 {best_diff:.1f}s)")
        print(f"        首段: {chosen[0]['text'][:60]}...")
        print(f"        末段: ...{chosen[-1]['text'][-60:]}")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n产物: {OUT_DIR}/")
    for f in sorted(OUT_DIR.glob("lips_*")):
        size_mb = f.stat().st_size / 1_048_576
        print(f"  {f.name}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())