"""汇总历史转写任务的阶段耗时（ASR / 后处理 / ITN），用于性能分析。

数据来源：
- `data/temp/<run_id>/asr/raw.stats.json`：load_seconds / infer_seconds / aux_seconds / rtf / batching
- `data/temp/<run_id>/postprocess/result_postprocess.json`：elapsed_seconds / itn_seconds
- `data/database/app.db`：项目名、音频时长、模型、墙钟起止

用法（在项目根目录）：
    python -I backend/scripts/bench_cost_summary.py [--limit 40] [--csv out.csv]

输出逐 run 明细 + 按「模型 × 音频时长档」聚合的中位数/均值。
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = ROOT / "data" / "temp"
DB_PATH = ROOT / "data" / "database" / "app.db"

BUCKETS = [
    ("<15s", 0, 15),
    ("15-60s", 15, 60),
    ("1-5min", 60, 300),
    ("5-10min", 300, 600),
    ("10-30min", 600, 1800),
    ("30-60min", 1800, 3600),
    (">=60min", 3600, 1e9),
]


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def collect(limit: int) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    runs = conn.execute(
        """
        SELECT r.id, r.model_name, r.status, r.started_at, r.finished_at,
               p.original_name, p.duration_ms
        FROM transcription_runs r JOIN projects p ON p.id = r.project_id
        ORDER BY r.started_at DESC
        """
    ).fetchall()

    rows: list[dict] = []
    for r in runs:
        if r["status"] != "COMPLETED":
            continue
        run_dir = TEMP_DIR / r["id"]
        stats = load_json(run_dir / "asr" / "raw.stats.json")
        post = load_json(run_dir / "postprocess" / "result_postprocess.json")
        if not stats or not post:
            continue
        started, finished = parse_ts(r["started_at"]), parse_ts(r["finished_at"])
        wall = (finished - started).total_seconds() if started and finished else None
        cross = (stats.get("low_conf") or {}).get("cross_check") or {}
        rows.append({
            "run_id": r["id"],
            "project": r["original_name"],
            "model": r["model_name"],
            "audio_s": stats.get("audio_seconds") or (r["duration_ms"] or 0) / 1000,
            "wall_s": wall,
            "asr_s": stats.get("elapsed_seconds"),
            "load_s": stats.get("load_seconds"),
            "infer_s": stats.get("infer_seconds"),
            "aux_s": stats.get("aux_seconds"),
            "rtf": stats.get("rtf"),
            "chunks": (stats.get("segmentation") or {}).get("chunks"),
            "batch": (stats.get("batching") or {}).get("applied_size"),
            "post_s": post.get("elapsed_seconds"),
            "itn_s": post.get("itn_seconds"),
            "cross_s": cross.get("seconds"),
            "marked": (stats.get("low_conf") or {}).get("marked_chars"),
        })
        if len(rows) >= limit:
            break
    return rows


def fmt(v, digits=1) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def bucket_of(seconds: float) -> str:
    for label, lo, hi in BUCKETS:
        if lo <= seconds < hi:
            return label
    return "?"


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["model"], bucket_of(row["audio_s"]))
        groups.setdefault(key, []).append(row)
    out = []
    for (model, bucket), items in sorted(groups.items()):
        def med(key):
            vals = [x[key] for x in items if x.get(key) is not None]
            return statistics.median(vals) if vals else None
        out.append({
            "model": model, "bucket": bucket, "n": len(items),
            "audio_s": med("audio_s"), "wall_s": med("wall_s"), "asr_s": med("asr_s"),
            "load_s": med("load_s"), "infer_s": med("infer_s"), "aux_s": med("aux_s"),
            "rtf": med("rtf"), "post_s": med("post_s"), "itn_s": med("itn_s"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    rows = collect(args.limit)
    if not rows:
        print("没有可用的完成 run（检查 data/temp 与数据库）", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: r["audio_s"])

    print("== 逐 run 明细（按音频时长排序）")
    header = ("run_id", "model", "audio_s", "wall_s", "asr_s", "load_s", "infer_s",
              "aux_s", "rtf", "chunks", "batch", "post_s", "itn_s", "cross_s")
    print("  ".join(header))
    for r in rows:
        print("  ".join([
            r["run_id"], r["model"].replace("Qwen3-ASR-", ""), fmt(r["audio_s"]),
            fmt(r["wall_s"]), fmt(r["asr_s"]), fmt(r["load_s"]), fmt(r["infer_s"]),
            fmt(r["aux_s"]), fmt(r["rtf"], 3), fmt(r["chunks"]), fmt(r["batch"]),
            fmt(r["post_s"]), fmt(r["itn_s"]), fmt(r["cross_s"]),
        ]))

    print("\n== 分组中位数（模型 × 时长档）")
    agg = aggregate(rows)
    print("  ".join(("model", "bucket", "n", "audio_s", "wall_s", "asr_s",
                    "load_s", "infer_s", "aux_s", "rtf", "post_s", "itn_s")))
    for a in agg:
        print("  ".join([
            a["model"].replace("Qwen3-ASR-", ""), a["bucket"], str(a["n"]),
            fmt(a["audio_s"]), fmt(a["wall_s"]), fmt(a["asr_s"]), fmt(a["load_s"]),
            fmt(a["infer_s"]), fmt(a["aux_s"]), fmt(a["rtf"], 3),
            fmt(a["post_s"]), fmt(a["itn_s"]),
        ]))

    if args.csv:
        with args.csv.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n已写入 {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())