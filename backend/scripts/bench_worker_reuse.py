#!/usr/bin/env python
"""基准：常驻 ASR worker 跨任务复用已加载模型的收益。

用法（backend 目录）::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/bench_worker_reuse.py \
        --audio ../data/projects/prj_xxx/audio/standardized.wav \
        --model ../models/Qwen3-ASR-0.6B --tasks 2

逐任务打印墙钟耗时与 ``asr_model_loaded`` 事件里的加载秒数：
第一个任务是冷启动，第二个起应命中缓存（加载 ≈ 0s）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND / "src"))

from local_asr.orchestration.asr_worker import AsrWorkerExecutor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="常驻 worker 复用基准")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--out-dir", default="output/bench_worker")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-chunk", type=float, default=60.0)
    parser.add_argument("--models-dir", default="../models")
    args = parser.parse_args()

    audio = Path(args.audio).resolve()
    model = Path(args.model).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = Path(args.models_dir).resolve()

    executor = AsrWorkerExecutor(Path(__file__).with_name("qwen3_asr_worker.py"))
    loaded: list[float] = []
    try:
        for i in range(args.tasks):
            events: list[dict] = []
            argv = [
                # 与 runner._asr_argv 一致：首元素是 ASR 脚本路径
                str(Path(__file__).with_name("qwen3_asr_transcribe.py")),
                "--model", str(model),
                "--audio", str(audio),
                "--output", str(out_dir / f"raw_{i}.txt"),
                "--device", args.device,
                "--segmentation", "vad",
                "--batch-size", str(args.batch_size),
                "--batch-max", str(args.batch_size),
                "--target-chunk", str(args.target_chunk),
                "--chunk-strategy", "balanced",
                "--overlap", "1.5",
                "--vad-model", str(models / "funasr-fsmn-vad"),
                "--forced-aligner", str(models / "Qwen3-ForcedAligner-0.6B"),
            ]
            started = time.perf_counter()
            result = executor(sys.executable, argv, on_event=events.append)
            wall = time.perf_counter() - started
            load = next((e.get("load_seconds") for e in events
                         if e.get("kind") == "asr_model_loaded"), None)
            loaded.append(load or 0.0)
            print(json.dumps({
                "task": i,
                "ok": result.ok,
                "returncode": result.returncode,
                "wall_seconds": round(wall, 2),
                "model_load_seconds": load,
            }, ensure_ascii=False), flush=True)
            if not result.ok:
                print(result.stderr_tail[-2000:], file=sys.stderr, flush=True)
                return 1
    finally:
        executor.close()
    if len(loaded) > 1:
        saved = loaded[0] - sum(loaded[1:]) / (len(loaded) - 1)
        print(f"[bench] 第二次起平均省模型加载 {saved:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())