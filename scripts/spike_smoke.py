# Windows Spike 冒烟测试（V3.0 附录 B 待答项 1、5 的初步数据）
# 验证：Qwen3-ASR-0.6B 在 RTX 4060 CUDA 上的模型加载、单条推理、batch 4 推理
# 输出：model_load_ms / 单条与 batch RTF / 加速比 / 峰值显存 / 峰值 RSS
#
# 用法：backend\.venv\Scripts\python.exe scripts\spike_smoke.py [测试音频目录]
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen3-ASR-0.6B"
DEFAULT_AUDIO_DIR = Path(r"D:\work\workspace\local-asr-portable\datasets\chinese-lips\processed_test")


def main() -> int:
    audio_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AUDIO_DIR
    wavs = sorted(audio_dir.glob("*.wav"))[:4]
    if len(wavs) < 4:
        print(f"[错误] 测试音频不足 4 条: {audio_dir}")
        return 1

    # 读取 LiPS 音频（已是 16kHz mono PCM16）
    audios = []
    total_sec = 0.0
    for w in wavs:
        data, sr = sf.read(w, dtype="float32")
        audios.append((data, sr))
        total_sec += len(data) / sr
    print(f"[输入] {len(wavs)} 条音频, 总时长 {total_sec:.1f}s, 采样率 {sr}Hz")
    for w, (d, s) in zip(wavs, audios):
        print(f"  - {w.name}: {len(d)/s:.1f}s")

    process = psutil.Process()
    rss_before = process.memory_info().rss / 1024**2

    # 1. 模型加载（冷启动计时，Spike 必答项）
    from qwen_asr import Qwen3ASRModel

    print("\n[加载] Qwen3-ASR-0.6B -> CUDA bf16 ...")
    t0 = time.perf_counter()
    asr = Qwen3ASRModel.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - t0) * 1000
    rss_after_load = process.memory_info().rss / 1024**2
    print(f"  模型加载耗时: {model_load_ms:.0f} ms（冷启动）")

    results = {"model_load_ms": round(model_load_ms), "audio_total_sec": round(total_sec, 1)}

    # 2. 单条推理（含 CUDA 预热）
    print("\n[推理] 单条（首次，含 CUDA kernel 预热）...")
    t0 = time.perf_counter()
    out = asr.transcribe([audios[0]], language="Chinese")
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - t0) * 1000
    dur0 = len(audios[0][0]) / audios[0][1]
    print(f"  首次推理: {first_ms:.0f} ms（音频 {dur0:.1f}s）")
    print(f"  转写文本: {out[0].text[:60]}")

    # 稳态单条
    single_times = []
    for (data, s) in audios:
        t0 = time.perf_counter()
        asr.transcribe([(data, s)], language="Chinese")
        torch.cuda.synchronize()
        single_times.append(time.perf_counter() - t0)
    single_total = sum(single_times)
    print(f"\n[推理] 单条稳态: 总 {single_total:.2f}s / 音频 {total_sec:.1f}s, RTF={single_total/total_sec:.3f}")

    # 3. batch 4 推理（等长块，Spike 待答项 5：CUDA batch 加速比）
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out_batch = asr.transcribe(audios, language="Chinese")
    torch.cuda.synchronize()
    batch_sec = time.perf_counter() - t0
    print(f"\n[推理] batch 4: {batch_sec:.2f}s / 音频 {total_sec:.1f}s, RTF={batch_sec/total_sec:.3f}")
    print(f"  加速比（单条串行/batch）: {single_total/batch_sec:.2f}x")
    for o in out_batch:
        print(f"  - {o.text[:50]}")

    # 4. 资源峰值
    vram_reserved = torch.cuda.max_memory_reserved() / 1024**3
    vram_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_rss = process.memory_info().rss / 1024**2
    print(f"\n[资源] 峰值显存: reserved {vram_reserved:.2f} GB / allocated {vram_allocated:.2f} GB")
    print(f"[资源] 峰值 RSS: {peak_rss:.0f} MB（加载前 {rss_before:.0f} MB）")

    results.update(
        {
            "first_inference_ms": round(first_ms),
            "single_rtf": round(single_total / total_sec, 3),
            "batch4_rtf": round(batch_sec / total_sec, 3),
            "batch_speedup": round(single_total / batch_sec, 2),
            "vram_reserved_gb": round(vram_reserved, 2),
            "vram_allocated_gb": round(vram_allocated, 2),
            "peak_rss_mb": round(peak_rss),
        }
    )
    print("\n=== Spike 冒烟结果（JSON）===")
    import json

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
