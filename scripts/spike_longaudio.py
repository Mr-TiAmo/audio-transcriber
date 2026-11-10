# Windows Spike：长音频边界测试（V3.0 附录 B 待答项 1补充 / 4 / 6 / 7）
#
# 测试内容（拼接 LiPS 音频构造长音频，无静音边界的严苛场景）：
#   A. 单次推理上限：60s/120s/300s/600s/1200s 整段直推，观测空转写/截断/OOM（待答项 6，MPS 基线 300s）
#   B. 60s 等长切块 + batch 4 全量识别：长音频 RTF 与端到端稳定性（待答项 4 / 7）
#   C. batch 上限：60s 等长块 batch 8 / 16，观测显存与加速比（待答项 1 补充）
#
# 用法：backend\.venv\Scripts\python.exe scripts\spike_longaudio.py [拼接总秒数，默认1200]
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen3-ASR-0.6B"
LIPS_DIR = Path(r"D:\work\workspace\local-asr-portable\datasets\chinese-lips\processed_test")
SR = 16000


def build_long_audio(total_sec: int) -> np.ndarray:
    """拼接 LiPS 音频到指定总时长（直接拼接，无静音插入，严苛边界场景）。"""
    chunks = []
    acc = 0
    for w in sorted(LIPS_DIR.glob("*.wav")):
        data, sr = sf.read(w, dtype="float32")
        if sr != SR:
            continue
        chunks.append(data)
        acc += len(data) / SR
        if acc >= total_sec:
            break
    audio = np.concatenate(chunks)
    return audio[: total_sec * SR]


def vram_reset():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def vram_peak() -> float:
    return torch.cuda.max_memory_reserved() / 1024**3


def main() -> int:
    total_sec = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    audio = build_long_audio(total_sec)
    print(f"[构造] 长音频: {len(audio)/SR:.0f}s（LiPS 拼接，无静音）")

    from qwen_asr import Qwen3ASRModel

    asr = Qwen3ASRModel.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, device_map="cuda", max_new_tokens=2048
    )
    process = psutil.Process()

    # A. 单次推理上限（待答项 6）
    print("\n=== A. 单次推理上限（整段直推） ===")
    for seg_sec in (60, 120, 300, 600, 1200):
        seg = audio[: seg_sec * SR]
        vram_reset()
        t0 = time.perf_counter()
        try:
            out = asr.transcribe([(seg, SR)], language="Chinese")
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            text = out[0].text
            status = "空转写!" if not text.strip() else ("疑似截断" if len(text) < seg_sec * 2 else "正常")
            print(
                f"  {seg_sec:>5}s 段: {ms:>7.0f} ms, 输出 {len(text):>4} 字 [{status}], "
                f"显存峰值 {vram_peak():.2f} GB"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  {seg_sec:>5}s 段: OOM!")
            vram_reset()

    # B. 60s 等长切块 + batch 4（待答项 4 / 7）
    print("\n=== B. 60s 等长切块 + batch 4 ===")
    vram_reset()
    chunks = [(audio[i * 60 * SR : (i + 1) * 60 * SR], SR) for i in range(len(audio) // (60 * SR))]
    t0 = time.perf_counter()
    outs = []
    for i in range(0, len(chunks), 4):
        outs.extend(asr.transcribe(chunks[i : i + 4], language="Chinese"))
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    empty = sum(1 for o in outs if not o.text.strip())
    chars = sum(len(o.text) for o in outs)
    print(
        f"  {len(chunks)} 块 x 60s, batch 4: {wall:.1f}s / 音频 {len(audio)/SR:.0f}s, "
        f"RTF={wall/(len(audio)/SR):.3f}"
    )
    print(f"  输出总字数 {chars}, 空转写块 {empty}, 显存峰值 {vram_peak():.2f} GB")

    # C. batch 上限（待答项 1 补充）
    print("\n=== C. batch 上限（60s 等长块） ===")
    base_chunks = chunks[:16]
    for bs in (4, 8, 16):
        vram_reset()
        t0 = time.perf_counter()
        for i in range(0, len(base_chunks), bs):
            asr.transcribe(base_chunks[i : i + bs], language="Chinese")
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 16 / len(base_chunks)  # 归一化到 16 块
        speech = 60 * len(base_chunks)
        print(
            f"  batch {bs:>2}: 16 块折算 {wall:.1f}s, RTF={wall/speech:.3f}, "
            f"显存峰值 {vram_peak():.2f} GB"
        )

    print(f"\n[资源] 进程 RSS: {process.memory_info().rss/1024**2:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
