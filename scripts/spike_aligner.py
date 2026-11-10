# Windows Spike：Qwen3-ForcedAligner 中文对齐质量测试（V3.0 附录 B 待答项 2）
# 决策依据：对齐可用 -> 时间戳升级为字符级方案；不可用 -> 回退 VAD 级片段时间戳
#
# 校验项（V3.0 §5.5 硬性规则）：
#   1. aligned_tokens 时间在音频时长范围内
#   2. start_ms <= end_ms，序列单调不回退
#   3. 单元类型（cjk_char / word）
#   4. 对齐文本与转写文本可对应
#   5. 单次对齐耗时与显存
#
# 用法：backend\.venv\Scripts\python.exe scripts\spike_aligner.py
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
ASR_DIR = ROOT / "models" / "Qwen3-ASR-0.6B"
ALIGNER_DIR = ROOT / "models" / "Qwen3-ForcedAligner-0.6B"
LIPS_DIR = Path(r"D:\work\workspace\local-asr-portable\datasets\chinese-lips\processed_test")


def check_tokens(tokens, duration_ms) -> list[str]:
    """按 V3.0 §5.5 硬性规则校验时间戳（ForcedAlignItem: start_time/end_time 秒），返回问题列表。"""
    issues = []
    prev_end = -1.0
    for t in tokens:
        # ForcedAlignItem 属性访问（秒）；兼容 dict 形式
        if isinstance(t, dict):
            s = t.get("start_time", t.get("startMs", -1))
            e = t.get("end_time", t.get("endMs", -1))
        else:
            s = getattr(t, "start_time", None)
            e = getattr(t, "end_time", None)
        if s is None or e is None:
            issues.append("缺失时间字段")
            continue
        s, e = float(s), float(e)  # 秒
        if s > e:
            issues.append(f"起止逆序 {s}>{e}")
        if s < 0 or e * 1000 > duration_ms + 100:  # 容差 100ms
            issues.append(f"越界 [{s},{e}s] > {duration_ms/1000:.1f}s")
        if s < prev_end:
            issues.append(f"时间回退 s={s} < prev_end={prev_end}")
        prev_end = max(prev_end, e)
    return issues


def main() -> int:
    wavs = sorted(LIPS_DIR.glob("*.wav"))[:3]
    audios = []
    for w in wavs:
        data, sr = sf.read(w, dtype="float32")
        audios.append((data, sr))

    from qwen_asr import Qwen3ASRModel

    print("[加载] Qwen3-ASR-0.6B + Qwen3-ForcedAligner-0.6B -> CUDA bf16 ...")
    t0 = time.perf_counter()
    asr = Qwen3ASRModel.from_pretrained(
        str(ASR_DIR),
        forced_aligner=str(ALIGNER_DIR),
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda"},
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  加载耗时: {load_ms:.0f} ms（含两个模型）")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    outs = asr.transcribe(audios, language="Chinese", return_time_stamps=True)
    torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000

    print(f"\n[推理] 3 条音频（ASR + 对齐）: {infer_ms:.0f} ms")
    print(f"[显存] 峰值 reserved: {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")

    ok = True
    for w, o, (data, sr) in zip(wavs, outs, audios):
        dur_ms = len(data) / sr * 1000
        print(f"\n=== {w.name}（{dur_ms/1000:.1f}s）===")
        print(f"  文本: {o.text[:60]}")
        # 时间戳字段探测
        ts = getattr(o, "time_stamps", None) or getattr(o, "timestamps", None)
        if ts is None:
            for attr in vars(o):
                if "time" in attr.lower() or "align" in attr.lower():
                    print(f"  发现字段: {attr} = {str(getattr(o, attr))[:200]}")
            print("  [未找到时间戳字段] 字段列表: {[a for a in vars(o)]}")
            ok = False
            continue
        print(f"  时间戳单元数: {len(ts)}")
        for t in ts[:5]:
            print(f"    {t}")
        issues = check_tokens(ts, dur_ms)
        if issues:
            ok = False
            print(f"  [问题] {issues[:5]}")
        else:
            print(f"  [校验] 时间戳合规（范围内/有序/起止正常）")

    print(f"\n=== 结论: {'对齐质量初步可用 ✓' if ok else '需要进一步排查'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
