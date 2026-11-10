# Windows Spike：CER 基线测试（LiPS 测试集 + 人工参考文本）
# 复用源项目 scripts/eval_cer.py 的中文数字归一化与编辑距离实现，保证与 macOS 测评口径一致
#
# 用法：backend\.venv\Scripts\python.exe scripts\spike_cer.py [样本数]
import csv
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen3-ASR-0.6B"
LIPS_DIR = Path(r"D:\work\workspace\local-asr-portable\datasets\chinese-lips")
EVAL_CER = Path(r"D:\work\workspace\local-asr-portable\scripts")

sys.path.insert(0, str(EVAL_CER))
from eval_cer import edit_stats, normalize  # noqa: E402


def main() -> int:
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    # 读取参考文本
    samples = []
    with open(LIPS_DIR / "meta_test.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            wav = LIPS_DIR / "processed_test" / f"{row['ID']}.wav"
            if wav.exists() and row["TEXT"].strip():
                samples.append((wav, row["TEXT"].strip()))
            if len(samples) >= n_samples:
                break
    print(f"[输入] {len(samples)} 条 LiPS 样本（有参考文本）")

    # 加载模型
    from qwen_asr import Qwen3ASRModel

    asr = Qwen3ASRModel.from_pretrained(str(MODEL_DIR), dtype=torch.bfloat16, device_map="cuda")

    # 逐条转写（batch 4）
    audios = []
    refs = []
    for wav, text in samples:
        data, sr = sf.read(wav, dtype="float32")
        audios.append((data, sr))
        refs.append(text)

    t0 = time.perf_counter()
    outs = []
    B = 4
    for i in range(0, len(audios), B):
        outs.extend(asr.transcribe(audios[i : i + B], language="Chinese"))
    elapsed = time.perf_counter() - t0
    total_sec = sum(len(d) / s for d, s in audios)
    print(f"[推理] batch 4: {elapsed:.1f}s / 音频 {total_sec:.0f}s (RTF={elapsed/total_sec:.3f})")

    # 计算 CER（与 macOS 测评同口径：数字归一化 + 去标点/口癖）
    total_err = total_ref = 0
    worst = []
    for ref, out in zip(refs, outs):
        r, h = normalize(ref), normalize(out.text)
        _, _, _, dist = edit_stats(r, h)  # 返回 (替换, 删除, 插入, 编辑距离)
        total_err += dist
        total_ref += len(r)
        worst.append((dist / max(len(r), 1), ref, out.text))
    cer = total_err / total_ref * 100
    print(f"\n[CER] Qwen3-ASR-0.6B @ RTX 4060: {cer:.2f}%（{total_err} 错 / {total_ref} 字）")
    print(f"[对照] macOS MPS 5 分钟财经 CER: 5.37%（不同语料，仅供参考）")

    worst.sort(reverse=True)
    print("\n[最差 5 条]")
    for rate, ref, hyp in worst[:5]:
        print(f"  CER {rate*100:.1f}% | 参考: {ref[:40]}")
        print(f"            | 识别: {normalize(hyp)[:40]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
