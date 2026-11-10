# 探针：确认 qwen-asr 0.0.6 下 beam-4 / 采样 的输出结构与 beam_indices 语义。
# 只用一段 20s 音频，验证 scores 形状、父 beam 索引、local logp 与路径分数是否吻合。
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from qwen3_asr_transcribe import _prepare_asr_inputs, load_model  # noqa: E402


def slice_wav(src: Path, dst: Path, start_s: float, end_s: float) -> None:
    with wave.open(str(src), "rb") as r:
        assert r.getframerate() == 16000 and r.getnchannels() == 1
        start = int(start_s * 16000)
        end = int(end_s * 16000)
        r.setpos(start)
        data = r.readframes(end - start)
        with wave.open(str(dst), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(data)


def main() -> None:
    stats = json.loads((ROOT.parent / "data/bench/analysis4/bench_10min_raw.stats.json")
                       .read_text(encoding="utf-8"))
    seg = stats["segments"][0]
    out_dir = ROOT / "output" / "confidence" / "p1p2"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "probe_seg0.wav"
    slice_wav(ROOT.parent / "data/bench/bench_10min.wav", wav, seg["start_s"], seg["end_s"])

    processor, model = load_model(str(ROOT.parent / "models/Qwen3-ASR-0.6B"), "cuda")
    texts, wavs = _prepare_asr_inputs(processor, [str(wav)], None, "Chinese")
    inputs = processor(text=texts, audio=wavs, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    input_len = inputs["input_ids"].shape[1]

    print("=== greedy ===")
    with torch.no_grad():
        g = model.generate(**inputs, max_new_tokens=512, output_scores=True)
    gseq = g.sequences[:, input_len:]
    print("greedy tokens", gseq.shape, "scores", len(g.scores), g.scores[0].shape)
    print("greedy text", processor.batch_decode(
        gseq, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][:60])

    print("=== beam 4 ===")
    with torch.no_grad():
        b = model.generate(**inputs, max_new_tokens=512, num_beams=4,
                           num_return_sequences=4, do_sample=False,
                           output_scores=True)
    print("input_len", input_len)
    print("seqs", b.sequences.shape, "scores", len(b.scores), b.scores[0].shape)
    print("beam_indices", None if b.beam_indices is None else b.beam_indices.shape)
    bseq = b.sequences[:, input_len:]
    print("bseq len", bseq.shape)
    for i in range(bseq.shape[0]):
        row = bseq[i].tolist()
        eos_pos = [j for j, x in enumerate(row) if x in (151645, 151643)]
        print(f"  row{i} first_eos={eos_pos[:3]} last_tokens={row[-6:]}")
    btexts = processor.batch_decode(bseq, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    for i, t in enumerate(btexts):
        print(f"  cand{i}", t[:50])
    if b.beam_indices is not None:
        bi = b.beam_indices[0].tolist()
        print("beam idx first 12", bi[:12], "len", len(bi))
        gen_ids = bseq[0].tolist()
        hits = 0
        for t in range(min(10, len(gen_ids))):
            parent = bi[t] if t < len(bi) else -1
            if parent < 0:
                continue
            local = torch.log_softmax(b.scores[t][parent].float(), dim=-1)
            sel = float(local[gen_ids[t]])
            print(f"  step{t} parent={parent} sel={sel:.4f} tid={gen_ids[t]}")
            hits += 1
        print("checked", hits)

    print("=== sample 2 ===")
    with torch.no_grad():
        s = model.generate(**inputs, max_new_tokens=512, do_sample=True,
                           temperature=0.7, top_p=0.9, num_return_sequences=2,
                           output_scores=True)
    print("seqs", s.sequences.shape, "scores", len(s.scores))
    stexts = processor.batch_decode(s.sequences[:, input_len:], skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    for i, t in enumerate(stexts):
        print(f"  samp{i}", t[:50])


if __name__ == "__main__":
    main()