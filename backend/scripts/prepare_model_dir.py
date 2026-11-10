"""补齐模型目录中 HF Processor 必需的文件（ModelScope 快照常缺）。

背景：ModelScope / 部分镜像下载的 Qwen3-ASR 目录只有 vocab.json + merges.txt，
缺 tokenizer.json、processor_config.json，`AutoProcessor.from_pretrained(...,
local_files_only=True)` 会直接报错。本脚本基于目录内已有词表生成缺失文件，
幂等，可重复执行。

用法：
    backend\\.venv\\Scripts\\python.exe backend\\scripts\\prepare_model_dir.py models\\Qwen3-ASR-1.7B
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("用法：python prepare_model_dir.py <模型目录>")
        return 2
    d = Path(sys.argv[1]).resolve()
    if not (d / "config.json").is_file():
        print(f"[错误] 不是模型目录（缺 config.json）：{d}")
        return 1

    # 个别仓库把音频特征配置写成 preprocessor_config.json，
    # AutoProcessor 需要 processor_config.json 才会创建 Qwen3ASRProcessor
    if not (d / "processor_config.json").is_file():
        src = d / "preprocessor_config.json"
        if src.is_file():
            shutil.copy(src, d / "processor_config.json")
            print("[补齐] processor_config.json")

    # 从 vocab.json + merges.txt 构建 fast tokenizer，写出 tokenizer.json /
    # special_tokens_map.json / added_tokens.json；全程离线
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(d), local_files_only=True, trust_remote_code=False)
    tok.save_pretrained(str(d))
    print(f"[补齐] tokenizer.json（{type(tok).__name__}，{len(tok)} tokens）")

    print(f"[完成] {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())