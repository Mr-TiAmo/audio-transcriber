#!/usr/bin/env python3
"""统一编排入口：识别 → 可选语义纠错 → 修改校验 → 数字规范化。

与 ``local-asr qwen3 transcribe`` 完全等价，二者共用
``local_asr.orchestration.cli`` 这一份实现。

各阶段在各自的 venv 中以子进程运行（ASR 用 .venv-funasr，纠错与 ITN 用
.venv-mlx），因此本脚本自身不需要安装 torch 或 mlx-lm。

用法::

    python scripts/local_asr.py transcribe \
        --audio input.wav \
        --model models/Qwen3-ASR-1.7B-hf \
        --device mps \
        --semantic-correction mlx \
        --corrector-model models/ChineseErrorCorrector4-4B-mlx-8bit \
        --itn \
        --glossary output/glossary.json \
        --output output/result.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from local_asr.orchestration.cli import main  # noqa: E402

if __name__ == "__main__":
    main()