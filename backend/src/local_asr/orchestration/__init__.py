"""跨解释器任务编排。

编排层把"识别 → 可选错误检测 → 纠错 → 校验 → 数字规范化"串成一条任务，
各阶段在各自的 venv 中以子进程执行，对外只暴露 :func:`run`。

CLI（``local-asr qwen3 transcribe`` / ``scripts/local_asr.py``）与未来的 GUI
都只依赖本包，不需要理解三个 venv 的内部实现。
"""

from local_asr.orchestration.environment import (
    EnvError,
    PythonEnv,
    discover,
    project_root,
)
from local_asr.orchestration.runner import (
    OrchestrationError,
    OrchestrationOptions,
    OrchestrationResult,
    run,
)

__all__ = [
    "EnvError",
    "OrchestrationError",
    "OrchestrationOptions",
    "OrchestrationResult",
    "PythonEnv",
    "discover",
    "project_root",
    "run",
]