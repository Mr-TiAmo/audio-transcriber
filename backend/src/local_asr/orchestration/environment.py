"""解释器环境探测与模块校验。

项目里三个 venv 的依赖是互斥的：

- ``.venv-funasr``：torch / transformers / funasr（ASR 识别）
- ``.venv-mlx``：mlx_lm（语义纠错）+ wetext（数字规范化）
- ``.venv``：编排层 + 单测（纯标准库 + av；faster-whisper 已改为可选 extra）

因此 ASR 与纠错**不能在同一解释器中完成**。编排层只通过子进程调用它们，
本模块负责在启动昂贵推理前把"解释器不存在 / 模块缺失"变成明确报错。

所有探测都是一次性的 ``python -c`` 调用，不导入任何业务代码。
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ENV_FUNASR",
    "ENV_MLX",
    "EnvError",
    "PythonEnv",
    "discover",
    "ensure",
    "project_root",
    "probe_modules",
]

ENV_FUNASR = "funasr"
ENV_MLX = "mlx"

# 逻辑名 -> (默认目录, 必须能导入的模块)
_DEFAULT_LAYOUT: dict[str, tuple[str, tuple[str, ...]]] = {
    ENV_FUNASR: (".venv-funasr", ("torch", "transformers")),
    # mlx_lm 只在启用语义纠错时必需，但探测时一并记录版本，
    # 避免 require() 把"未探测"误判成"未安装"。
    ENV_MLX: (".venv-mlx", ("wetext", "mlx_lm")),
}

_PROBE_TIMEOUT_SECONDS = 30.0


class EnvError(RuntimeError):
    """解释器或依赖不满足要求。"""


@dataclass(frozen=True)
class PythonEnv:
    """一个可用于执行子进程的解释器。"""

    name: str
    python: Path
    directory: Path
    versions: dict[str, str | None] = field(default_factory=dict)

    def require(self, modules: tuple[str, ...], purpose: str) -> None:
        """校验模块可用；缺失时给出可执行的修复建议。"""
        missing = [m for m in modules if self.versions.get(m) is None]
        if not missing:
            return
        raise EnvError(
            f"{purpose}需要 {self.name} 环境具备 {', '.join(missing)}，"
            f"但 {self.python} 中不可用。请检查 {self.directory} 的依赖，"
            "编排层不会自动安装或切换环境。"
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "python": str(self.python),
            "directory": str(self.directory),
            "versions": dict(self.versions),
        }


def project_root() -> Path:
    """项目根目录（src/local_asr/orchestration/environment.py 往上三级）。"""
    return Path(__file__).resolve().parents[3]


def _venv_python(directory: Path) -> Path:
    candidate = directory / ("Scripts/python.exe" if (directory / "Scripts").is_dir() else "bin/python")
    return candidate


def probe_modules(python: Path, modules: tuple[str, ...], *, timeout: float = _PROBE_TIMEOUT_SECONDS) -> dict[str, str | None]:
    """用目标解释器查询模块可用性与版本；不导入业务代码。"""
    code = (
        "import json, importlib.util\n"
        "from importlib.metadata import PackageNotFoundError, version\n"
        "names = json.loads(%r)\n"
        "out = {}\n"
        "for name in names:\n"
        "    if importlib.util.find_spec(name) is None:\n"
        "        out[name] = None\n"
        "        continue\n"
        "    try:\n"
        "        out[name] = version(name.replace('_', '-'))\n"
        "    except PackageNotFoundError:\n"
        "        out[name] = 'unknown'\n"
        "print(json.dumps(out))\n"
    ) % (json.dumps(list(modules)))
    try:
        completed = subprocess.run(
            [str(python), "-I", "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EnvError(f"无法执行解释器 {python}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        raise EnvError(f"探测 {python} 失败: {detail[-1] if detail else completed.returncode}")
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise EnvError(f"探测 {python} 返回了非 JSON 输出") from exc
    if not isinstance(parsed, dict):
        raise EnvError(f"探测 {python} 返回结构异常")
    return {str(k): (None if v is None else str(v)) for k, v in parsed.items()}


def discover(root: Path | None = None, *, layout: dict[str, tuple[str, tuple[str, ...]]] | None = None) -> dict[str, PythonEnv]:
    """探测项目内的已知解释器。

    只识别约定目录；不存在或不可执行时不会臆造回退路径。
    """
    base = Path(root) if root is not None else project_root()
    layout = layout or _DEFAULT_LAYOUT
    envs: dict[str, PythonEnv] = {}
    for name, (relative, modules) in layout.items():
        directory = (base / relative).resolve()
        python = _venv_python(directory)
        if not python.is_file():
            continue
        envs[name] = PythonEnv(
            name=name, python=python, directory=directory,
            versions=probe_modules(python, modules),
        )
    return envs


def ensure(envs: dict[str, PythonEnv], name: str, modules: tuple[str, ...], purpose: str, root: Path | None = None) -> PythonEnv:
    """取指定环境并校验依赖；缺环境或缺模块都明确失败。"""
    env = envs.get(name)
    if env is None:
        base = Path(root) if root is not None else project_root()
        expected = base / _DEFAULT_LAYOUT[name][0]
        raise EnvError(f"{purpose}需要 {name} 环境，但未找到解释器: {_venv_python(expected)}")
    env.require(modules, purpose)
    return env


def default_layout() -> dict[str, tuple[str, tuple[str, ...]]]:
    """返回默认布局副本，供测试或自定义部署覆盖。"""
    return {k: (d, tuple(m)) for k, (d, m) in _DEFAULT_LAYOUT.items()}