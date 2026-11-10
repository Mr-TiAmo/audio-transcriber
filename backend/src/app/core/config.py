# 应用配置加载（config/default.toml + 运行时路径解析）
# 路径全部解析为绝对路径；项目根 = backend/src/app/core 向上 4 级
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# 项目根目录：audio-transcriber/
ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = ROOT / "config" / "default.toml"


@dataclass(frozen=True)
class Paths:
    """运行期目录布局（V3.0 §9.1）。"""

    root: Path
    config: Path
    data: Path
    projects: Path
    database: Path
    temp: Path
    logs: Path
    models: Path
    scripts: Path  # 编排层子进程脚本（backend/scripts）
    ffmpeg_bin: Path


@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int
    config: dict = field(repr=False, default_factory=dict)
    paths: Paths | None = None


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    """加载默认配置并解析路径。"""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    app = raw.get("app", {})
    # 配置中的相对路径统一相对 config/ 目录解析（config/../data -> <root>/data）
    cfg_dir = path.parent
    data_dir = (cfg_dir / app.get("data_dir", "../data")).resolve()
    ffmpeg_cfg = raw.get("ffmpeg", {})
    ffmpeg_bin = (cfg_dir / ffmpeg_cfg.get("bin_dir", "../tools/ffmpeg/bin")).resolve()

    paths = Paths(
        root=ROOT,
        config=path,
        data=data_dir,
        projects=data_dir / "projects",
        database=data_dir / "database",
        temp=data_dir / "temp",
        logs=data_dir / "logs",
        models=ROOT / "models",
        scripts=ROOT / "backend" / "scripts",
        ffmpeg_bin=ffmpeg_bin,
    )
    # 确保运行目录存在
    for d in (paths.projects, paths.database, paths.temp, paths.logs):
        d.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        host=app.get("host", "127.0.0.1"),
        port=int(app.get("port", 8730)),
        config=raw,
        paths=paths,
    )


# 可选 ASR 模型：键是前端下拉框的值，值是 models/ 下的目录名。
# 0.6B 默认（快、约 1.8GB）；1.7B 高质量（约 4GB，精度更高、显存占用更大）。
ASR_MODELS: dict[str, str] = {
    "0.6b": "Qwen3-ASR-0.6B",
    "1.7b": "Qwen3-ASR-1.7B",
}
# 前端下拉框文案
ASR_MODEL_LABELS: dict[str, str] = {
    "0.6b": "0.6B（快速，默认）",
    "1.7b": "1.7B（高质量）",
}
DEFAULT_ASR_MODEL = "0.6b"


def asr_model_dir(cfg: AppConfig, key: str = DEFAULT_ASR_MODEL) -> Path:
    """按前端模型键解析模型目录；键非法时抛 ValueError（由上层转 409）。"""
    name = ASR_MODELS.get(key)
    if name is None:
        raise ValueError(f"未知模型：{key}")
    return cfg.paths.models / name


def is_model_installed(model_dir: Path) -> bool:
    """模型目录是否完整可用。

    ModelScope 边下边写：下载中的权重是 `*.incomplete` 临时文件，
    只查 config.json 会把「正在下载」误判为可用，任务随后在加载阶段失败。
    """
    if not (model_dir / "config.json").is_file():
        return False
    return not any(model_dir.glob("*.incomplete"))


# 默认 ASR 模型目录（V3.0：0.6B 默认引擎）
def default_model_dir(cfg: AppConfig) -> Path:
    return asr_model_dir(cfg, DEFAULT_ASR_MODEL)
