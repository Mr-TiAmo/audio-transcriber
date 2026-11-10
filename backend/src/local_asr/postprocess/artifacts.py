"""共享的产物预检、不可变原文保存和原子发布；不依赖推理框架。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    import resource  # Unix 专属
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text_exact(path: Path) -> str:
    """保留 UTF-8 原文的空白和换行，不做 universal-newline 转换。"""
    return path.read_bytes().decode("utf-8")


def artifact_paths(directory: Path, stem: str) -> dict[str, Path]:
    if not stem or stem in (".", "..") or Path(stem).name != stem or "\\" in stem:
        raise ValueError("stem 必须是非空文件名前缀，不能包含路径")
    return {
        "raw": directory / f"{stem}_raw.txt",
        "corrected": directory / f"{stem}_corrected.txt",
        "final": directory / f"{stem}_final.txt",
        "edits": directory / f"{stem}_edits.jsonl",
        "stats": directory / f"{stem}_postprocess.json",
    }


def same_path(a: Path, b: Path) -> bool:
    return a.resolve() == b.resolve() or (a.exists() and b.exists() and a.samefile(b))


def preflight_paths(
    paths: dict[str, Path],
    *,
    inputs: tuple[Path, ...] = (),
    raw_keys: tuple[str, ...] = ("raw",),
    raw_text: str | None = None,
    allow_input_raw: bool = False,
) -> None:
    """只检查，不创建目录；既有派生文件默认拒绝，raw 仅同内容复用。"""
    items = list(paths.items())
    for index, (key, path) in enumerate(items):
        for _, other in items[:index]:
            if same_path(path, other):
                raise ValueError(f"输出路径冲突: {path} 与 {other}")
        for source in inputs:
            if same_path(path, source) and not (allow_input_raw and key in raw_keys):
                raise ValueError(f"输入与输出路径冲突: {source} 与 {path}")
        if path.is_symlink():
            raise ValueError(f"输出不能是符号链接: {path}")
        if path.exists():
            if not path.is_file():
                raise ValueError(f"输出不是普通文件: {path}")
            if key not in raw_keys:
                raise FileExistsError(f"产物已存在，请换目录或 stem: {path}")
            if raw_text is not None and path.read_bytes() != raw_text.encode("utf-8"):
                raise FileExistsError(f"原文检查点内容不同，禁止覆盖: {path}")
        parent = path.parent
        while not parent.exists():
            parent = parent.parent
        if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
            raise PermissionError(f"输出目录不可写: {parent}")


def _sync_directory(directory: Path) -> None:
    # Windows 不允许以 O_RDONLY 打开目录（PermissionError），且没有 POSIX 目录 fsync 语义；
    # 文件本身的 fsync 已保证数据落盘，目录条目同步在 Windows 上依赖 NTFS 元数据日志，跳过即可。
    if sys.platform == "win32":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str, *, overwrite: bool = True) -> None:
    """原子写文本：同目录暂存 + fsync + 落位，读者永远看不到半截文件。

    与 :func:`publish_text` 的区别：本函数面向**允许替换**的用户输出
    （编排层发布 final 文本，``--overwrite`` 时覆盖旧文件），用
    ``os.replace`` 原子替换；``publish_text`` 面向不可变产物，
    用 ``os.link`` 独占创建、绝不替换已存在的目标。

    Args:
        path: 目标路径；父目录不存在时自动创建。
        text: 写入内容（UTF-8）。
        overwrite: False 时目标已存在则抛 FileExistsError。

    Raises:
        ValueError: 目标是符号链接（防跳转写入）。
        FileExistsError: ``overwrite=False`` 且目标已存在。
    """
    if path.is_symlink():
        raise ValueError(f"拒绝写入符号链接: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"输出已存在: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except Exception:
        # 落位失败时清理暂存文件，不在目标目录留垃圾
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def publish_text(path: Path, text: str, *, immutable: bool = False) -> None:
    """同目录暂存并独占发布；原子创建不替换目标，防止预检后的覆盖竞争。"""
    data = text.encode("utf-8")
    if path.is_symlink():
        raise ValueError(f"拒绝写入符号链接: {path}")
    if path.exists():
        if immutable and path.is_file() and path.read_bytes() == data:
            return
        raise FileExistsError(f"禁止覆盖已有产物: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".asr-publish-") as temp:
        staged = Path(temp) / "content"
        with staged.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staged, path)
        except FileExistsError:
            if not (immutable and not path.is_symlink() and path.is_file()
                    and path.read_bytes() == data):
                raise
        _sync_directory(path.parent)


def runtime_versions() -> dict[str, str | None]:
    result = {"python": sys.version.split()[0]}
    for name in ("wetext", "mlx-lm", "transformers", "torch", "av"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def peak_rss_mb() -> float:
    if resource is not None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(rss / (1024 * 1024 if sys.platform == "darwin" else 1024), 2)
    # Windows：用 psutil 取当前 RSS（无历史峰值，取当前值作为近似）
    import psutil  # 局部导入，避免 Unix 平台强制依赖

    return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)


def json_hash(value: object) -> str:
    return text_hash(json.dumps(value, ensure_ascii=False, sort_keys=True))


def model_manifest(path: Path) -> dict:
    """记录小型配置内容指纹及权重文件清单，不将 stat 冒充权重内容校验。"""
    metadata = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.glob("*.json")) if p.is_file()
    }
    weights = []
    for p in sorted(path.glob("*.safetensors")):
        stat = p.stat()
        weights.append({"name": p.name, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    manifest = {
        "metadata_sha256": metadata,
        "weights": weights,
        "weight_identity_scope": "file_metadata_not_content_hash",
    }
    return {**manifest, "manifest_sha256": json_hash(manifest)}


def source_fingerprint() -> str:
    """无 Git 元数据时也能识别规则实现变化，不读取模型权重。"""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
