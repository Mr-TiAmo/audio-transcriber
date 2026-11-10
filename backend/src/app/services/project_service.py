# 项目导入服务（V3.0 §4.3 ProjectService + AudioProbeService 职责）
# ffprobe 预检 -> 保存原件 -> 16kHz 标准化副本
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from local_asr.audio import audio_duration, prepare_pcm16k

from ..core.config import AppConfig
from ..core.database import Database, new_id, now_iso

# 导入约束（V3.0 §2.1 / §13.2）
MAX_DURATION_MS = 60 * 60 * 1000  # 长音频上限 60 分钟
MAX_FILE_BYTES = 2 * 1024**3  # 2GB 上传上限


class ProjectError(ValueError):
    """导入校验失败（用户可读）。"""


@dataclass
class ImportResult:
    project_id: str
    original_name: str
    duration_ms: int
    format: str
    codec: str
    sample_rate: int
    channels: int


def ensure_ffmpeg_on_path(cfg: AppConfig) -> None:
    """把捆绑 FFmpeg 注入本进程 PATH（子进程编排继承，不污染系统）。"""
    bin_dir = str(cfg.paths.ffmpeg_bin)
    if bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def _ffprobe(cfg: AppConfig, path: Path) -> dict:
    """ffprobe JSON 预检；不可解码时抛 ProjectError。"""
    ffprobe = cfg.paths.ffmpeg_bin / "ffprobe.exe"
    if not ffprobe.exists():
        ffprobe = Path(shutil.which("ffprobe") or "")
    if not ffprobe.exists():
        raise ProjectError("未找到 ffprobe（tools/ffmpeg 或系统 PATH）")
    try:
        out = subprocess.run(
            [
                str(ffprobe), "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(path),
            ],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise ProjectError(f"音频文件无法解码: {exc.stderr[:200]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ProjectError("ffprobe 预检超时（文件可能损坏）") from exc
    return json.loads(out)


class ProjectService:
    def __init__(self, cfg: AppConfig, db: Database):
        self.cfg = cfg
        self.db = db

    def import_audio(self, filename: str, data: bytes) -> ImportResult:
        """导入单个音频：预检 -> 建项目目录 -> 复制原件 -> 标准化。

        任一步失败都整体清理项目目录，不残留垃圾（探索性测试发现：
        ffprobe 校验失败会留下孤儿目录与源文件）。
        """
        if not data:
            raise ProjectError("空文件")
        if len(data) > MAX_FILE_BYTES:
            raise ProjectError("文件超过 2GB 上限")

        project_id = new_id("prj")
        project_dir = self.cfg.paths.projects / project_id
        try:
            return self._import_into_dir(filename, data, project_id, project_dir)
        except Exception:
            shutil.rmtree(project_dir, ignore_errors=True)
            raise

    def _import_into_dir(self, filename: str, data: bytes, project_id: str,
                         project_dir: Path) -> ImportResult:
        audio_dir = project_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # 磁盘名用 UUID 防路径穿越；原名仅作展示（§13.2）
        safe_ext = Path(filename).suffix.lower()
        if safe_ext and not safe_ext.isascii():
            safe_ext = ""
        source_path = audio_dir / f"source{safe_ext}"
        source_path.write_bytes(data)

        # ffprobe 预检（§5.1：容器/音轨/编码/时长）
        probe = _ffprobe(self.cfg, source_path)
        streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
        if not streams:
            source_path.unlink(missing_ok=True)
            raise ProjectError("文件不包含可解码音轨")
        stream = streams[0]
        fmt = probe.get("format", {})
        duration_s = float(fmt.get("duration") or stream.get("duration") or 0)
        if duration_s <= 0:
            duration_s = audio_duration(source_path)
        duration_ms = int(duration_s * 1000)
        if duration_ms > MAX_DURATION_MS:
            source_path.unlink(missing_ok=True)
            raise ProjectError(f"音频时长 {duration_s/60:.0f} 分钟超过 60 分钟上限")

        # 16kHz mono 标准化副本（V3.0 §5.1：原件只读，副本供模型）
        pcm_path = audio_dir / "standardized.wav"
        try:
            prepare_pcm16k(source_path, pcm_path)
        except RuntimeError as exc:
            source_path.unlink(missing_ok=True)
            raise ProjectError(str(exc)) from exc
        # prepare_pcm16k 对已是目标格式的文件返回原路径：显式复制为标准化副本
        if not pcm_path.exists():
            shutil.copy2(source_path, pcm_path)

        self.db.execute(
            """INSERT INTO projects (id, original_name, audio_path, pcm_path, file_size,
               duration_ms, format, codec, sample_rate, channels, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'READY', ?, ?)""",
            (
                project_id, filename, source_path.name, pcm_path.name, len(data),
                duration_ms, fmt.get("format_name"), stream.get("codec_name"),
                int(stream.get("sample_rate") or 0), int(stream.get("channels") or 0),
                now_iso(), now_iso(),
            ),
        )
        return ImportResult(
            project_id=project_id,
            original_name=filename,
            duration_ms=duration_ms,
            format=fmt.get("format_name") or "",
            codec=stream.get("codec_name") or "",
            sample_rate=int(stream.get("sample_rate") or 0),
            channels=int(stream.get("channels") or 0),
        )

    def delete_project(self, project_id: str) -> None:
        """级联删除项目：磁盘目录 + 全部关联行（单事务）。

        外键约束（PRAGMA foreign_keys=ON）下直接删 projects 会因
        transcription_runs/transcript_segments 等关联行报 IntegrityError，
        必须按"子表先删、父表后删"顺序在同一事务内清理。
        """
        shutil.rmtree(self.cfg.paths.projects / project_id, ignore_errors=True)
        run_scope = "(SELECT id FROM transcription_runs WHERE project_id=?)"
        self.db.execute_transaction([
            ("DELETE FROM transcript_segments WHERE run_id IN " + run_scope, (project_id,)),
            ("DELETE FROM postprocess_runs WHERE run_id IN " + run_scope, (project_id,)),
            ("DELETE FROM edit_operations WHERE project_id=?", (project_id,)),
            ("DELETE FROM transcription_runs WHERE project_id=?", (project_id,)),
            ("DELETE FROM projects WHERE id=?", (project_id,)),
        ])

    def get_project(self, project_id: str) -> dict | None:
        row = self.db.query_one("SELECT * FROM projects WHERE id = ?", (project_id,))
        return dict(row) if row else None

    def list_projects(self) -> list[dict]:
        rows = self.db.query(
            "SELECT id, original_name, duration_ms, status, created_at, updated_at "
            "FROM projects ORDER BY created_at DESC"
        )
        return [dict(r) for r in rows]

    def audio_path(self, project_id: str) -> Path | None:
        row = self.db.query_one(
            "SELECT audio_path FROM projects WHERE id = ?", (project_id,))
        if not row:
            return None
        return self.cfg.paths.projects / project_id / "audio" / row["audio_path"]
