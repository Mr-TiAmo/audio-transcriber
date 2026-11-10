# SQLite 持久层（V3.0 §9.2 表结构 / §9.3 存储原则）
# 单机单用户：WAL + busy_timeout + 外键；同步 sqlite3（路由在线程池执行）
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id            TEXT PRIMARY KEY,
    original_name TEXT NOT NULL,
    audio_path    TEXT NOT NULL,          -- 项目内音频相对路径（source 原件）
    pcm_path      TEXT NOT NULL,          -- 16kHz mono 标准化副本相对路径
    file_size     INTEGER NOT NULL,
    duration_ms   INTEGER NOT NULL,
    format        TEXT,
    codec         TEXT,
    sample_rate   INTEGER,
    channels      INTEGER,
    status        TEXT NOT NULL DEFAULT 'CREATED',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcription_runs (
    id                  TEXT PRIMARY KEY,     -- run_id
    project_id          TEXT NOT NULL REFERENCES projects(id),
    status              TEXT NOT NULL,        -- CREATED/PROCESSING/COMPLETED/FAILED/CANCELLED
    stage               TEXT,
    progress_percent    REAL NOT NULL DEFAULT 0,
    processed_until_ms  INTEGER,
    processed_speech_ms INTEGER,
    total_speech_ms     INTEGER,
    elapsed_ms          INTEGER,
    eta_ms              INTEGER,
    model_name          TEXT,
    strategy            TEXT,
    batch_size          INTEGER,
    error_code          TEXT,
    error_message       TEXT,
    work_dir            TEXT,                 -- 编排层工作目录（中间产物所在）
    started_at          TEXT,
    finished_at         TEXT
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id                TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL REFERENCES transcription_runs(id),
    sequence_no       INTEGER NOT NULL,
    start_ms          INTEGER NOT NULL,       -- VAD 级（必有）
    end_ms            INTEGER NOT NULL,
    original_text     TEXT NOT NULL,          -- ASR 原文（raw）
    edited_text       TEXT,                   -- 用户最终文本，未编辑为 NULL
    confidence_value  REAL,                   -- Qwen3-ASR 无置信度：恒 NULL
    confidence_source TEXT NOT NULL DEFAULT 'none',
    risk_flags        TEXT NOT NULL DEFAULT '[]',   -- JSON 数组
    review_status     TEXT NOT NULL DEFAULT 'pending',
    aligned_tokens    TEXT,                   -- JSON（Aligner 落地后填）
    low_conf_spans    TEXT,                   -- JSON [[start,end],...] 低置信字符区间
    review_priority   REAL,                   -- 段级风险分（高=优先复核）
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_run ON transcript_segments(run_id, sequence_no);

CREATE TABLE IF NOT EXISTS postprocess_runs (
    id                     TEXT PRIMARY KEY,
    run_id                 TEXT NOT NULL REFERENCES transcription_runs(id),
    semantic_correction    TEXT NOT NULL DEFAULT 'none',
    corrector_model_version TEXT,
    itn_rule_fingerprint   TEXT,
    edits                  TEXT,
    created_at             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edit_operations (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id),
    run_id      TEXT NOT NULL REFERENCES transcription_runs(id),
    op_type     TEXT NOT NULL,      -- single_edit / batch_replace / undo / redo
    payload     TEXT NOT NULL,      -- JSON 明细（批量替换整体保存，支持整组撤销）
    created_at  TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Database:
    """SQLite 连接管理（线程安全：check_same_thread=False + 每调用加锁）。"""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")        # §9.3：单机 WAL
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        # 迁移：为现有库补齐 low_conf_spans 列（V3.x 置信度标亮）
        for ddl in (
            "ALTER TABLE transcript_segments ADD COLUMN low_conf_spans TEXT",
            # P1 段级风险排序（2026-11）：旧库补齐，已有则忽略
            "ALTER TABLE transcript_segments ADD COLUMN review_priority REAL",
        ):
            try:
                self._conn.execute(ddl)
            except Exception:
                pass  # 列已存在
        self._conn.commit()
        import threading

        self._lock = threading.Lock()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def execute_transaction(self, statements: list[tuple[str, tuple]]) -> None:
        """单事务执行多条语句（级联删除用）：任一失败整体回滚，不留半删状态。"""
        with self._lock:
            try:
                for sql, params in statements:
                    self._conn.execute(sql, params)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
