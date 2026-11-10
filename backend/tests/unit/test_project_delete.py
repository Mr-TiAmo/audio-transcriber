"""项目级联删除单测（探索性 UI 测试发现的真实 bug 回归）。

背景：PRAGMA foreign_keys=ON 下，直接 DELETE FROM projects 会因
transcription_runs / transcript_segments / edit_operations 等关联行抛
IntegrityError（E2E test 2：删除已完成转写的项目返回 500）。
delete_project 必须按"子表先删、父表后删"顺序在单事务内清理 5 张表 + 磁盘目录。
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from app.core.database import Database, new_id, now_iso
from app.services.project_service import ProjectService


def _make_service(tmp_path):
    db = Database(tmp_path / "app.db")
    cfg = SimpleNamespace(paths=SimpleNamespace(projects=tmp_path / "projects"))
    return db, ProjectService(cfg, db)  # type: ignore[arg-type]


def _insert_project(db: Database, project_id: str) -> str:
    """插入项目行，返回 project_id。"""
    db.execute(
        """INSERT INTO projects (id, original_name, audio_path, pcm_path, file_size,
           duration_ms, status, created_at, updated_at)
           VALUES (?, 'x.wav', 'source.wav', 'standardized.wav', 1, 1000,
                   'READY', ?, ?)""",
        (project_id, now_iso(), now_iso()),
    )
    return project_id


def _insert_run_with_relations(db: Database, project_id: str) -> str:
    """插入一条 COMPLETED run 及其 segment/postprocess/edit 关联行，返回 run_id。"""
    run_id = new_id("run")
    db.execute(
        "INSERT INTO transcription_runs (id, project_id, status) VALUES (?, ?, 'COMPLETED')",
        (run_id, project_id),
    )
    db.execute(
        """INSERT INTO transcript_segments (id, run_id, sequence_no, start_ms, end_ms,
           original_text, updated_at) VALUES (?, ?, 0, 0, 1000, '文本', ?)""",
        (new_id("seg"), run_id, now_iso()),
    )
    db.execute(
        "INSERT INTO postprocess_runs (id, run_id, created_at) VALUES (?, ?, ?)",
        (new_id("pp"), run_id, now_iso()),
    )
    db.execute(
        """INSERT INTO edit_operations (id, project_id, run_id, op_type, payload, created_at)
           VALUES (?, ?, ?, 'single_edit', '{}', ?)""",
        (new_id("edit"), project_id, run_id, now_iso()),
    )
    return run_id


def test_direct_delete_project_row_hits_foreign_key(tmp_path):
    """bug 场景本身：有关联行时直接删 projects 行必须抛 IntegrityError。"""
    db, _ = _make_service(tmp_path)
    pid = _insert_project(db, new_id("prj"))
    _insert_run_with_relations(db, pid)
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("DELETE FROM projects WHERE id=?", (pid,))


def test_delete_project_cascades_all_tables(tmp_path):
    """delete_project 后 5 张表中该项目的所有行都被清理。"""
    db, svc = _make_service(tmp_path)
    pid = _insert_project(db, new_id("prj"))
    run_id = _insert_run_with_relations(db, pid)

    svc.delete_project(pid)

    assert db.query_one("SELECT id FROM projects WHERE id=?", (pid,)) is None
    assert db.query_one("SELECT id FROM transcription_runs WHERE id=?", (run_id,)) is None
    assert db.query("SELECT id FROM transcript_segments WHERE run_id=?", (run_id,)) == []
    assert db.query("SELECT id FROM postprocess_runs WHERE run_id=?", (run_id,)) == []
    assert db.query("SELECT id FROM edit_operations WHERE project_id=?", (pid,)) == []


def test_delete_project_without_relations(tmp_path):
    """空项目（无 run）删除回归：修复前也能删，修复后不能退化。"""
    db, svc = _make_service(tmp_path)
    pid = _insert_project(db, new_id("prj"))
    svc.delete_project(pid)
    assert db.query_one("SELECT id FROM projects WHERE id=?", (pid,)) is None


def test_delete_project_keeps_other_projects(tmp_path):
    """只删目标项目：其他项目及其关联数据不受影响。"""
    db, svc = _make_service(tmp_path)
    pid_a = _insert_project(db, new_id("prj"))
    pid_b = _insert_project(db, new_id("prj"))
    _insert_run_with_relations(db, pid_a)
    run_b = _insert_run_with_relations(db, pid_b)

    svc.delete_project(pid_a)

    assert db.query_one("SELECT id FROM projects WHERE id=?", (pid_b,)) is not None
    assert db.query_one("SELECT id FROM transcription_runs WHERE id=?", (run_b,)) is not None


def test_delete_project_removes_disk_dir(tmp_path):
    """磁盘项目目录被删除；目录不存在时也不抛错（ignore_errors）。"""
    db, svc = _make_service(tmp_path)
    pid = _insert_project(db, new_id("prj"))
    project_dir = tmp_path / "projects" / pid
    (project_dir / "audio").mkdir(parents=True)
    (project_dir / "audio" / "source.wav").write_bytes(b"fake")

    svc.delete_project(pid)
    assert not project_dir.exists()

    # 再删一次（目录与行都已不在）：幂等，不抛错
    svc.delete_project(pid)
