"""转写模型选择单测（任务栏下拉框后端）。

覆盖三层：
1. config.asr_model_dir：键 -> models/ 目录名解析，未知键抛 ValueError；
2. TaskService.start_run：未知键 / 未安装目录在创建任务前就失败（409 语义），
   正常路径把模型目录传给后台线程、DB 记录模型名；
3. router：GET /models 的 available 标记、POST runs 的默认模型与 409 透传。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from app.api.v1 import router as router_module
from app.core.config import (
    ASR_MODELS,
    DEFAULT_ASR_MODEL,
    AppConfig,
    asr_model_dir,
)
from app.core.database import Database, new_id, now_iso
from app.services import task_service as ts_module
from app.services.task_service import (
    TaskError, TaskService, batch_for_model, chunk_seconds_for_model,
)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path) -> AppConfig:
    """最小 AppConfig：models / temp / projects / scripts 四个路径。"""
    models = tmp_path / "models"
    models.mkdir()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    paths = SimpleNamespace(
        models=models,
        temp=tmp_path / "temp",
        projects=tmp_path / "projects",
        scripts=scripts,
    )
    return AppConfig(host="127.0.0.1", port=8730, config={}, paths=paths)  # type: ignore[arg-type]


def _install_model(cfg: AppConfig, key: str) -> None:
    """在 models/ 下造一个带 config.json 的模型目录。"""
    d = cfg.paths.models / ASR_MODELS[key]  # type: ignore[union-attr]
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}", encoding="utf-8")


class _FakeThread:
    """替换 threading.Thread：记录参数但不真正执行编排层。"""

    created: list["_FakeThread"] = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started = False
        _FakeThread.created.append(self)

    def start(self) -> None:
        self.started = True


@pytest.fixture(autouse=True)
def _reset_router_globals():
    yield
    # 避免其他测试读到本文件注入的假服务
    router_module.init(None, None, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# config.asr_model_dir
# ---------------------------------------------------------------------------

def test_asr_model_dir_resolves_known_keys(tmp_path):
    cfg = _make_cfg(tmp_path)
    assert asr_model_dir(cfg, "0.6b").name == "Qwen3-ASR-0.6B"
    assert asr_model_dir(cfg, "1.7b").name == "Qwen3-ASR-1.7B"
    # 默认键
    assert asr_model_dir(cfg).name == ASR_MODELS[DEFAULT_ASR_MODEL]


def test_asr_model_dir_rejects_unknown_key(tmp_path):
    cfg = _make_cfg(tmp_path)
    with pytest.raises(ValueError):
        asr_model_dir(cfg, "42b")


# ---------------------------------------------------------------------------
# batch/切块选择：1.7B 在 8GB 卡上用 30s 块 + batch=2
# ---------------------------------------------------------------------------

def test_batch_for_model_17b_uses_batch2_with_short_chunks():
    # 1.7B + batch=4 × 60s 会溢出到 WDDM 共享显存；30s 块 × batch=2
    # 实测 infer 125.7s→39.4s（RTF 0.210→0.066），峰值约 5.3GB
    assert batch_for_model(Path("Qwen3-ASR-1.7B")) == (2, 2)
    assert chunk_seconds_for_model(Path("Qwen3-ASR-1.7B")) == 30.0


def test_batch_for_model_06b_keeps_batch4():
    assert batch_for_model(Path("Qwen3-ASR-0.6B")) == (4, 4)
    assert chunk_seconds_for_model(Path("Qwen3-ASR-0.6B")) == 60.0


# ---------------------------------------------------------------------------
# TaskService.start_run：模型在任务创建前校验
# ---------------------------------------------------------------------------

def _make_task_service(tmp_path):
    db = Database(tmp_path / "app.db")
    return db, TaskService(_make_cfg(tmp_path), db)


def _insert_project(db: Database) -> str:
    pid = new_id("prj")
    db.execute(
        """INSERT INTO projects (id, original_name, audio_path, pcm_path, file_size,
           duration_ms, status, created_at, updated_at)
           VALUES (?, 'x.wav', 'source.wav', 'standardized.wav', 1, 1000,
                   'READY', ?, ?)""",
        (pid, now_iso(), now_iso()),
    )
    return pid


def test_start_run_rejects_unknown_model(tmp_path):
    db, svc = _make_task_service(tmp_path)
    pid = _insert_project(db)
    with pytest.raises(TaskError, match="未知模型"):
        svc.start_run(pid, model="42b")
    # 未创建任务行
    assert db.query_one("SELECT id FROM transcription_runs") is None


def test_start_run_rejects_uninstalled_model(tmp_path):
    db, svc = _make_task_service(tmp_path)
    pid = _insert_project(db)
    with pytest.raises(TaskError, match="模型未安装"):
        svc.start_run(pid, model="1.7b")
    assert db.query_one("SELECT id FROM transcription_runs") is None


def test_start_run_rejects_half_downloaded_model(tmp_path):
    """ModelScope 下载中（存在 *.incomplete）视同未安装，避免跑到加载阶段才失败。"""
    db, svc = _make_task_service(tmp_path)
    pid = _insert_project(db)
    model_dir = svc.cfg.paths.models / ASR_MODELS["1.7b"]
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model-00001-of-00002.safetensors.incomplete").write_text("")
    with pytest.raises(TaskError, match="下载未完成"):
        svc.start_run(pid, model="1.7b")


def test_start_run_records_model_and_passes_dir(tmp_path, monkeypatch):
    db, svc = _make_task_service(tmp_path)
    pid = _insert_project(db)
    _install_model(svc.cfg, "1.7b")
    _FakeThread.created.clear()
    monkeypatch.setattr(ts_module.threading, "Thread", _FakeThread)

    run_id = svc.start_run(pid, model="1.7b")

    row = db.query_one("SELECT model_name FROM transcription_runs WHERE id=?", (run_id,))
    assert row["model_name"] == "Qwen3-ASR-1.7B"
    assert len(_FakeThread.created) == 1
    thread = _FakeThread.created[0]
    assert thread.started
    # 线程参数倒数第二位是解析后的模型目录（_execute 用它构造
    # OrchestrationOptions），末位是快速模式开关；
    # TaskService 以关键字方式构造 Thread，位置参数在 kwargs["args"] 里
    assert thread.kwargs["args"][-2] == svc.cfg.paths.models / "Qwen3-ASR-1.7B"
    assert thread.kwargs["args"][-1] is False


# ---------------------------------------------------------------------------
# router：模型列表与创建任务
# ---------------------------------------------------------------------------

class _FakeTasks:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fast_flags: list[bool] = []

    def start_run(self, project_id: str, model: str = DEFAULT_ASR_MODEL,
                  fast: bool = False) -> str:
        self.calls.append((project_id, model))
        self.fast_flags.append(fast)
        if model not in ASR_MODELS:
            raise TaskError(f"未知模型：{model}")
        return "run_test"


def _make_client(cfg: AppConfig, tasks: _FakeTasks) -> TestClient:
    app = FastAPI()
    app.include_router(router_module.router)
    router_module.init(cfg, None, tasks)  # type: ignore[arg-type]
    return TestClient(app)


def test_list_models_marks_availability(tmp_path):
    cfg = _make_cfg(tmp_path)
    _install_model(cfg, "0.6b")
    resp = _make_client(cfg, _FakeTasks()).get("/api/v1/models")
    assert resp.status_code == 200
    items = {m["key"]: m for m in resp.json()}
    assert set(items) == {"0.6b", "1.7b"}
    assert items["0.6b"]["available"] is True
    assert items["1.7b"]["available"] is False
    assert "1.7" in items["1.7b"]["label"]


def test_list_models_half_downloaded_marked_unavailable(tmp_path):
    cfg = _make_cfg(tmp_path)
    d = cfg.paths.models / ASR_MODELS["1.7b"]
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "model-00001-of-00002.safetensors.incomplete").write_text("")
    items = {m["key"]: m for m in _make_client(cfg, _FakeTasks()).get("/api/v1/models").json()}
    assert items["1.7b"]["available"] is False


def test_create_run_defaults_to_06b_without_body(tmp_path):
    tasks = _FakeTasks()
    resp = _make_client(_make_cfg(tmp_path), tasks).post("/api/v1/projects/prj_1/runs")
    assert resp.status_code == 200
    assert resp.json()["runId"] == "run_test"
    assert tasks.calls == [("prj_1", DEFAULT_ASR_MODEL)]


def test_create_run_passes_selected_model(tmp_path):
    tasks = _FakeTasks()
    resp = _make_client(_make_cfg(tmp_path), tasks).post(
        "/api/v1/projects/prj_1/runs", json={"model": "1.7b"})
    assert resp.status_code == 200
    assert tasks.calls == [("prj_1", "1.7b")]


def test_create_run_passes_fast_mode(tmp_path):
    tasks = _FakeTasks()
    resp = _make_client(_make_cfg(tmp_path), tasks).post(
        "/api/v1/projects/prj_1/runs", json={"model": "0.6b", "fast": True})
    assert resp.status_code == 200
    assert tasks.fast_flags == [True]


def test_create_run_unknown_model_returns_409(tmp_path):
    resp = _make_client(_make_cfg(tmp_path), _FakeTasks()).post(
        "/api/v1/projects/prj_1/runs", json={"model": "42b"})
    assert resp.status_code == 409
    assert "未知模型" in resp.json()["detail"]