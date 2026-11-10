# 本地 API v1 路由（V3.0 §11 接口清单）
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ...core.config import (
    ASR_MODEL_LABELS,
    ASR_MODELS,
    DEFAULT_ASR_MODEL,
    AppConfig,
    is_model_installed,
)
from ...core.database import now_iso
from ...services.edit_service import (
    EditError,
    apply_replace,
    preview_replace,
    record_single_edit,
    undo_operation,
)
from ...services.project_service import ProjectError, ProjectService
from ...services.segment_service import load_segments
from ...services.task_service import TaskError, TaskService

router = APIRouter(prefix="/api/v1")

# 延迟注入（main.py 启动时调用 init）
_cfg: AppConfig | None = None
_projects: ProjectService | None = None
_tasks: TaskService | None = None


def init(cfg: AppConfig, projects: ProjectService, tasks: TaskService) -> None:
    global _cfg, _projects, _tasks
    _cfg, _projects, _tasks = cfg, projects, tasks


def _tasks_or_404() -> TaskService:
    if _tasks is None:
        raise HTTPException(503, "服务未就绪")
    return _tasks


# ------------------------------------------------------------------
# 健康检查（§11：探活统一走 /api/v1 前缀；/health 保留给 start.bat 兼容）
# ------------------------------------------------------------------
_APP_INFO = {"app": "AudioTranscriber", "version": "0.1.0"}


def health_payload() -> dict:
    """健康检查负载（/api/v1/health 与 /health 共用，避免两处漂移）。

    `ready` 表示服务层（配置 + 数据库 + TaskService）已初始化完成：
    FastAPI 的路由在 startup 回调之前就已可访问，此时返回 503 会让
    探活脚本误判服务不可用，故始终 200，用字段区分启动中/就绪。
    """
    ready = _tasks is not None
    return {"status": "ok" if ready else "starting", "ready": ready, **_APP_INFO}


@router.get("/health")
def health() -> dict:
    return health_payload()


# ------------------------------------------------------------------
# 项目（§11：导入/列表/详情/删除）
# ------------------------------------------------------------------
@router.post("/projects")
async def create_project(file: UploadFile = File(...)) -> dict:
    """multipart 导入单个音频、预检、创建项目。"""
    data = await file.read()
    try:
        result = _projects.import_audio(file.filename or "audio", data)  # type: ignore[union-attr]
    except ProjectError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "id": result.project_id,
        "originalName": result.original_name,
        "durationMs": result.duration_ms,
        "format": result.format,
        "codec": result.codec,
        "sampleRate": result.sample_rate,
        "channels": result.channels,
    }


@router.get("/projects")
def list_projects() -> list[dict]:
    return _projects.list_projects()  # type: ignore[union-attr]


@router.get("/projects/{project_id}")
def get_project(project_id: str) -> dict:
    p = _projects.get_project(project_id)  # type: ignore[union-attr]
    if not p:
        raise HTTPException(404, "项目不存在")
    return p


@router.delete("/projects/{project_id}")
def delete_project(project_id: str) -> dict:
    tasks = _tasks_or_404()
    run = tasks.current_run()
    if run and run["project_id"] == project_id:
        raise HTTPException(409, "项目正在转写，禁止删除")
    p = _projects.get_project(project_id)  # type: ignore[union-attr]
    if not p:
        raise HTTPException(404, "项目不存在")
    # 级联删除下沉到 service（外键约束下需按子表→父表顺序单事务清理）
    _projects.delete_project(project_id)  # type: ignore[union-attr]
    return {"deleted": project_id}


# ------------------------------------------------------------------
# 任务（§11：创建/取消/查询）
# ------------------------------------------------------------------
class RunCreateRequest(BaseModel):
    """转写任务参数：模型键（0.6b 默认 / 1.7b 高质量）。

    fast=True 为「快速模式」：跳过 ForcedAligner 字符级对齐（省 2–6s），
    句级时间戳退回按字数插值，低置信标亮不受影响（只依赖 token entropy）。

    crossCheck=True 为「交叉验证」：主模型推理后调用第二模型（FunASR
    Paraformer，CPU）整片转写，仅核验低置信词组/词语——两模型一致的
    词取消标亮、不一致的保留。第二模型与主推理并行，短音频额外等待
    约 30–48s，默认关闭。
    """
    model: str = DEFAULT_ASR_MODEL
    fast: bool = False
    crossCheck: bool = False


@router.get("/models")
def list_models() -> list[dict]:
    """可用 ASR 模型列表（前端下拉框）。未安装的模型标记 available=false。"""
    cfg = _cfg
    assert cfg is not None
    return [
        {
            "key": key,
            "name": name,
            "label": ASR_MODEL_LABELS.get(key, name),
            "available": is_model_installed(cfg.paths.models / name),
        }
        for key, name in ASR_MODELS.items()
    ]


@router.post("/projects/{project_id}/runs")
def create_run(project_id: str, payload: RunCreateRequest | None = None) -> dict:
    model = payload.model if payload else DEFAULT_ASR_MODEL
    fast = payload.fast if payload else False
    cross_check = payload.crossCheck if payload else False
    try:
        run_id = _tasks_or_404().start_run(
            project_id, model=model, fast=fast, cross_check=cross_check)
    except TaskError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"runId": run_id, "status": "PROCESSING"}


@router.post("/projects/{project_id}/runs/{run_id}/cancel")
def cancel_run(project_id: str, run_id: str) -> dict:
    """协作式取消（§8.3）：终止 ASR/后处理子进程，任务标记 CANCELLED。"""
    try:
        _tasks_or_404().cancel_run(run_id)
    except TaskError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"runId": run_id, "status": "CANCELLING"}


@router.get("/tasks/current")
def current_task() -> dict:
    run = _tasks_or_404().current_run()
    return run or {"status": "IDLE"}


@router.get("/tasks/{run_id}/events")
async def task_events(run_id: str) -> StreamingResponse:
    """SSE 进度订阅（§11.1）：连接即发 task_snapshot，此后转发编排层事件。"""
    tasks = _tasks_or_404()
    queue = tasks.hub.subscribe()

    async def stream():
        try:
            run = tasks.get_run(run_id) or {"runId": run_id, "status": "UNKNOWN"}
            yield f"event: task_snapshot\ndata: {json.dumps(run, ensure_ascii=False)}\n\n"
            while True:
                if queue:
                    event = queue.pop(0)
                    if event.get("runId") == run_id:
                        kind = event.get("kind", "progress").replace("_", "-")
                        yield f"event: {kind}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    # 终态后结束流
                    if str(event.get("kind", "")).startswith("task_"):
                        return
                else:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(1)
        finally:
            tasks.hub.unsubscribe(queue)

    return StreamingResponse(stream(), media_type="text/event-stream")


# ------------------------------------------------------------------
# 转写内容（§11：片段查询/编辑/导出/音频）
# ------------------------------------------------------------------
@router.get("/projects/{project_id}/transcript")
def get_transcript(project_id: str) -> dict:
    row = _tasks.db.query_one(  # type: ignore[union-attr]
        "SELECT id FROM transcription_runs WHERE project_id=? "
        "AND status='COMPLETED' ORDER BY started_at DESC LIMIT 1", (project_id,))
    if not row:
        raise HTTPException(404, "该项目尚无完成的转写")
    return {"runId": row["id"], "segments": load_segments(_tasks.db, row["id"])}  # type: ignore[union-attr]


@router.patch("/segments/{segment_id}")
def update_segment(segment_id: str, body: dict) -> dict:
    """保存单段编辑/复核状态（§12.5 自动保存的后端落点）。

    成功后回传 updated_at，前端用它丢弃迟到的旧保存结果。
    """
    tasks = _tasks_or_404()
    row = tasks.db.query_one("SELECT id, edited_text FROM transcript_segments WHERE id=?", (segment_id,))
    if not row:
        raise HTTPException(404, "片段不存在")
    edited = body.get("editedText")
    review = body.get("reviewStatus")
    tasks.db.execute(
        "UPDATE transcript_segments SET edited_text=?, review_status=?, updated_at=? WHERE id=?",
        (edited, review or "pending", now_iso(), segment_id),
    )
    if edited != row["edited_text"]:
        # 单段编辑审计（op_type=single_edit，轻量记录 from/to）
        proj = tasks.db.query_one(
            "SELECT p.id AS project_id, s.run_id FROM transcript_segments s "
            "JOIN transcription_runs r ON r.id = s.run_id "
            "JOIN projects p ON p.id = r.project_id WHERE s.id=?", (segment_id,))
        if proj:
            record_single_edit(
                tasks.db, proj["project_id"], proj["run_id"],
                segment_id, row["edited_text"], edited)
    saved_at = tasks.db.query_one(
        "SELECT updated_at FROM transcript_segments WHERE id=?", (segment_id,))["updated_at"]
    return {"id": segment_id, "saved": True, "updatedAt": saved_at}


# ------------------------------------------------------------------
# 查找替换（§12.5：先 preview 再 apply，批量替换记 edit_operation 可整组撤销）
# ------------------------------------------------------------------
@router.post("/projects/{project_id}/replace")
def replace_text(project_id: str, body: dict) -> dict:
    mode = body.get("mode", "preview")
    find, replace = body.get("find", ""), body.get("replace", "")
    try:
        if mode == "preview":
            return {"mode": "preview", "hits": preview_replace(_tasks.db, project_id, find, replace)}  # type: ignore[union-attr]
        if mode == "apply":
            return {"mode": "apply", **apply_replace(_tasks.db, project_id, find, replace)}  # type: ignore[union-attr]
    except EditError as exc:
        raise HTTPException(422, str(exc)) from exc
    raise HTTPException(422, f"未知 mode: {mode}")


@router.post("/edit-operations/{operation_id}/undo")
def undo_edit(operation_id: str) -> dict:
    try:
        return undo_operation(_tasks.db, operation_id)  # type: ignore[union-attr]
    except EditError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/projects/{project_id}/export")
def export_txt(project_id: str) -> FileResponse:
    """TXT 导出：默认取 final_text（edited ?? original），无时间戳、无内部标记。"""
    tasks = _tasks_or_404()
    row = tasks.db.query_one(
        "SELECT id FROM transcription_runs WHERE project_id=? "
        "AND status='COMPLETED' ORDER BY started_at DESC LIMIT 1", (project_id,))
    if not row:
        raise HTTPException(404, "该项目尚无完成的转写")
    segments = load_segments(tasks.db, row["id"])
    text = "".join(s["finalText"] for s in segments)
    out = _cfg.paths.projects / project_id / "exports" / "transcript.txt"  # type: ignore[union-attr]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return FileResponse(out, filename="transcript.txt", media_type="text/plain")


@router.get("/projects/{project_id}/audio")
def get_audio(project_id: str, _range: str | None = None) -> FileResponse:
    """音频读取；FileResponse 自动支持 HTTP Range（§12.4 验收项）。"""
    path = _projects.audio_path(project_id)  # type: ignore[union-attr]
    if not path or not path.exists():
        raise HTTPException(404, "音频不存在")
    return FileResponse(path, media_type="audio/mpeg")
