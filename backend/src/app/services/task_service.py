# 任务服务（V3.0 §8 生命周期 / §4.1 编排分层 / §11.1 SSE）
# 单任务互斥 + 后台线程驱动编排层（ASR/后处理在子进程中执行，API 不加载模型）
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from local_asr.orchestration.asr_worker import AsrWorkerExecutor
from local_asr.orchestration.environment import PythonEnv, probe_modules
from local_asr.orchestration.process_manager import ProcessCancelled
from local_asr.orchestration.runner import OrchestrationOptions, OrchestrationResult, run

from ..core.config import DEFAULT_ASR_MODEL, AppConfig, asr_model_dir, is_model_installed
from ..core.database import Database, new_id, now_iso
from .segment_service import build_segments_from_stats, save_segments

# 任务终态（§8.1）
TERMINAL_STATES = ("COMPLETED", "FAILED", "CANCELLED")

# 子进程原始事件 -> (DB stage 枚举, 前端 phase 语义)。
# 编排层 TaskState 的 phase/progress 事件只进 event_bus（服务层未接），
# SSE 实际只转发子进程原始事件，因此在服务层翻译出前端需要的
# phase_updated / progress_updated（§7.2 / §12.1），并持久化到 DB
# （刷新页面后 task_snapshot 才能恢复真实阶段与进度）。
_STAGE_BY_KIND = {
    "asr_started": ("TRANSCRIBING", "transcribing"),
    "postprocess_started": ("POSTPROCESSING", "postprocessing"),
}


def batch_for_model(model_dir: Path) -> tuple[int, int]:
    """按模型显存占用选择 batch 大小（batch_size, batch_max）。

    8GB 卡上 0.6B 用 batch=4 已验证可用（长音频峰值约 7.8GB）；
    1.7B fp16 权重多约 2.6GB，batch=4 × 60s chunk 会溢出到 WDDM 共享显存
    （推理慢约 20 倍，表现为任务长时间停在 0% 进度）。实测改为 30s 块 ×
    batch=2：599s 音频 infer 125.7s → 39.4s（RTF 0.210 → 0.066），
    峰值显存约 5.3GB，文本与逐条推理几乎一致（仅 4 处数字写法差异）。
    """
    if "1.7" in model_dir.name:
        return 2, 2
    return 4, 4


def chunk_seconds_for_model(model_dir: Path) -> float:
    """切块目标秒数：1.7B 用 30s 以配合 batch=2（见 ``batch_for_model``）。"""
    return 30.0 if "1.7" in model_dir.name else 60.0


class TaskError(RuntimeError):
    """任务操作失败（用户可读）。"""


class EventHub:
    """SSE 事件总线：编排层事件 -> 订阅者队列。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[dict] = []  # 每个订阅者一个待发事件列表

    def publish(self, event: dict) -> None:
        with self._lock:
            for q in self._subscribers:
                q.append(event)

    def subscribe(self) -> list[dict]:
        q: list[dict] = []
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: list[dict]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)


class TaskService:
    def __init__(self, cfg: AppConfig, db: Database):
        self.cfg = cfg
        self.db = db
        self.hub = EventHub()
        self._mutex = threading.Lock()  # 进程内互斥（§8.2）
        self._cancel_events: dict[str, threading.Event] = {}  # runId -> 取消事件
        self._cancel_lock = threading.Lock()
        # ASR 常驻 worker（懒启动）：同一进程连续任务复用模型权重，
        # 第二个任务起省 5.5–10.5s 加载 + 2–9s 对齐模型加载。
        self._asr_worker = AsrWorkerExecutor(cfg.paths.scripts / "qwen3_asr_worker.py")

    # ------------------------------------------------------------------
    # 异常恢复（§8.4：启动时把遗留 PROCESSING 标记为中断失败）
    # ------------------------------------------------------------------
    def recover_interrupted(self) -> int:
        rows = self.db.query(
            "SELECT id FROM transcription_runs WHERE status IN ('PROCESSING', 'CREATED')")
        for row in rows:
            self.db.execute(
                "UPDATE transcription_runs SET status='FAILED', error_code='APP_INTERRUPTED', "
                "error_message='应用异常退出，任务被中断', finished_at=? WHERE id=?",
                (now_iso(), row["id"]),
            )
        return len(rows)

    # ------------------------------------------------------------------
    # 单任务互斥检查
    # ------------------------------------------------------------------
    def _assert_no_active(self) -> None:
        row = self.db.query_one(
            "SELECT id FROM transcription_runs WHERE status = 'PROCESSING'")
        if row:
            raise TaskError(f"已有任务在执行中（{row['id']}），全局同一时刻仅允许一个转写任务")

    # ------------------------------------------------------------------
    # 创建并启动任务
    # ------------------------------------------------------------------
    def start_run(self, project_id: str, model: str = DEFAULT_ASR_MODEL,
                  fast: bool = False) -> str:
        project = self.db.query_one("SELECT * FROM projects WHERE id=?", (project_id,))
        if not project:
            raise TaskError("项目不存在")
        # 模型选择在任务创建时锁定：校验键与目录，避免跑到一半才失败
        try:
            model_dir = asr_model_dir(self.cfg, model)
        except ValueError as exc:
            raise TaskError(str(exc)) from exc
        if not is_model_installed(model_dir):
            raise TaskError(
                f"模型未安装或下载未完成：{model_dir.name}"
                f"（请下载到 {model_dir} 后重试）")

        with self._mutex:
            self._assert_no_active()
            run_id = new_id("run")
            work_dir = self.cfg.paths.temp / run_id
            output_dir = self.cfg.paths.projects / project_id / "runs" / run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            self.db.execute(
                "INSERT INTO transcription_runs (id, project_id, status, stage, model_name, "
                "work_dir, started_at) VALUES (?, ?, 'PROCESSING', 'VALIDATING', ?, ?, ?)",
                (run_id, project_id, model_dir.name, str(work_dir), now_iso()),
            )

        thread = threading.Thread(
            target=self._execute,
            args=(run_id, project_id, work_dir, output_dir, model_dir, fast),
            name=f"task-{run_id}", daemon=True,
        )
        thread.start()
        return run_id

    # ------------------------------------------------------------------
    # 取消（§8.3：协作式取消，终止子进程后标记 CANCELLED）
    # ------------------------------------------------------------------
    def cancel_run(self, run_id: str) -> None:
        run = self.get_run(run_id)
        if not run:
            raise TaskError("任务不存在")
        if run["status"] != "PROCESSING":
            raise TaskError(f"任务已处于终态（{run['status']}），无需取消")
        with self._cancel_lock:
            event = self._cancel_events.get(run_id)
        if event is None:
            # 竞态：任务恰好在查询后结束
            raise TaskError("任务刚已结束，取消未生效")
        event.set()

    def _relay_event(self, run_id: str, event: dict) -> None:
        """转发子进程原始事件，并翻译出前端语义事件（阶段/进度）。"""
        self.hub.publish({"runId": run_id, **event})
        kind = str(event.get("kind", ""))
        mapped = _STAGE_BY_KIND.get(kind)
        if mapped is not None:
            db_stage, phase = mapped
            self.db.execute(
                "UPDATE transcription_runs SET stage=? WHERE id=?", (db_stage, run_id))
            self.hub.publish({"runId": run_id, "kind": "phase_updated", "phase": phase})
        elif kind == "asr_window":
            # 长音频 vad 模式：每个推理窗口一条进度（短音频 full 模式无此事件，
            # 进度从 0% 直接到完成，属预期）
            total = max(int(event.get("total", 1)), 1)
            percent = round(100.0 * int(event.get("index", 0)) / total, 1)
            self.db.execute(
                "UPDATE transcription_runs SET progress_percent=? WHERE id=?",
                (percent, run_id))
            self.hub.publish({"runId": run_id, "kind": "progress_updated", "percent": percent})

    def _execute(self, run_id: str, project_id: str, work_dir: Path,
                 output_dir: Path, model_dir: Path, fast: bool = False) -> None:
        """后台线程：驱动编排层（子进程执行 ASR + 后处理），全程发 SSE 事件。"""
        project = self.db.query_one("SELECT * FROM projects WHERE id=?", (project_id,))
        pcm = self.cfg.paths.projects / project_id / "audio" / project["pcm_path"]
        final_path = output_dir / "final.txt"
        cancel_event = threading.Event()
        with self._cancel_lock:
            self._cancel_events[run_id] = cancel_event
        try:
            # 句级跳转依赖字符级时间戳；快速模式跳过对齐（省 2–6s），
            # 句级时间戳退回按字数插值；模型缺失时同样自动降级。
            aligner_dir = self.cfg.paths.models / "Qwen3-ForcedAligner-0.6B"
            # 长音频（>10min，V3.0 §5）用 DP 全局切分：一次性求「块长偏离 + 硬切惩罚
            # − 静音奖励」总代价最小。60min 实测静音吸附率 43%→64%、硬切 33→20，
            # 代价是块长标准差 4.4s→8.5s（batch 仍可用）。短音频块数少，balanced 更稳。
            duration_ms = int(project["duration_ms"] or 0)
            chunk_strategy = "dp" if duration_ms > 10 * 60 * 1000 else "balanced"
            batch_size, batch_max = batch_for_model(model_dir)
            target_chunk = chunk_seconds_for_model(model_dir)
            options = OrchestrationOptions(
                audio=pcm,
                output=final_path,
                model=model_dir,
                device="cuda",
                language="Chinese",
                correction="none",
                enable_itn=True,
                work_dir=work_dir,
                overwrite=True,
                segmentation="auto",
                batch_size=batch_size,
                batch_auto=False,
                batch_max=batch_max,
                chunk_strategy=chunk_strategy,
                target_chunk=target_chunk,
                # 重叠 1.5s：0.5s 时块边界词被截断（「至」/「然」），
                # 且重复内容不足 4 字无法去重（「英国」/「英国人」）
                overlap=1.5,
                vad_model=self.cfg.paths.models / "funasr-fsmn-vad",
                aligner=aligner_dir if aligner_dir.is_dir() and not fast else None,
                # 第二模型交叉验证（FunASR Paraformer）：两模型不一致的字符作为
                # 独立第二档标亮，bench 召回 29%→66%、精度持平。FunASR 子进程
                # 有约 48s 固定开销（加载 26s + 推理 16s），短音频占比过高，
                # 故仅对长音频（≥10min）启用；模型缺失或失败时脚本内部降级。
                cross_check="auto" if duration_ms >= 10 * 60 * 1000 else None,
            )
            # Windows 单 venv：ASR 与后处理共用当前解释器（V3.0 §2.4 跨 venv 能力保留，
            # 环境显式注入避免依赖目录探测）
            py = Path(sys.executable)
            env = PythonEnv(
                name="main",
                python=py,
                directory=py.parent.parent,
                versions=probe_modules(py, ("torch", "transformers", "wetext")),
            )
            result: OrchestrationResult = run(
                options,
                envs={"funasr": env, "mlx": env},
                scripts_dir=self.cfg.paths.scripts,
                on_event=lambda e: self._relay_event(run_id, e),
                cancel=cancel_event,
                asr_executor=self._asr_worker,
            )
            # 构建片段（stats.json：chunk 时间 + 去重后文本）
            stats_path = Path(result.raw_text_path).with_suffix(".stats.json")
            segments = build_segments_from_stats(stats_path)
            save_segments(self.db, run_id, segments)
            self.db.execute(
                "UPDATE transcription_runs SET status='COMPLETED', stage='SAVING', "
                "progress_percent=100, finished_at=? WHERE id=?",
                (now_iso(), run_id),
            )
            self.hub.publish({
                "runId": run_id, "kind": "task_completed",
                "finalTextPath": str(final_path), "segments": len(segments),
            })
        except ProcessCancelled:
            # 用户取消（§8.3）：标记 CANCELLED 而非 FAILED
            self.db.execute(
                "UPDATE transcription_runs SET status='CANCELLED', "
                "error_code='USER_CANCELLED', finished_at=? WHERE id=?",
                (now_iso(), run_id),
            )
            self.hub.publish({"runId": run_id, "kind": "task_cancelled"})
        except Exception as exc:  # 失败隔离（§8.5）：记录错误阶段与可读信息
            self.db.execute(
                "UPDATE transcription_runs SET status='FAILED', error_code='RUN_FAILED', "
                "error_message=?, finished_at=? WHERE id=?",
                (str(exc)[:500], now_iso(), run_id),
            )
            self.hub.publish({
                "runId": run_id, "kind": "task_failed", "message": str(exc)[:500],
            })
        finally:
            with self._cancel_lock:
                self._cancel_events.pop(run_id, None)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_run(self, run_id: str) -> dict | None:
        row = self.db.query_one("SELECT * FROM transcription_runs WHERE id=?", (run_id,))
        return dict(row) if row else None

    def current_run(self) -> dict | None:
        row = self.db.query_one(
            "SELECT * FROM transcription_runs WHERE status='PROCESSING' ORDER BY started_at DESC")
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # 应用退出：停止常驻 worker（释放显存）
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._asr_worker.close()
