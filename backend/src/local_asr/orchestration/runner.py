"""跨解释器任务编排。

链路：ASR（.venv-funasr）→ 纠错/校验/ITN（.venv-mlx）→ 最终产物。

编排层只做三件事：
1. 按阶段启动子进程，把每一步的输入输出接到磁盘上的确定路径；
2. 把子进程的结构化事件翻译成统一事件，交给 CLI / GUI；
3. 在产出最终文本前校验上一阶段产物完整，失败即停止，不猜测续跑。

它**不**导入 torch / mlx_lm，也不把两者加载进同一进程。
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from local_asr.core.events import EventBus
from local_asr.core.models import EventType
from local_asr.orchestration import environment
from local_asr.orchestration.environment import EnvError, PythonEnv
from local_asr.orchestration.process_manager import ProcessError, StepResult, run_step
from local_asr.orchestration.task_state import TaskState, make_task_id
from local_asr.postprocess.artifacts import atomic_write_text

__all__ = [
    "OrchestrationError",
    "OrchestrationOptions",
    "OrchestrationResult",
    "run",
]

CorrectionMode = Literal["none", "mlx"]
SegmentationMode = Literal["auto", "full", "vad"]

# 后处理子进程事件 -> 编排事件
_POSTPROCESS_EVENT_MAP = {
    "correction_completed": EventType.CORRECTION_COMPLETED,
    "verify_completed": EventType.VERIFY_COMPLETED,
    "itn_completed": EventType.ITN_COMPLETED,
}


class OrchestrationError(RuntimeError):
    """编排链路失败。``stage`` 指出失败阶段。"""

    def __init__(self, message: str, *, stage: str, detail: str = "") -> None:
        super().__init__(message)
        self.stage = stage
        self.detail = detail


@dataclass(frozen=True)
class OrchestrationOptions:
    """一次完整转写任务的全部输入。"""

    audio: Path
    output: Path
    model: Path
    device: str = "mps"
    language: str = "Chinese"
    correction: CorrectionMode = "none"
    corrector_model: Path | None = None
    enable_itn: bool = True
    glossary: Path | None = None
    export_intermediates: bool = False
    work_dir: Path | None = None
    overwrite: bool = False
    segmentation: SegmentationMode = "auto"
    batch_size: int = 1
    batch_auto: bool = False
    batch_max: int = 3
    chunk_strategy: str = "silence"
    target_chunk: float = 120.0
    balance_tolerance: float = 0.15
    # 每块两侧扩展的重叠秒数。V3.0 设计为 0.5s，但实测 0.5s 时块边界的
    # 词常被截断、且重复内容 < 4 字无法去重（「英国」/「英国人」）；
    # 1.5s 让下一块完整听到边界词，去重记录从 2 条增至 9 条、边界通顺。
    overlap: float = 1.5
    asr_timeout: float | None = None
    postprocess_timeout: float | None = None
    vad_model: Path | None = None  # FunASR FSMN-VAD 模型目录；None 用脚本默认
    aligner: Path | None = None  # Qwen3-ForcedAligner 目录；None 不产出字符时间
    # 第二模型交叉验证："auto"=现场跑 FunASR Paraformer，或传已生成的文本路径；
    # None 关闭。不一致字符在低置信打分中加分（bench 召回约翻倍）。
    cross_check: str | None = None


@dataclass(frozen=True)
class OrchestrationResult:
    """编排结果。"""

    task_id: str
    status: Literal["complete", "degraded"]
    final_text_path: Path
    raw_text_path: Path
    work_dir: Path
    stats: dict = field(default_factory=dict)
    events: tuple = field(default_factory=tuple)


def _validate(options: OrchestrationOptions) -> None:
    if not options.audio.is_file():
        raise OrchestrationError(f"音频不存在: {options.audio}", stage="preflight")
    if not options.model.is_dir():
        raise OrchestrationError(f"ASR 模型目录不存在: {options.model}", stage="preflight")
    if options.correction not in ("none", "mlx"):
        raise OrchestrationError(f"未知纠错模式: {options.correction}", stage="preflight")
    if options.correction == "mlx":
        if options.corrector_model is None:
            raise OrchestrationError("correction=mlx 需要 corrector_model", stage="preflight")
        if not options.corrector_model.is_dir():
            raise OrchestrationError(
                f"纠错模型目录不存在: {options.corrector_model}", stage="preflight")
    if options.segmentation not in ("auto", "full", "vad"):
        raise OrchestrationError(f"未知切分模式: {options.segmentation}", stage="preflight")
    if options.batch_size < 1:
        raise OrchestrationError(f"batch_size 必须 >= 1: {options.batch_size}", stage="preflight")
    if options.batch_max < 1:
        raise OrchestrationError(f"batch_max 必须 >= 1: {options.batch_max}", stage="preflight")
    if options.chunk_strategy not in ("silence", "balanced", "dp"):
        raise OrchestrationError(
            f"未知切片策略: {options.chunk_strategy}（silence / balanced / dp）",
            stage="preflight")
    if options.target_chunk <= 0:
        raise OrchestrationError(f"target_chunk 必须为正数: {options.target_chunk}", stage="preflight")
    if not 0 <= options.balance_tolerance < 0.5:
        raise OrchestrationError(
            f"balance_tolerance 必须满足 0 <= tol < 0.5: {options.balance_tolerance}",
            stage="preflight")
    if not 0 <= options.overlap <= 5:
        raise OrchestrationError(
            f"overlap 必须满足 0 <= overlap <= 5: {options.overlap}", stage="preflight")
    if options.batch_auto and options.batch_size != 1:
        raise OrchestrationError(
            "batch_auto 与 batch_size 互斥；自动模式下批大小由内存护栏决定",
            stage="preflight")
    if options.glossary is not None and not options.glossary.is_file():
        raise OrchestrationError(f"词典文件不存在: {options.glossary}", stage="preflight")
    if options.output.exists() and not options.overwrite:
        raise OrchestrationError(
            f"输出已存在: {options.output}；使用 --overwrite 覆盖", stage="preflight")
    if options.aligner is not None and not options.aligner.is_dir():
        raise OrchestrationError(
            f"对齐模型目录不存在: {options.aligner}", stage="preflight")


def _asr_argv(options: OrchestrationOptions, scripts: Path, raw_path: Path) -> list[str]:
    argv = [
        str(scripts / "qwen3_asr_transcribe.py"),
        "--model", str(options.model),
        "--audio", str(options.audio),
        "--output", str(raw_path),
        "--device", options.device,
        "--language", options.language,
        "--segmentation", options.segmentation,
        "--batch-size", str(options.batch_size),
        "--chunk-strategy", options.chunk_strategy,
        "--target-chunk", str(options.target_chunk),
        "--balance-tolerance", str(options.balance_tolerance),
        "--overlap", str(options.overlap),
        "--batch-max", str(options.batch_max),
    ]
    if options.batch_auto:
        argv.append("--batch-auto")
    if options.vad_model is not None:
        argv += ["--vad-model", str(options.vad_model)]
    if options.aligner is not None:
        # 同时产出字符级时间戳（stats.char_times）与对齐置信特征
        argv += ["--forced-aligner", str(options.aligner)]
    if options.cross_check:
        argv += ["--cross-check", options.cross_check]
    return argv


def _postprocess_argv(options: OrchestrationOptions, scripts: Path, raw_path: Path,
                      out_dir: Path) -> list[str]:
    argv = [
        str(scripts / "qwen3_asr_postprocess.py"),
        "--input", str(raw_path),
        "--output-dir", str(out_dir),
        "--stem", "result",
        "--corrector", options.correction,
    ]
    if options.correction == "mlx" and options.corrector_model is not None:
        argv += ["--corrector-model", str(options.corrector_model)]
    if not options.enable_itn:
        argv.append("--no-itn")
    if options.glossary is not None:
        argv += ["--glossary", str(options.glossary)]
    return argv


def _publish_output(path: Path, text: str, *, overwrite: bool) -> None:
    """把最终文本原子发布到用户指定路径。

    复用 :func:`~local_asr.postprocess.artifacts.atomic_write_text`：
    同目录暂存 + fsync + ``os.replace`` 落位；``overwrite=False`` 且
    目标已存在时抛 FileExistsError（preflight 已提前拦截，这里是兜底，
    防止 preflight 与发布之间的竞态窗口内文件被创建）。
    """
    atomic_write_text(path, text, overwrite=overwrite)


def _read_stats(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _merge_stats(asr: dict, post: dict) -> dict:
    """合并两阶段统计，保留各自作用域，不做同名字段覆盖式合并。"""
    return {
        "schema_version": 1,
        "asr": asr,
        "postprocess": post,
        "postprocess_stats": post.get("stats") if isinstance(post.get("stats"), dict) else None,
    }


def run(
    options: OrchestrationOptions,
    *,
    event_bus: EventBus | None = None,
    envs: dict[str, PythonEnv] | None = None,
    scripts_dir: Path | None = None,
    on_event: Callable[[dict], None] | None = None,
    cancel: "threading.Event | None" = None,
    asr_executor: Callable[..., StepResult] | None = None,
) -> OrchestrationResult:
    """执行完整链路。异常时已发布 TASK_FAILED。

    cancel: 协作式取消事件（§8.3）；触发后终止当前子进程并抛
    ``ProcessCancelled``（不发布 TASK_FAILED，由调用方标记 CANCELLED）。
    asr_executor: 可选的 ASR 执行器，签名与 ``run_step`` 一致；服务层
    注入常驻 worker 以复用已加载的模型权重。默认每次新建子进程。
    """
    _validate(options)
    scripts = Path(scripts_dir) if scripts_dir is not None else environment.project_root() / "scripts"

    task_id = make_task_id()
    work_dir = Path(options.work_dir) if options.work_dir is not None else Path("work") / task_id / "qwen3"
    asr_dir, post_dir = work_dir / "asr", work_dir / "postprocess"
    raw_path = asr_dir / "raw.txt"
    asr_stats_path = raw_path.with_suffix(".stats.json")

    state = TaskState(task_id, event_bus)
    started = time.perf_counter()

    def _forward(event: dict) -> None:
        if on_event is not None:
            on_event(event)

    try:
        state.emit(
            EventType.TASK_STARTED,
            input_path=str(options.audio),
            output_path=str(options.output),
            correction=options.correction,
            itn=options.enable_itn,
        )
        state.phase("preflight", "正在检查运行环境...")

        envs = envs if envs is not None else environment.discover()
        funasr_env = environment.ensure(
            envs, environment.ENV_FUNASR, ("torch", "transformers"),
            "ASR 识别", root=environment.project_root())
        mlx_modules = ("wetext", "mlx_lm") if options.correction == "mlx" else ("wetext",)
        mlx_env = environment.ensure(
            envs, environment.ENV_MLX, mlx_modules, "文本后处理",
            root=environment.project_root())

        # ---- 阶段 1：ASR ----
        state.phase("transcribing", "正在识别音频...")
        state.emit(EventType.ASR_STARTED, model=str(options.model), device=options.device)

        def _on_asr_event(event: dict) -> None:
            _forward(event)
            if event.get("kind") == "asr_window":
                percent = 100.0 * int(event.get("index", 0)) / max(int(event.get("total", 1)), 1)
                state.emit(EventType.PROGRESS_UPDATED, percent=round(percent, 1),
                           completed_chunks=int(event.get("index", 0)),
                           total_chunks=int(event.get("total", 1)))

        execute_asr = asr_executor or run_step
        asr_step = execute_asr(
            funasr_env.python, _asr_argv(options, scripts, raw_path),
            name="asr", on_event=_on_asr_event,
            timeout=options.asr_timeout, cwd=environment.project_root(),
            cancel=cancel,
        )
        if not asr_step.ok:
            raise OrchestrationError(
                f"ASR 退出码 {asr_step.returncode}",
                stage="asr", detail=_tail(asr_step))
        if not raw_path.is_file():
            raise OrchestrationError(f"ASR 未产出原始文本: {raw_path}", stage="asr")
        raw_text = raw_path.read_text(encoding="utf-8")
        if not raw_text.strip():
            raise OrchestrationError("ASR 输出为空，停止后续处理", stage="asr")
        asr_stats = _read_stats(asr_stats_path)
        state.emit(EventType.ASR_COMPLETED, elapsed_seconds=asr_step.elapsed_seconds,
                   chars=len(raw_text), raw_path=str(raw_path))

        # ---- 阶段 2：纠错 / 校验 / ITN ----
        state.phase("postprocessing", "正在纠错与规范化数字...")
        state.emit(EventType.CORRECTION_STARTED, corrector=options.correction)

        def _on_post_event(event: dict) -> None:
            _forward(event)
            mapped = _POSTPROCESS_EVENT_MAP.get(str(event.get("kind")))
            if mapped is None:
                return
            payload = {k: v for k, v in event.items() if k != "kind"}
            state.emit(mapped, **payload)

        post_step = run_step(
            mlx_env.python, _postprocess_argv(options, scripts, raw_path, post_dir),
            name="postprocess", on_event=_on_post_event,
            timeout=options.postprocess_timeout, cwd=environment.project_root(),
            cancel=cancel,
        )
        if post_step.returncode not in (0, 3):
            raise OrchestrationError(
                f"后处理退出码 {post_step.returncode}",
                stage="postprocess", detail=_tail(post_step))
        final_path = post_dir / "result_final.txt"
        if not final_path.is_file():
            raise OrchestrationError(f"后处理未产出最终文本: {final_path}", stage="postprocess")
        final_text = final_path.read_text(encoding="utf-8")
        post_stats = _read_stats(post_dir / "result_postprocess.json")

        # ---- 阶段 3：发布与校验 ----
        _publish_output(options.output, final_text, overwrite=options.overwrite)
        if options.export_intermediates:
            _export(post_dir, options.output, overwrite=options.overwrite)

        stats = _merge_stats(asr_stats, post_stats)
        stats.update({
            "task_id": task_id,
            "stage": "orchestration",
            "status": "degraded" if post_step.returncode == 3 else "complete",
            "work_dir": str(work_dir),
            "raw_text_path": str(raw_path),
            "final_text_path": str(options.output),
            "correction": options.correction,
            "itn_enabled": options.enable_itn,
            "envs": {name: env.describe() for name, env in envs.items()},
            "asr_step": _step_stats(asr_step),
            "postprocess_step": _step_stats(post_step),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "elapsed_scope": "orchestration_end_to_end",
        })
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "orchestration.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

        degraded = post_step.returncode == 3
        if degraded:
            reasons = post_stats.get("degradations") or []
            state.degraded(reasons, output_path=str(options.output))
        else:
            state.completed(output_path=str(options.output), chars=len(final_text))
        return OrchestrationResult(
            task_id=task_id,
            status="degraded" if degraded else "complete",
            final_text_path=Path(options.output),
            raw_text_path=raw_path,
            work_dir=work_dir,
            stats=stats,
            events=state.events,
        )
    except OrchestrationError as exc:
        _fail(state, str(exc), exc.stage)
        raise
    except (EnvError, ProcessError) as exc:
        _fail(state, str(exc), getattr(exc, "stage", "environment"))
        raise


def _fail(state: TaskState, message: str, stage: str) -> None:
    if state.is_terminal:
        return
    try:
        state.failed(message, stage=stage)
    except RuntimeError:
        pass


def _tail(step: StepResult, limit: int = 20) -> str:
    lines = [line for line in (step.stderr_tail or step.stdout_tail).splitlines() if line.strip()]
    return "\n".join(lines[-limit:])


def _step_stats(step: StepResult) -> dict:
    return {
        "name": step.name,
        "returncode": step.returncode,
        "elapsed_seconds": step.elapsed_seconds,
        "event_count": len(step.events),
    }


_EXPORT_FILES = {
    "raw": ("result_raw.txt", "_raw.txt"),
    "corrected": ("result_corrected.txt", "_corrected.txt"),
    "final": ("result_final.txt", "_final.txt"),
    "edits": ("result_edits.jsonl", "_edits.jsonl"),
    "stats": ("result_postprocess.json", "_postprocess.json"),
}


def _export(post_dir: Path, output: Path, *, overwrite: bool) -> None:
    """把中间产物复制到输出目录旁边，便于人工复核。"""
    for source_name, suffix in _EXPORT_FILES.values():
        source = post_dir / source_name
        if not source.is_file():
            continue
        target = output.with_name(output.stem + suffix)
        if target.resolve() == output.resolve():
            continue
        if target.exists() and not overwrite:
            raise FileExistsError(f"中间产物已存在: {target}")
        shutil.copy2(source, target)