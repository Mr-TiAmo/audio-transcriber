"""统一编排的 Typer 命令。

两个入口共用这一份实现：

- ``scripts/local_asr.py transcribe ...``
- ``local-asr qwen3 transcribe ...``

参数刻意拆成三个互不耦合的开关：
``--semantic-correction`` 决定是否纠错、``--itn/--no-itn`` 决定是否规范化数字、
``--export-intermediates`` 决定是否把中间产物复制到输出目录旁边。
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from local_asr.core.events import EventBus
from local_asr.core.models import EventType, TaskEvent
from local_asr.orchestration.runner import (
    OrchestrationError,
    OrchestrationOptions,
    OrchestrationResult,
    run,
)

app = typer.Typer(
    name="qwen3",
    help="Qwen3-ASR 统一编排：识别 + 可选语义纠错 + 数字规范化",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _root() -> None:
    """Qwen3-ASR 统一编排：跨 venv 调度识别、纠错与数字规范化。

    没有这个回调时，Typer 会把唯一的 ``transcribe`` 命令折叠成默认命令，
    ``local_asr.py transcribe --audio ...`` 中的 ``transcribe`` 会被当成音频路径。
    """


# 事件 -> 控制台文案
_EVENT_LABELS = {
    EventType.TASK_STARTED: "任务开始",
    EventType.ASR_STARTED: "识别中",
    EventType.ASR_COMPLETED: "识别完成",
    EventType.CORRECTION_STARTED: "语义纠错中",
    EventType.CORRECTION_COMPLETED: "纠错完成",
    EventType.VERIFY_COMPLETED: "修改校验完成",
    EventType.ITN_COMPLETED: "数字规范化完成",
    EventType.TASK_COMPLETED: "任务完成",
    EventType.TASK_DEGRADED: "任务降级完成",
    EventType.TASK_FAILED: "任务失败",
}


def _renderer(quiet: bool):
    """构造事件渲染回调；quiet 模式下只保留终态。"""
    def _render(event: TaskEvent) -> None:
        label = _EVENT_LABELS.get(event.event_type)
        if label is None:
            return
        if quiet and event.event_type not in (
            EventType.TASK_COMPLETED, EventType.TASK_DEGRADED, EventType.TASK_FAILED,
        ):
            return
        detail = ""
        payload = event.payload
        if event.event_type == EventType.ASR_COMPLETED:
            detail = f"（{payload.get('chars', 0)} 字，{payload.get('elapsed_seconds', 0):.1f}s）"
        elif event.event_type == EventType.VERIFY_COMPLETED:
            detail = (f"（接受 {payload.get('edits_accepted', 0)} 条，"
                      f"复核 {payload.get('edits_review', 0)} 条）")
        elif event.event_type == EventType.ITN_COMPLETED:
            detail = f"（{payload.get('elapsed_seconds', 0):.1f}s）"
        elif event.event_type == EventType.TASK_DEGRADED:
            detail = f"：{'; '.join(payload.get('reasons') or [])}"
        elif event.event_type == EventType.TASK_FAILED:
            detail = f"（{payload.get('stage', '?')}）: {payload.get('error', '')}"
        style = {
            EventType.TASK_FAILED: "red",
            EventType.TASK_DEGRADED: "yellow",
            EventType.TASK_COMPLETED: "green",
        }.get(event.event_type, "cyan")
        console.print(f"[{style}]\\[{label}]{detail}[/{style}]")
    return _render


def _options(
    audio: str,
    output: str,
    model: str,
    device: str,
    language: str,
    segmentation: str,
    batch_size: int,
    batch_auto: bool,
    batch_max: int,
    chunk_strategy: str,
    target_chunk: float,
    balance_tolerance: float,
    overlap: float,
    semantic_correction: str,
    corrector_model: str | None,
    itn: bool,
    glossary: str | None,
    export_intermediates: bool,
    work_dir: str | None,
    overwrite: bool,
    cross_check: str | None,
) -> OrchestrationOptions:
    if semantic_correction not in ("none", "mlx"):
        raise typer.BadParameter("--semantic-correction 只能是 none 或 mlx")
    if segmentation not in ("auto", "full", "vad"):
        raise typer.BadParameter("--segmentation 只能是 auto、full 或 vad")
    if batch_size < 1:
        raise typer.BadParameter("--batch-size 必须 >= 1")
    if batch_auto and batch_size != 1:
        raise typer.BadParameter("--batch-auto 与 --batch-size 互斥")
    if batch_max < 1:
        raise typer.BadParameter("--batch-max 必须 >= 1")
    if chunk_strategy not in ("silence", "balanced", "dp"):
        raise typer.BadParameter("--chunk-strategy 只能是 silence、balanced 或 dp")
    if target_chunk <= 0:
        raise typer.BadParameter("--target-chunk 必须为正数")
    if not 0 <= balance_tolerance < 0.5:
        raise typer.BadParameter("--balance-tolerance 必须满足 0 <= tol < 0.5")
    if not 0 <= overlap <= 5:
        raise typer.BadParameter("--overlap 必须满足 0 <= overlap <= 5")
    return OrchestrationOptions(
        audio=Path(audio).resolve(),
        output=Path(output).resolve(),
        model=Path(model).resolve(),
        device=device,
        language=language,
        segmentation=segmentation,  # type: ignore[arg-type]
        batch_size=batch_size,
        batch_auto=batch_auto,
        batch_max=batch_max,
        chunk_strategy=chunk_strategy,
        target_chunk=target_chunk,
        balance_tolerance=balance_tolerance,
        overlap=overlap,
        correction=semantic_correction,  # type: ignore[arg-type]
        corrector_model=Path(corrector_model).resolve() if corrector_model else None,
        enable_itn=itn,
        glossary=Path(glossary).resolve() if glossary else None,
        export_intermediates=export_intermediates,
        work_dir=Path(work_dir).resolve() if work_dir else None,
        overwrite=overwrite,
        # auto 原样透传（由 ASR 脚本现场调用第二模型）；文件路径转绝对路径，
        # 子进程 cwd 与 CLI 不同也能找到
        cross_check=(
            None if cross_check is None
            else "auto" if cross_check == "auto"
            else str(Path(cross_check).resolve())
        ),
    )


@app.command()
def transcribe(
    audio: str = typer.Option(..., "--audio", "-a", help="音频文件路径"),
    output: str = typer.Option("./output/result.txt", "--output", "-o", help="最终文本路径"),
    model: str = typer.Option("models/Qwen3-ASR-1.7B-hf", "--model", "-m", help="ASR 模型目录"),
    device: str = typer.Option("mps", "--device", "-d", help="mps / cpu / cuda"),
    language: str = typer.Option("Chinese", "--language", "-l", help="语言提示"),
    segmentation: str = typer.Option(
        "auto", "--segmentation",
        help="长音频切分策略：auto（按时长自动）/ full（整段识别）/ vad（按语音边界切分）"),
    batch_size: int = typer.Option(
        1, "--batch-size", help="每批识别的块数；默认 1 即逐条，产物可复现"),
    batch_auto: bool = typer.Option(
        False, "--batch-auto", help="按块长分桶并自动决定批大小（受内存护栏限制）"),
    batch_max: int = typer.Option(
        3, "--batch-max", help="自动分桶的批大小上限（16 GB 机器实测 4）"),
    chunk_strategy: str = typer.Option(
        "silence", "--chunk-strategy",
        help="切片策略：silence（最长静音优先）/ balanced（先均分块数再吸附静音）"
             "/ dp（动态规划全局最优，长音频推荐）"),
    target_chunk: float = typer.Option(
        120.0, "--target-chunk", help="目标块长（秒）；balanced 下决定块数"),
    balance_tolerance: float = typer.Option(
        0.15, "--balance-tolerance", help="balanced 切点偏离理想位置的比例；0 表示严格等长"),
    overlap: float = typer.Option(
        1.5, "--overlap",
        help="每块两侧扩展的重叠秒数；1.5 可避免块边界切词（设计原值 0.5）"),
    semantic_correction: str = typer.Option(
        "none", "--semantic-correction",
        help="语义纠错：none（只做数字规范化）/ mlx（本地 CEC4）"),
    corrector_model: str | None = typer.Option(
        None, "--corrector-model", help="纠错模型目录，--semantic-correction mlx 时必填"),
    itn: bool = typer.Option(True, "--itn/--no-itn", help="是否执行数字规范化"),
    glossary: str | None = typer.Option(None, "--glossary", help="实体词典 JSON 路径"),
    export_intermediates: bool = typer.Option(
        False, "--export-intermediates", help="把 raw/corrected/edits 复制到输出目录旁边"),
    work_dir: str | None = typer.Option(None, "--work-dir", help="中间产物目录"),
    overwrite: bool = typer.Option(False, "--overwrite", help="覆盖已存在的输出"),
    cross_check: str | None = typer.Option(
        None, "--cross-check",
        help="第二模型交叉验证：auto=现场调用 FunASR Paraformer（模型缺失时自动降级）；"
             "也可传已生成的第二模型文本路径"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="仅输出最终路径"),
) -> None:
    """识别音频并按需纠错、规范化数字。"""
    options = _options(
        audio, output, model, device, language, segmentation, batch_size, batch_auto,
        batch_max, chunk_strategy, target_chunk, balance_tolerance, overlap,
        semantic_correction, corrector_model, itn, glossary, export_intermediates,
        work_dir, overwrite, cross_check,
    )
    bus = EventBus()
    bus.subscribe(_renderer(quiet))
    try:
        result = run(options, event_bus=bus)
    except OrchestrationError as exc:
        console.print(f"[red]失败（{exc.stage}）: {exc}[/red]")
        if exc.detail:
            console.print(exc.detail, style="dim")
        raise typer.Exit(code=1)
    except Exception as exc:  # 环境探测等非编排异常
        console.print(f"[red]失败: {exc}[/red]")
        raise typer.Exit(code=1)
    _print_result(result, quiet)


def _print_result(result: OrchestrationResult, quiet: bool) -> None:
    if quiet:
        console.print(str(result.final_text_path))
        return
    post = result.stats.get("postprocess", {}) or {}
    console.print(f"[dim]任务: {result.task_id}[/dim]")
    console.print(f"[dim]中间产物: {result.work_dir}[/dim]")
    if post:
        console.print(
            f"[dim]字符: raw {post.get('raw_chars', 0)} → "
            f"corrected {post.get('corrected_chars', 0)} → final {post.get('final_chars', 0)}[/dim]")
    console.print(f"[green]输出: {result.final_text_path}[/green]")


def main() -> None:
    """``scripts/local_asr.py`` 的入口。"""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()