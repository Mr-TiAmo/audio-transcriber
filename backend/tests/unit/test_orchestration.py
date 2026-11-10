"""跨解释器编排层测试。

全部用假子进程（``sys.executable`` + 临时脚本）验证调度逻辑，
不加载 torch / mlx_lm，也不依赖真实模型。
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from local_asr.core.events import EventBus
from local_asr.core.models import EventType
from local_asr.orchestration import environment
from local_asr.orchestration.environment import EnvError, PythonEnv
from local_asr.orchestration.process_manager import ProcessError, run_step
from local_asr.orchestration.runner import OrchestrationError, OrchestrationOptions, run
from local_asr.orchestration.task_state import TaskState


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

def test_probe_reports_missing_module_as_none():
    versions = environment.probe_modules(Path(sys.executable), ("json", "definitely_not_installed_pkg"))
    assert versions["json"] is not None
    assert versions["definitely_not_installed_pkg"] is None


def test_discover_returns_empty_without_expected_venvs(tmp_path):
    assert environment.discover(tmp_path) == {}


def test_ensure_rejects_missing_environment(tmp_path):
    with pytest.raises(EnvError, match="funasr"):
        environment.ensure({}, environment.ENV_FUNASR, ("torch",), "ASR 识别", root=tmp_path)


def test_require_lists_missing_modules():
    env = PythonEnv("mlx", Path(sys.executable), Path("/tmp"), {"wetext": "0.2.1", "mlx_lm": None})
    with pytest.raises(EnvError, match="mlx_lm"):
        env.require(("wetext", "mlx_lm"), "文本后处理")


# ---------------------------------------------------------------------------
# process_manager
# ---------------------------------------------------------------------------

def _write_script(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_run_step_splits_events_from_logs(tmp_path):
    script = _write_script(tmp_path / "emit.py", """
        import json, sys
        print("普通日志")
        print("@@EVENT@@ " + json.dumps({"kind": "step_a", "value": 1}))
        print("@@EVENT@@ not-json")
        print("@@EVENT@@ " + json.dumps({"kind": "step_b"}))
        print("尾部日志")
    """)
    seen = []
    result = run_step(Path(sys.executable), [str(script)], name="demo", on_event=seen.append)
    assert result.ok
    assert [e["kind"] for e in result.events] == ["step_a", "step_b"]
    assert [e["kind"] for e in seen] == ["step_a", "step_b"]
    assert "普通日志" in result.stdout_tail
    assert "尾部日志" in result.stdout_tail
    assert "not-json" not in {e.get("kind") for e in result.events}


def test_run_step_keeps_nonzero_exit_for_caller(tmp_path):
    script = _write_script(tmp_path / "fail.py", """
        import sys
        print("boom", file=sys.stderr)
        sys.exit(3)
    """)
    result = run_step(Path(sys.executable), [str(script)], name="demo")
    assert result.returncode == 3
    assert not result.ok
    assert "boom" in result.stderr_tail


def test_run_step_times_out(tmp_path):
    script = _write_script(tmp_path / "slow.py", """
        import time
        time.sleep(5)
    """)
    with pytest.raises(ProcessError, match="超过"):
        run_step(Path(sys.executable), [str(script)], name="slow", timeout=0.5)


def test_run_step_cancel_terminates_subprocess(tmp_path):
    """协作式取消（§8.3）：推理期间无输出也要能终止子进程并抛 ProcessCancelled。"""
    import threading
    import time as _time

    from local_asr.orchestration.process_manager import ProcessCancelled

    script = _write_script(tmp_path / "sleep.py", """
        import time
        time.sleep(30)
    """)
    cancel = threading.Event()
    started = _time.perf_counter()
    with pytest.raises(ProcessCancelled, match="取消"):
        # 后台线程延迟触发，模拟运行中用户点击取消
        threading.Timer(0.5, cancel.set).start()
        run_step(Path(sys.executable), [str(script)], name="cancel-demo", cancel=cancel)
    assert _time.perf_counter() - started < 10


def test_run_step_cancel_after_finish_is_noop(tmp_path):
    """步骤正常结束后才触发取消：不得误抛 ProcessCancelled。"""
    import threading

    script = _write_script(tmp_path / "fast.py", """
        print("done")
    """)
    cancel = threading.Event()
    result = run_step(Path(sys.executable), [str(script)], name="fast", cancel=cancel)
    assert result.ok
    cancel.set()  # 步骤已结束，finally 已让监视线程退出，不影响结果


def test_run_step_reports_unlaunchable_interpreter(tmp_path):
    with pytest.raises(ProcessError, match="无法启动"):
        run_step(tmp_path / "no-such-python", ["x"], name="missing")


# ---------------------------------------------------------------------------
# task_state
# ---------------------------------------------------------------------------

def test_task_state_sequence_and_terminal_guard():
    bus = EventBus()
    received = []
    bus.subscribe(received.append)
    state = TaskState("t1", bus)
    state.emit(EventType.TASK_STARTED)
    state.emit(EventType.ASR_STARTED)
    state.completed(output_path="/tmp/x")
    assert [e.sequence for e in received] == [1, 2, 3]
    assert [e.event_type for e in received] == [
        EventType.TASK_STARTED, EventType.ASR_STARTED, EventType.TASK_COMPLETED]
    with pytest.raises(RuntimeError, match="终止态"):
        state.emit(EventType.ASR_COMPLETED)


def test_task_state_degraded_carries_reasons():
    state = TaskState("t2")
    event = state.degraded(["纠错段 0:10 失败"])
    assert event.payload["reasons"] == ["纠错段 0:10 失败"]
    assert state.is_terminal


# ---------------------------------------------------------------------------
# runner（假子进程）
# ---------------------------------------------------------------------------

_FAKE_ASR = """
import argparse, json, pathlib
p = argparse.ArgumentParser()
for opt in ("--model", "--audio", "--output", "--device", "--language", "--prompt"):
    p.add_argument(opt)
p.add_argument("--window", type=float, default=0)
p.add_argument("--segmentation", default="auto")
p.add_argument("--full-max-seconds", type=float, default=300.0)
p.add_argument("--target-chunk", type=float, default=120.0)
p.add_argument("--max-chunk", type=float, default=180.0)
p.add_argument("--min-silence-ms", type=int, default=200)
p.add_argument("--overlap", type=float, default=0.5)
p.add_argument("--chunk-strategy", default="silence")
p.add_argument("--balance-tolerance", type=float, default=0.15)
p.add_argument("--balance-min-silence-ms", type=int, default=40)
p.add_argument("--vad-model")
p.add_argument("--max-new-tokens", type=int, default=4096)
p.add_argument("--batch-size", type=int, default=1)
p.add_argument("--batch-auto", action="store_true")
p.add_argument("--batch-tolerance", type=float, default=0.2)
p.add_argument("--batch-max", type=int, default=3)
p.add_argument("--batch-max-driver-mb", type=int, default=8000)
p.add_argument("--raw-output")
a = p.parse_args()
out = pathlib.Path(a.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("成交额三万亿", encoding="utf-8")
pathlib.Path(str(out.with_suffix(".stats.json"))).write_text(
    json.dumps({"stage": "asr", "chars": 6}), encoding="utf-8")
print("@@EVENT@@ " + json.dumps({"kind": "asr_completed", "chars": 6}))
"""

_FAKE_POST = """
import argparse, json, pathlib
p = argparse.ArgumentParser()
for opt in ("--input", "--output-dir", "--stem", "--corrector", "--corrector-model", "--glossary"):
    p.add_argument(opt)
p.add_argument("--no-itn", action="store_true")
a = p.parse_args()
out = pathlib.Path(a.output_dir)
out.mkdir(parents=True, exist_ok=True)
(out / f"{a.stem}_raw.txt").write_text(pathlib.Path(a.input).read_text(encoding="utf-8"), encoding="utf-8")
(out / f"{a.stem}_corrected.txt").write_text("成交额三万亿", encoding="utf-8")
(out / f"{a.stem}_final.txt").write_text("成交额3万亿", encoding="utf-8")
(out / f"{a.stem}_edits.jsonl").write_text("", encoding="utf-8")
status = "complete" if not a.no_itn else "complete"
(out / f"{a.stem}_postprocess.json").write_text(json.dumps({
    "status": status, "raw_chars": 6, "corrected_chars": 6, "final_chars": 5,
    "edits_total": 0, "edits_accepted": 0, "edits_review": 0,
    "itn_notes": 0, "itn_edits": 1, "itn_enabled": not a.no_itn,
    "corrector_load_seconds": 0.1, "corrector_generate_seconds": 0.2,
    "review_seconds": 0.01, "itn_seconds": 0.01,
}), encoding="utf-8")
for kind in ("correction_completed", "verify_completed", "itn_completed"):
    print("@@EVENT@@ " + json.dumps({"kind": kind}))
"""


def _fake_envs(tmp_path: Path) -> dict[str, PythonEnv]:
    versions = {"torch": "2.15.0", "transformers": "5.13.0", "wetext": "0.2.1", "mlx_lm": "0.32.2"}
    return {
        environment.ENV_FUNASR: PythonEnv("funasr", Path(sys.executable), tmp_path, dict(versions)),
        environment.ENV_MLX: PythonEnv("mlx", Path(sys.executable), tmp_path, dict(versions)),
    }


def _fake_scripts(tmp_path: Path, post_exit: int = 0) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    _write_script(scripts / "qwen3_asr_transcribe.py", _FAKE_ASR)
    _write_script(scripts / "qwen3_asr_postprocess.py", _FAKE_POST + f"\nraise SystemExit({post_exit})\n")
    return scripts


def _options(tmp_path: Path, **kwargs) -> OrchestrationOptions:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"stub")
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    base = dict(audio=audio, output=tmp_path / "out.txt", model=model,
                work_dir=tmp_path / "work")
    base.update(kwargs)
    return OrchestrationOptions(**base)


def test_runner_completes_and_publishes_final(tmp_path):
    bus = EventBus()
    events = []
    bus.subscribe(events.append)
    result = run(
        _options(tmp_path), event_bus=bus,
        envs=_fake_envs(tmp_path), scripts_dir=_fake_scripts(tmp_path),
    )
    assert result.status == "complete"
    assert result.final_text_path.read_text(encoding="utf-8") == "成交额3万亿"
    assert result.raw_text_path.read_text(encoding="utf-8") == "成交额三万亿"
    kinds = [e.event_type for e in events]
    assert kinds[0] == EventType.TASK_STARTED
    assert EventType.ASR_COMPLETED in kinds
    assert EventType.CORRECTION_COMPLETED in kinds
    assert EventType.VERIFY_COMPLETED in kinds
    assert EventType.ITN_COMPLETED in kinds
    assert kinds[-1] == EventType.TASK_COMPLETED
    assert result.stats["status"] == "complete"
    assert (result.work_dir / "orchestration.json").is_file()


def test_runner_marks_degraded_when_postprocess_exits_three(tmp_path):
    events = []
    bus = EventBus()
    bus.subscribe(events.append)
    result = run(
        _options(tmp_path), event_bus=bus,
        envs=_fake_envs(tmp_path), scripts_dir=_fake_scripts(tmp_path, post_exit=3),
    )
    assert result.status == "degraded"
    assert events[-1].event_type == EventType.TASK_DEGRADED


def test_runner_fails_when_asr_exits_nonzero(tmp_path):
    scripts = _fake_scripts(tmp_path)
    _write_script(scripts / "qwen3_asr_transcribe.py", "import sys\nsys.exit(2)\n")
    events = []
    bus = EventBus()
    bus.subscribe(events.append)
    with pytest.raises(OrchestrationError, match="ASR"):
        run(_options(tmp_path), event_bus=bus, envs=_fake_envs(tmp_path), scripts_dir=scripts)
    assert events[-1].event_type == EventType.TASK_FAILED
    assert events[-1].payload["stage"] == "asr"


def test_runner_rejects_existing_output_without_overwrite(tmp_path):
    options = _options(tmp_path)
    options.output.write_text("旧内容", encoding="utf-8")
    with pytest.raises(OrchestrationError, match="已存在"):
        run(options, envs=_fake_envs(tmp_path), scripts_dir=_fake_scripts(tmp_path))


def test_runner_overwrites_when_allowed(tmp_path):
    options = _options(tmp_path, overwrite=True)
    options.output.write_text("旧内容", encoding="utf-8")
    result = run(options, envs=_fake_envs(tmp_path), scripts_dir=_fake_scripts(tmp_path))
    assert result.final_text_path.read_text(encoding="utf-8") == "成交额3万亿"


def test_runner_exports_intermediates_on_request(tmp_path):
    options = _options(tmp_path, export_intermediates=True)
    run(options, envs=_fake_envs(tmp_path), scripts_dir=_fake_scripts(tmp_path))
    assert (tmp_path / "out_raw.txt").is_file()
    assert (tmp_path / "out_corrected.txt").is_file()
    assert (tmp_path / "out_edits.jsonl").is_file()


def test_runner_requires_corrector_model_for_mlx(tmp_path):
    with pytest.raises(OrchestrationError, match="corrector_model"):
        run(_options(tmp_path, correction="mlx"),
            envs=_fake_envs(tmp_path), scripts_dir=_fake_scripts(tmp_path))


def test_runner_checks_mlx_modules_when_correction_enabled(tmp_path):
    envs = _fake_envs(tmp_path)
    envs[environment.ENV_MLX] = PythonEnv(
        "mlx", Path(sys.executable), tmp_path, {"wetext": "0.2.1", "mlx_lm": None})
    model = tmp_path / "corrector"
    model.mkdir()
    with pytest.raises(EnvError, match="mlx_lm"):
        run(_options(tmp_path, correction="mlx", corrector_model=model),
            envs=envs, scripts_dir=_fake_scripts(tmp_path))

def test_runner_forwards_segmentation_to_asr(tmp_path):
    """切分策略必须透传给识别脚本，否则 CLI 参数只是装饰。"""
    from local_asr.orchestration.runner import _asr_argv

    argv = _asr_argv(_options(tmp_path, segmentation="vad"),
                     tmp_path / "scripts", tmp_path / "raw.txt")
    assert argv[argv.index("--segmentation") + 1] == "vad"


def test_runner_forwards_batching_to_asr(tmp_path):
    from local_asr.orchestration.runner import _asr_argv

    argv = _asr_argv(_options(tmp_path, batch_size=2),
                     tmp_path / "scripts", tmp_path / "raw.txt")
    assert argv[argv.index("--batch-size") + 1] == "2"
    assert "--batch-auto" not in argv

    argv = _asr_argv(_options(tmp_path, batch_auto=True),
                     tmp_path / "scripts", tmp_path / "raw.txt")
    assert argv[argv.index("--batch-size") + 1] == "1"
    assert "--batch-auto" in argv


def test_runner_rejects_invalid_batch_options(tmp_path):
    scripts = _fake_scripts(tmp_path)
    with pytest.raises(OrchestrationError, match="batch_size"):
        run(_options(tmp_path, batch_size=0),
            envs=_fake_envs(tmp_path), scripts_dir=scripts)
    with pytest.raises(OrchestrationError, match="互斥"):
        run(_options(tmp_path, batch_size=2, batch_auto=True),
            envs=_fake_envs(tmp_path), scripts_dir=scripts)
