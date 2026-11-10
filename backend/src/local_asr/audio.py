"""音频探测与格式转换（纯标准库 + 可选 PyAV/ffmpeg）。

Qwen3-ASR 的 processor 只接受 **16 kHz 单声道 PCM16 WAV**，VAD 与切块也
基于该格式；本模块负责把任意输入音频归一到这一格式，并提供切片与时长
探测能力。

设计约束：
- 不导入 torch / funasr / transformers，保证在 ``.venv`` 下可直接单测；
- PyAV 与系统 ffmpeg 都是**惰性导入/探测**：二者任一可用即可转换，
  ffmpeg 优先（子进程转换不占本进程内存），PyAV 兜底（wheel 自带
  libav，不依赖系统安装）；
- 文件按**内容**而非扩展名解码，扩展名与容器不符（如 M4A 命名为
  ``.mp3``）也能正确处理。

三个转写脚本（qwen3_asr / faster_whisper / funasr）共用本模块，
不再各自维护 ``audio_duration`` 等重复实现。
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

__all__ = [
    "audio_duration",
    "can_convert_audio",
    "is_pcm16k_mono_wav",
    "prepare_pcm16k",
    "slice_wav",
]


# ---------------------------------------------------------------------------
# 探测
# ---------------------------------------------------------------------------

def audio_duration(path: Path) -> float:
    """音频时长（秒）。

    三级回退，按成本从低到高：

    1. ``wave``（标准库）——只对规范 WAV 有效，最快且零依赖；
    2. PyAV——读取容器级 ``duration`` 元数据（微秒），覆盖 mp3/m4a 等；
    3. ``ffprobe`` 子进程——PyAV 也未安装时的最后手段。

    Raises:
        RuntimeError: 三条路径都不可用（缺 PyAV 且缺 ffprobe）。
    """
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError, OSError):
        pass
    try:
        import av
    except ImportError:
        av = None  # type: ignore[assignment]
    if av is not None:
        with av.open(str(path)) as container:
            # 容器 duration 单位是微秒；个别流式容器可能没有该元数据
            return (container.duration or 0) / 1_000_000.0
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError(
            f"无法读取音频时长（缺少 PyAV 与 ffprobe）: {path}\n"
            f"请先转换：ffmpeg -i {path} -ac 1 -ar 16000 -c:a pcm_s16le out.wav")
    completed = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(completed.stdout.strip())


def is_pcm16k_mono_wav(path: Path) -> bool:
    """判断输入是否已经是模型需要的 16 kHz 单声道 PCM16 WAV。

    四个条件缺一不可：单声道、16000 Hz 采样率、16-bit 样本宽度、
    未压缩 PCM（comptype ``NONE``）。任何一项不满足都需要走转换。
    """
    try:
        with wave.open(str(path), "rb") as w:
            return (w.getnchannels() == 1 and w.getframerate() == 16000
                    and w.getsampwidth() == 2 and w.getcomptype() == "NONE")
    except (wave.Error, EOFError, OSError):
        return False


def can_convert_audio() -> bool:
    """当前环境是否具备「任意格式 → 16 kHz WAV」的转换能力。

    系统 ffmpeg 或 PyAV 任一可用即返回 True；preflight 阶段用它把
    「无法转换」提前变成明确报错，而不是等模型加载完才失败。
    """
    if shutil.which("ffmpeg"):
        return True
    try:
        import av  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# 转换
# ---------------------------------------------------------------------------

def _convert_with_ffmpeg(src: Path, dst: Path) -> bool:
    """系统 ffmpeg 转换；未安装返回 False（由调用方回退 PyAV）。

    ``-nostdin`` 防止子进程抢占终端输入；转换在子进程完成，
    不占用本进程内存，对超长音频更友好。
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    subprocess.run(
        [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        check=True, timeout=1800,
    )
    return True


def _pcm_bytes(frame) -> bytes:
    """取 PyAV ``AudioFrame`` 的有效 PCM 字节。

    ``planes[0]`` 的缓冲区长度按 64 字节对齐，尾部含 padding；直接整块
    写出会把音频拉长 7~24%（实测 8s→9.91s、331.9s→357.6s，见回归测试
    ``test_pcm_bytes_strips_frame_padding``）。s16 单声道每样本 2 字节，
    因此按 ``frame.samples * 2`` 截断。
    """
    return bytes(frame.planes[0])[: frame.samples * 2]


def _convert_with_av(src: Path, dst: Path) -> bool:
    """PyAV 兜底转换：wheel 自带 libav，不依赖系统 ffmpeg。

    流程：解码任意容器 → ``AudioResampler`` 重采样为 s16/mono/16 kHz →
    逐帧写入 WAV。重采样器 ``resample(None)`` 冲刷尾部缓冲，
    否则最后不足一帧的音频会被丢弃。
    """
    try:
        import av
    except ImportError:
        return False
    with av.open(str(src)) as container:
        if not container.streams.audio:
            raise RuntimeError(f"文件不含音频流: {src}")
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=16000)
        with wave.open(str(dst), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    out.writeframes(_pcm_bytes(converted))
            for converted in resampler.resample(None):
                out.writeframes(_pcm_bytes(converted))
    return True


def prepare_pcm16k(src: Path, dst: Path) -> Path:
    """把任意音频转成 16 kHz 单声道 PCM16 WAV。

    已经是目标格式时**直接返回原路径**（不产生副本）；否则写入 ``dst``
    并返回 ``dst``。转换顺序：系统 ffmpeg 优先（子进程、不占内存），
    失败或未安装时回退 PyAV。

    Returns:
        实际可用的 16 kHz WAV 路径（``src`` 或 ``dst``）。

    Raises:
        RuntimeError: 两条转换路径都失败；报错信息附带手工转换命令。
    """
    if is_pcm16k_mono_wav(src):
        return src

    errors: list[str] = []
    try:
        if _convert_with_ffmpeg(src, dst):
            return dst
    except subprocess.CalledProcessError as exc:
        errors.append(f"ffmpeg 退出码 {exc.returncode}")
    except subprocess.TimeoutExpired:
        errors.append("ffmpeg 转换超时")
    try:
        if _convert_with_av(src, dst):
            return dst
    except ImportError:
        errors.append("未安装 PyAV")
    except Exception as exc:  # 解码器对损坏/不支持的容器抛出的各类异常
        errors.append(f"PyAV 解码失败: {exc}")

    detail = "；".join(errors) if errors else "未找到 ffmpeg，且未安装 PyAV"
    raise RuntimeError(
        f"无法把 {src} 转换为 16 kHz 单声道 WAV（{detail}）。\n"
        f"请先转换：ffmpeg -i {src} -ac 1 -ar 16000 -c:a pcm_s16le out.wav")


# ---------------------------------------------------------------------------
# 切片
# ---------------------------------------------------------------------------

def slice_wav(src: Path, start_ms: int, end_ms: int, dst: Path) -> None:
    """按毫秒区间从 16 kHz WAV 中切出一段，写成独立 WAV。

    只支持 ``wave`` 可读的 PCM WAV（即 :func:`prepare_pcm16k` 的产物）；
    毫秒→帧号按采样率换算，写入时沿用源文件的全部参数
    （声道/采样率/位宽），保证切块与整段识别的输入格式一致。
    """
    with wave.open(str(src), "rb") as w:
        sr = w.getframerate()
        params = w.getparams()
        w.setpos(int(start_ms * sr / 1000))
        data = w.readframes(max(0, int((end_ms - start_ms) * sr / 1000)))
    with wave.open(str(dst), "wb") as o:
        o.setparams(params)
        o.writeframes(data)
