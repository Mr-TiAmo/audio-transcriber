"""语义纠错适配器。

本轮只提供不训练、不联网的本地推理路径：
- ``PassthroughCorrector``：不做纠错（A 基线 / 无模型时的降级）
- ``MLXCorrector``：Apple Silicon 上的 MLX-LM 4-bit 推理

模型准备需显式授权，运行只接受完整本地路径；缺文件直接报错，不回退联网。
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "BaseCorrector",
    "CorrectionResult",
    "MLXCorrector",
    "PassthroughCorrector",
    "build_corrector",
    "SYSTEM_PROMPT",
]

# CEC4-4B 官方指令模板（模型卡原文）。实测：官方指令纠错能力明显强于自拟的
# 保守指令，且不会把 prompt 本身复述进输出；ASR 场景的"不改数字/不过度润色"
# 由 verify 的改动校验兜底，而不是靠改提示词。
#
# user 消息只放待校对原文，不带"待校对文本："之类的标签：一旦加标签，模型会把
# 标签连同正文一起复述出来（见 2026-11-05 实测），使解析器判定为泄漏。
SYSTEM_PROMPT = (
    "假如你是一名专业的纠错专家，请分析输入句子的语法错误类型和修改原因，"
    "并只输出纠正后的语句，错误类型如下：错别字、词语搭配错误、词性错误、"
    "语序错误、成分残缺、成分赘余、关联词使用错误、指代不明、语义逻辑不通、无误。"
)

# 只接受正文或单个标准 think 块后的正文；模板泄漏直接回退，不猜测清理。
_END_THINK_RE = re.compile(r"</think\s*>")
_LEAK_RE = re.compile(r"<\|im_end\|>|<\|im_start\|>|^\s*(?:user|assistant|system)\s*$", re.M)
# 4-bit 模型偶尔在 think 块后复刻 chat template，把 "user" 行写进正文；
# 它只是角色标记，剥掉首行是确定性操作，不涉及内容猜测。
_ROLE_LINE_RE = re.compile(r"^\s*(?:user|assistant|system)\s*\n")
_ECHO_RE = re.compile(r"^\s*(?:只输出|注意：输入是|假如你是一名专业的纠错专家)")
_NO_ERROR_RE = re.compile(r"错误类型\s*[：:]\s*无误")
_SENT_END = re.compile(r"(?<=[。！？；!?;])")


@dataclass
class CorrectionResult:
    """纠错阶段结果。"""

    text: str
    model: str
    segments: int = 0
    failed_segments: int = 0
    truncated: bool = False
    notes: list[str] = field(default_factory=list)
    segment_results: list[CorrectionSegment] = field(default_factory=list)
    load_seconds: float = 0.0
    generate_seconds: float = 0.0
    mlx_peak_bytes: int | None = None


@dataclass(frozen=True)
class CorrectionSegment:
    """保留原文范围；失败原因只记录状态，不保存模型分析内容。"""

    start: int
    end: int
    source: str
    candidate: str
    status: str = "ok"
    reason: str = ""
    prompt_tokens: int = 0
    generation_tokens: int = 0
    finish_reason: str | None = None
    prompt_tps: float = 0.0
    generation_tps: float = 0.0


@dataclass(frozen=True)
class StreamGeneration:
    """一次 MLX ``stream_generate`` 调用的输出与资源指标。

    命名刻意区别于 :class:`local_asr.asr.generation.GenerationResult`：
    后者描述 ASR 音频推理的完整性判定（EOS/触顶），本类描述纠错模型
    单段文本生成的流式结果（finish_reason/token 吞吐/显存峰值），
    二者字段语义完全不同，同名会误导跨模块阅读。

    Attributes:
        text: 解析后的纠错正文；空串表示生成失败或输出被判为模板泄漏。
        finish_reason: ``stop``（正常）/ ``length``（触顶）/ ``error``（异常）。
        prompt_tokens: 提示 token 数（含 think 块）。
        generation_tokens: 生成 token 数。
        mlx_peak_bytes: MLX allocator 峰值字节数；CPU 回退时为 None。
        prompt_tps: prefill 吞吐（tokens/s），用于反推 prompt cache 收益。
        generation_tps: 解码吞吐（tokens/s）。
    """

    text: str
    finish_reason: str | None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    mlx_peak_bytes: int | None = None
    prompt_tps: float = 0.0
    generation_tps: float = 0.0


def require_module(name: str, hint: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"当前解释器 {sys.executable} 缺少 {name}。{hint}")


def validate_local_model(path: Path, *, asr: bool = False) -> dict:
    """本地文件预检，绝不把目录字符串作为远程仓库名。"""
    if not path.is_dir():
        raise RuntimeError(f"模型目录不存在: {path}；不会联网下载")
    required = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    if asr:
        required.append("processor_config.json")
    for name in required:
        file = path / name
        if not file.is_file() or not file.stat().st_size:
            raise RuntimeError(f"模型文件缺失或为空: {file}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
    weights = list(path.glob("*.safetensors"))
    if not weights or any(not p.is_file() or not p.stat().st_size for p in weights):
        raise RuntimeError(f"本地模型权重缺失或为空: {path}")
    for index in path.glob("*.safetensors.index.json"):
        manifest = json.loads(index.read_text(encoding="utf-8"))
        for name in set(manifest["weight_map"].values()):
            file = path / name
            if file.parent.resolve() != path.resolve() or not file.is_file() or not file.stat().st_size:
                raise RuntimeError(f"本地分片权重不完整: {file}")
    return config


class BaseCorrector:
    """纠错器基类。"""

    name = "base"

    def preflight(self) -> None:
        """不加载权重的配置检查；简单纠错器无需额外检查。"""

    def prepare(self) -> None:
        """可选的模型加载阶段，与生成耗时分开记录。"""

    def correct(self, text: str) -> CorrectionResult:  # pragma: no cover - 抽象
        raise NotImplementedError


class PassthroughCorrector(BaseCorrector):
    """不做语义纠错，原样返回（A 基线）。"""

    name = "none"

    def correct(self, text: str) -> CorrectionResult:
        return CorrectionResult(text=text, model=self.name)


class MLXCorrector(BaseCorrector):
    """基于 MLX-LM 的本地纠错器。

    Args:
        model_path: 本地模型目录（必须存在）。
        model_name: 展示名，默认取目录名。
        max_tokens: 单段输出预算。官方 README 用 512，实测正常段输出
            193–282 token；给出异常复读留太多空间只会拖慢失败，故取 512。
        target_chars: 单段目标字符数（方案建议 200–400）。
        repetition_penalty: 重复惩罚。官方 README 要求 1.1；缺失时 4-bit
            模型会在 think 块后连续输出多个"错误类型/修改原因"，不产出
            纠正句（2026-11-05 实测：close_think=4、无正文）。
        context_sentences: 附带的前文句数。实测 CEC4-4B（含 4-bit）会把
            前文整段复述进输出，导致该段错位，因此默认 0；需要时可显式打开。
    """

    name = "mlx"

    def __init__(
        self,
        model_path: str | Path,
        model_name: str | None = None,
        max_tokens: int = 512,
        target_chars: int = 300,
        context_sentences: int = 0,
        max_input_tokens: int = 2048,
        repetition_penalty: float = 1.1,
    ) -> None:
        if min(max_tokens, target_chars, max_input_tokens) <= 0 or context_sentences < 0:
            raise ValueError("token/字符预算必须为正数，上下文句数不能为负数")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty 不能小于 1.0；1.0 表示不惩罚")
        self.model_path = Path(model_path).resolve()
        self.max_input_tokens = max_input_tokens
        self.model_metadata: dict = {}
        self.model_name = model_name or self.model_path.name
        self.max_tokens = max_tokens
        self.target_chars = target_chars
        self.context_sentences = context_sentences
        self.repetition_penalty = repetition_penalty
        # 20 是 MLX 默认值，只覆盖最近 20 token，压不住"整段分析块复读"；
        # 128 实测能让复读段收敛到单个 think 块，且不影响正常段。
        self.repetition_context_size = 128
        self._model = None
        self._tokenizer = None

    # -- 资源 -------------------------------------------------------------

    def preflight(self) -> None:
        self.model_metadata = validate_local_model(self.model_path)
        require_module("mlx_lm", "请用 .venv-mlx/bin/python scripts/qwen3_asr_postprocess.py 处理 raw；ASR 与纠错分两次调用。")
        installed = importlib.metadata.version("mlx-lm")
        release = re.match(r"^(\d+)\.(\d+)\.(\d+)", installed)
        if release is None or tuple(map(int, release.groups())) < (0, 32, 2):
            raise RuntimeError(
                f"当前解释器 {sys.executable} 的 mlx-lm={installed} 未纳入支持范围；"
                "流式终止元数据按 0.32.2 接口验证，请使用已有 .venv-mlx/bin/python，"
                "或经授权准备 mlx-lm>=0.32.2；不会自动安装。"
            )

    def prepare(self) -> None:
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self.preflight()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from mlx_lm import load

        model, tokenizer = load(
            str(self.model_path),
            tokenizer_config={"local_files_only": True, "trust_remote_code": False},
        )
        self._model, self._tokenizer = model, tokenizer

    # -- 分段 -------------------------------------------------------------

    def _segments(self, text: str) -> list[tuple[str, str]]:
        """按句切分并合并为目标段；返回 (前文上下文, 目标段)。"""
        sentences = [s for s in _SENT_END.split(text) if s]
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            for start in range(0, len(sentence), self.target_chars):
                part = sentence[start:start + self.target_chars]
                if current and len(current) + len(part) > self.target_chars:
                    chunks.append(current)
                    current = ""
                current += part
        if current:
            chunks.append(current)

        result = []
        consumed = 0
        for chunk in chunks:
            previous = [s for s in _SENT_END.split(text[:consumed]) if s]
            context = "".join(previous[-self.context_sentences:]) if self.context_sentences else ""
            consumed += len(chunk)
            pending = [chunk]
            while pending:
                target = pending.pop(0)
                if self._tokenizer is not None:
                    # 上下文只作参考；先舍弃超预算上下文，再缩小目标原文范围。
                    if len(self._prompt_tokens(context, target)) > self.max_input_tokens:
                        context = ""
                    if len(self._prompt_tokens(context, target)) > self.max_input_tokens:
                        if len(target) <= 1:
                            raise ValueError("输入 token 预算不足以容纳提示及一个字符")
                        middle = len(target) // 2
                        pending[0:0] = [target[:middle], target[middle:]]
                        continue
                result.append((context, target))
        return result

    # -- 推理 -------------------------------------------------------------

    def _prompt_tokens(self, context: str, target: str) -> list[int]:
        # 不加"待校对文本："标签：模型会把它复述进输出。上下文默认关闭，
        # 仅在显式开启时以纯文本前置。
        user = f"{context}\n\n{target}" if context else target

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._tokenizer.encode(prompt, add_special_tokens=False)

    def _logits_processors(self):
        """重复惩罚；官方 README 要求 1.1，缺失会触发 think 块复读。"""
        from mlx_lm.sample_utils import make_logits_processors

        return make_logits_processors(
            repetition_penalty=self.repetition_penalty,
            repetition_context_size=self.repetition_context_size,
        )

    def _generate(self, context: str, target: str) -> StreamGeneration:
        from mlx_lm import stream_generate

        prompt = self._prompt_tokens(context, target)
        if len(prompt) > self.max_input_tokens:
            raise ValueError("输入 token 预算超限")
        processors = self._logits_processors()
        acc = ""
        last = None
        no_error = False
        try:
            for response in stream_generate(
                self._model, self._tokenizer, prompt=prompt,
                max_tokens=self.max_tokens, logits_processors=processors,
            ):
                acc += response.text
                last = response
                # 模型判「无误」时不会主动停：它会把原文整段复述一遍，长段
                # 甚至复读到 max_tokens（2026-11-05 长音频实测：559 字判无误
                # 后复述到 768 上限，整段被退回原文）。think 块闭合且已确认
                # 无误时，后面的正文只可能是复述，立即停止并采用原文。
                if (_END_THINK_RE.search(acc) and _NO_ERROR_RE.search(acc)):
                    no_error = True
                    break
        except Exception:
            return StreamGeneration(
                "", "error", last.prompt_tokens if last else len(prompt),
                last.generation_tokens if last else 0,
                int(last.peak_memory * 1e9) if last else None,
                getattr(last, "prompt_tps", 0.0) if last else 0.0,
                getattr(last, "generation_tps", 0.0) if last else 0.0,
            )
        if last is None:
            return StreamGeneration("", None, len(prompt))
        if no_error:
            return StreamGeneration(
                target, "stop", last.prompt_tokens, last.generation_tokens,
                int(last.peak_memory * 1e9),
                getattr(last, "prompt_tps", 0.0), getattr(last, "generation_tps", 0.0),
            )
        return StreamGeneration(
            self._parse_output(acc, target),
            last.finish_reason, last.prompt_tokens, last.generation_tokens,
            int(last.peak_memory * 1e9),
            getattr(last, "prompt_tps", 0.0), getattr(last, "generation_tps", 0.0),
        )

    @staticmethod
    def _parse_output(out: str, original: str | None = None) -> str:
        """接受正文或单个标准 think 块后的正文；未知格式和模板泄漏返回空串。

        特别地，模型明确给出「错误类型：无误」且没有正文时返回原文：
        这是模型判定"无需修改"的确定结论，不是格式异常。
        """
        tags = list(_END_THINK_RE.finditer(out))
        if len(tags) > 1:
            return ""
        body = out
        if tags:
            prefix = out[:tags[0].start()].lstrip()
            # chat template 可预填 <think>，故允许仅出现一个关闭标签。
            if "<think>" in prefix and (not prefix.startswith("<think>") or prefix.count("<think>") != 1):
                return ""
            body = out[tags[0].end():]
        # 先剥离正文首行的角色标记，再判断是否仍有模板泄漏。
        body = _ROLE_LINE_RE.sub("", body, count=1)
        if _LEAK_RE.search(body):
            return ""
        if re.search(r"<[^>]*think|<\|", body, re.I):
            return ""
        if any(marker in body for marker in ("前文（只读", "待校对文本：", "错误类型：", "修改原因：")):
            return ""
        body = body.strip()
        # 「错误类型：无误」是模型的确定结论，正文只是原文复述（长段可能复述
        # 多遍），因此无论正文是否为空都直接采用原文。
        if original is not None and _NO_ERROR_RE.search(out):
            return original
        if not body:
            return ""
        if _ECHO_RE.match(body):
            return ""
        # 兜底：纠正句不应显著长于原文；出现超长正文说明模型跑偏（复读、
        # 分析块残留），宁可整段回退也不把异常文本当纠错结果。
        if original and len(body) > 3 * max(len(original), 1):
            return ""
        return body

    def correct(self, text: str) -> CorrectionResult:
        started = time.perf_counter()
        self._ensure_loaded()
        load_seconds = time.perf_counter() - started
        segments = self._segments(text)
        started = time.perf_counter()
        results: list[CorrectionSegment] = []
        peak = None
        offset = 0
        for context, target in segments:
            try:
                generated = self._generate(context, target)
            except Exception as exc:
                generated = StreamGeneration("", "error")
                reason = f"生成失败: {type(exc).__name__}"
            else:
                reason = "" if generated.finish_reason == "stop" and generated.text else "生成未正常结束或输出解析失败"
            status = "failed" if reason else "ok"
            candidate = generated.text if status == "ok" else target
            results.append(CorrectionSegment(
                offset, offset + len(target), target, candidate, status, reason,
                generated.prompt_tokens, generated.generation_tokens, generated.finish_reason,
                generated.prompt_tps, generated.generation_tps,
            ))
            if generated.mlx_peak_bytes is not None:
                peak = max(peak or 0, generated.mlx_peak_bytes)
            offset += len(target)
        return CorrectionResult(
            text="".join(s.candidate for s in results), model=self.model_name,
            segments=len(results), failed_segments=sum(s.status != "ok" for s in results),
            truncated=any(s.finish_reason == "length" for s in results),
            segment_results=results, load_seconds=load_seconds,
            generate_seconds=time.perf_counter() - started, mlx_peak_bytes=peak,
        )


def build_corrector(kind: str | None, model_path: str | None = None, **kwargs) -> BaseCorrector:
    """按名称构造纠错器。"""
    if kind in (None, "", "none", "off"):
        return PassthroughCorrector()
    if kind == "mlx":
        if not model_path:
            raise ValueError("mlx 纠错器需要 --corrector-model 指定本地模型目录")
        return MLXCorrector(model_path, **kwargs)
    raise ValueError(f"未知纠错器: {kind}")