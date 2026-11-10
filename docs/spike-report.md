# Windows Spike 报告

测试平台：NVIDIA GeForce RTX 4060 Laptop GPU（8GB 显存）/ 驱动 CUDA 13.4 / torch 2.11.0+cu128 / bf16 / Windows 10
测试音频：Chinese-LiPS `processed_test`（16kHz mono PCM16，中文朗读）；长音频为 LiPS 拼接（无静音，严苛场景）
对照基线：V3.0 设计的 macOS Apple Silicon MPS 实测数据
报告日期：2026-11-06（核心项完成）

## 1. 冒烟测试 ✅（`scripts/spike_smoke.py`）

| 指标 | RTX 4060 (CUDA bf16) | MPS 基线（V3.0） | 结论 |
|---|---:|---:|---|
| 模型加载（冷启动） | **1483 ms** | — | 本地 NVMe 极快 |
| 单条稳态 RTF | **0.086** | 0.110 | 快 1.28× |
| batch 4 RTF | **0.030** | — | — |
| batch 4 加速比 | **2.87×** | 1.98× | 符合「CUDA batch 收益更高」预期 |
| 峰值显存 | **2.02 GB** | — | — |

## 1.5 CER 基线 ✅（`scripts/spike_cer.py`，LiPS 50 条）

**CER 0.77%**（19 错 / 2481 字），RTF 0.030（batch 4）。与源项目 LiPS 同语料量级一致。主要错误：「的/地」「供能/功能」同音字与序号写法（「四、」→「4」），属 ITN/定向修正可处理范围。

## 2. ForcedAligner 中文对齐质量 ✅（`scripts/spike_aligner.py`）——**关键决策项**

**结论：Windows CUDA 上可用，时间戳方案升级为字符级。**

- 3/3 条音频时间戳校验全部合规（范围内 / 单调不回退 / 起止正常）
- 对齐单元为 CJK 字符（`ForcedAlignItem(text='说', start_time=0.24, end_time=0.4)`，秒单位）
- 双模型（ASR + Aligner）峰值显存 3.57 GB，8GB 卡安全
- 加载 4.4s；3 条音频 ASR+对齐推理 2.0s
- **落地含义**：`transcribe(..., return_time_stamps=True)` 原生输出 `aligned_tokens`；`unit_type = cjk_char` 命名与 V3.0 §3.2 预判一致；VAD 级片段时间戳降为回退方案

## 3. 长音频边界 ✅（`scripts/spike_longaudio.py`，1200s 拼接音频）

### 单次推理上限（待答项 6）

| 整段时长 | 耗时 | 输出 | 显存峰值 | 判定 |
|---|---:|---:|---:|---|
| 60s | 4.8s | 269 字 | 1.93 GB | 正常 |
| 120s | 9.1s | 500 字 | 3.50 GB | 正常 |
| 300s | 65.0s | 1242 字 | 6.75 GB | 正常（接近显存上限） |
| 600s | **1276.6s（21 分钟）** | 2414 字 | 13.88 GB | **不可用**：溢出物理显存走 WDDM 共享内存，慢 20 倍 |
| 1200s | — | — | — | OOM |

**结论：CUDA 单次推理实用上限 ≈ 300s，与 MPS 一致。** V3.0 的「强制 VAD 分段、禁止整段长音频」约束在 Windows 上同样必须执行。

### 60s 切块 + batch 4（待答项 4 / 7）

- 20 分钟音频：**51.2s，RTF 0.043**，20 块 0 空转写，显存峰值 3.38 GB
- 60 分钟外推：约 2.6 分钟 + 加载 1.5s —— 满足 V3.0 验收预期
- **结论：60s 块长 + batch 4 策略在 CUDA 上完整成立**（待答项 4 确认）

### batch 上限（待答项 1 补充，60s 等长块）

| batch | RTF | 显存峰值 | 判定 |
|---|---:|---:|---|
| 4 | 0.040 | 3.38 GB | **默认推荐** |
| 8 | 0.039 | 5.12 GB | 可用（收益仅 +2.5%） |
| 16 | 0.036 | 8.82 GB | 不推荐：已溢出 8GB 物理显存（WDDM 共享） |

**结论：8GB 卡 batch 上限为 8；batch 4→8 收益边际，维持 V3.0 默认 batch 4。**

## 4. 附录 B 待答项总览

| 待答项 | 状态 | 答案 |
|---|---|---|
| 1. RTF / 显存 / batch 上限 | ✅ | RTF 0.086/0.030；显存 2.02GB；batch 上限 8（推荐 4） |
| 2. ForcedAligner 质量 | ✅ | **可用，字符级时间戳，3/3 合规** |
| 3. CEC4-4B BF16 | ⏸ 未测 | 可选后处理（默认关闭），非阻塞项，阶段 1 补测 |
| 4. 60s 块长 | ✅ | 成立（RTF 0.043，0 空转写） |
| 5. batch 加速比 | ✅ | 2.87× > MPS 1.98× |
| 6. 单次推理上限 | ✅ | **≈300s（与 MPS 一致）** |
| 7. 60 分钟端到端 | ✅ 外推 | 20 分钟实测稳定；60 分钟约 2.6 分钟（待真实长音频复核） |
| 补充. FFmpeg 分发 | ✅ | 项目内捆绑 BtbN 静态构建（含 NVDEC） |

## 5. API 适配差异（踩坑记录）

| 坑 | 修正 |
|---|---|
| wetext 0.2.1 `operator='chn'` 失效 | `operator='itn'` |
| Qwen3ASR 不在 transformers 公开路径 | `from qwen_asr import Qwen3ASRModel`（qwen-asr 锁 transformers==4.57.6） |
| `python -m huggingface.cli` 已移除 | `hf download` |
| `--local-dir-use-symlinks` 已删除 | 不传该参数 |
| language 用全称 | `language="Chinese"` |
| `torch_dtype` 弃用警告 | 统一用 `dtype` |

## 6. 对后续开发的直接指导

1. **时间戳走字符级方案**：ASR + ForcedAligner 同卡加载（3.57GB），`return_time_stamps=True` 一次调用产出文本 + `aligned_tokens`；VAD 级时间戳保留为 Aligner 失败时的降级路径（V3.0 §5.5 硬性规则照常执行）。
2. **切块参数无需调整**：60s / batch 4 / jitter 策略原样移植；单次推理硬上限 300s 写入护栏。
3. **单 venv 方案成立**：ASR + Aligner 同环境无冲突；CEC4-4B 同为 transformers 栈，预计可合并（待验证）。
4. **WDDM 显存溢出是隐性性能陷阱**：600s 段不报错但慢 20 倍——所有显存护栏必须按「物理显存内」设计，不能依赖共享内存兜底。
