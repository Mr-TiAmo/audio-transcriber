# 转写耗时分析：0.6B / 1.7B × 音频长度（2026-11-08）

## 1. 方法

- 数据来源：`data/temp/<run_id>/asr/raw.stats.json`（`load_seconds` / `infer_seconds` /
  `aux_seconds` / `rtf` / `batching` / `segmentation`）、
  `postprocess/result_postprocess.json`（`itn_seconds` / `elapsed_seconds`）、
  `data/database/app.db`（项目时长、模型、墙钟起止）。
- 汇总脚本：`backend/scripts/bench_cost_summary.py`（逐 run 明细 + 按模型/时长档中位数）。
  ```bash
  python -I backend/scripts/bench_cost_summary.py --limit 60
  ```
- 统计口径：近期（2026-11-07 至 11-08）完成的 run，按音频时长分组取**中位数**；
  已剔除失败/取消任务。模型为 Qwen3-ASR-0.6B / 1.7B，分段 `balanced`、
  overlap 1.5s、低置信参数 `max_ratio=0.01 / score_floor=0.8`，
  0.6B 长音频 `batch=4`，1.7B 因显存限制 `batch=1`。

## 2. 实测结果（墙钟中位数，优化前基线）

| 音频 | 模型 | wall | ASR 合计 | 加载 | 推理 | aux(对齐) | 后处理 | ITN | infer RTF |
|---|---|---|---|---|---|---|---|---|---|
| 8.4 s | 0.6B | 12–13 s | 10.2 | 5.6 | 3.5 | 2.0 | 0.2 | 0.1 | 0.42 |
| 8.4 s | 1.7B | 20–21 s | 17.7 | 10.2 | 6.4 | 3.0 | 0.2 | 0.1 | 0.76 |
| 10.7 s | 0.6B | 11–15 s | 9–12 | 6.0–8.4 | 1.5–5.3 | 2.0–2.1 | 0.3 | 0.2 | 0.15–0.49 |
| 310.7 s | 0.6B | 51 s | 42.2 | 5.7 | 35.4 | 4.6 | 6.7 | 6.6 | 0.114 |
| 310.7 s | 1.7B | 68 s | 58.8 | 9.2 | 48.5 | 5.8 | 6.6 | 6.5 | 0.156 |
| 599.1 s | 0.6B | 47 s | 32.3 | 5.7 | 25.3 | 5.2 | 12.6 | 12.5 | 0.042 |
| 599.1 s | 1.7B | 152 s | 135.9 | 9.1 | 125.7 | 8.6 | 13.8 | 13.7 | 0.210 |
| 3540 s | 0.6B | 192 s | 119.9 | 6.4 | 112.1 | — | 70.4 | 70.3 | 0.032 |

> 注：310.7 s 的同音频、同参数两次运行分别为 34 s（infer 18.5 s）与 52 s（infer 35.5 s），
> 说明本机 GPU 状态波动可达 ~1.9×，单次数据不可当作精确基准。

### 2.1 异常样本

| 场景 | 现象 | 可能原因 |
|---|---|---|
| 599 s，0.6B，交叉验证开启 | wall 120 s（infer 96.1 s + cross-check 46.6 s） | FunASR 冷启动 26 s + 推理 16 s；主 ASR 同时段异常变慢 |
| 3540 s，0.6B | wall 299 s（infer 156.6 s、ITN 128.2 s） | GPU 降频/占用；ITN 处理量异常放大 |
| 310 s，0.6B | 34 s vs 52 s 两次 | GPU 状态波动（同参数同音频） |

## 3. 成本模型（优化前基线）

> 本节描述 **2026-11-08 优化落地之前** 的行为，用于对照第 4 节的改后实测。
> 「每次任务重新加载」「1.7B batch=1」「ITN 串行」等前提已不再成立。

墙钟 ≈ `固定开销` + `加载` + `aux` + `推理` + `后处理` + `入库`（约 1–3 s）

- **加载**：0.6B ≈ 5.5–6.5 s；1.7B ≈ 8–10.5 s（每次任务重新加载）。
- **推理**：`音频语音时长 × RTF`，RTF 取决于批量与内容：
  - batch=4、块数 ≥5：0.03–0.06（10 min 以上稳定在 0.03）；
  - 短音频单条（batch=1）：0.15–0.76，固定开销占比极高；
  - 1.7B 只能 batch=1：0.156（310 s）、0.210（599 s）。
- **aux（ForcedAligner 对齐）**：2–6 s，随文本/段数增长。
- **后处理**：约 `4.5–5 ms/字`，其中 wetext ITN 占 96%+（difflib 仅 0.3–5 s）。
  实测：1419 字 → 6.6 s；2420 字 → 12.5 s；15558 字 → 70.3 s。

### 3.1 固定开销占比

| 音频 | 0.6B 加载+aux 占 wall | 1.7B 加载+aux 占 wall |
|---|---|---|
| 8.4 s | 7.6 / 13 ≈ **58%** | 13.2 / 20.5 ≈ **64%** |
| 310.7 s | 10.3 / 51 ≈ 20% | 15.0 / 68 ≈ 22% |
| 599.1 s | 10.9 / 47 ≈ 23% | 17.7 / 152 ≈ 12% |
| 3540 s | 6.4 / 192 ≈ 3% | — |

结论：**5 分钟以内，模型加载与对齐是主要成本；10 分钟以上，推理主导；后处理随文本量线性增长**。

## 4. 优化实现与实测（2026-11-08 已全部落地）

### P0：ASR 模型常驻（已实现）

`scripts/qwen3_asr_worker.py` 常驻进程 + `local_asr/orchestration/asr_worker.py`
执行器（签名同 `run_step`，服务层注入 `runner.run(asr_executor=...)`）。
`qwen3_asr_transcribe.py` 对 ASR / ForcedAligner / VAD 三份权重做模块级缓存，
切换 0.6B/1.7B 时释放旧权重并 `empty_cache`，每次任务后归还临时显存块。
取消/超时终止 worker，下次任务自动重启；worker 崩溃只影响当前任务。

实测（同一 worker 连续任务，文本逐字符一致）：

| 场景 | 首次（冷启动） | 第二次起 | 省 |
|---|---|---|---|
| 8.4 s / 0.6B | 11.82 s | 0.70–0.85 s | ~11 s（加载 6.5 s + 对齐 2 s + GPU 预热） |
| 599 s / 0.6B | 31.21 s | 22.09 s | 9.1 s（加载 5.35 s + 对齐 + VAD） |
| 8.4 s / 0.6B（生产链路） | 10.55 s | 1.93 s | 8.6 s |
| 310 s / 0.6B（生产链路） | — | 17.38 s（load 0、infer 17.35、RTF 0.056） | 对比旧 51 s |
| 310 s / 1.7B（生产链路） | 43.77 s（load 8.95、infer 34.73、aux 5.96） | 第二次起再省 ~15 s | 对比旧 68 s |

代价：常驻显存约 2.4 GB（0.6B + 对齐模型），8 GB 卡安全。

### P0：ITN（wetext）优化（已实现，方案与预期不同）

初版实现「数字候选预筛 + 连续命中句并行」，10 min 文本 12.2 s → 1.44 s；
但 60 min 文本实测发现预筛**会漏改**：wetext 对「单个数词 + 任意名词」也会转换
（「西湖十景」→「西湖10景」），而「十分」「一起」「一季度」保留——边界无法用正则枚举。

最终改为**全句切块 + 多进程并行**（不跳过任何句子），并保留快速路径
（短文本，或既无数词也无词典时直接整篇）：

| 样本 | 整篇 | 全句切块（8 进程） | 一致 |
|---|---|---|---|
| 60 min / 15558 字（run_793da67b85f4） | 66.9 s | **13.6 s**（生产 pipeline 实测） | ✅ |
| 60 min / 15558 字（run_8b323fd26bea） | 68–90 s | 13.6–29.5 s | ✅ |
| 10 min / 2420 字 | ~12 s | ~3 s | ✅ |

并行进程数默认 `min(8, CPU 核数)`（24 核实测 4 进程 20.8 s、8 进程 13.6 s）。

### P1：交叉验证并行（已实现）

FunASR（CPU）在 `prepare_pcm16k` 后立即后台启动，与主 ASR 推理并行；
主推理结束后只等剩余部分。599 s 实测：后台总量 37.37 s，映射耗时仅 9.79 s
（串行时新增 46.6 s），且 FunASR 用 CPU 不抢主模型显存。

### P1：1.7B 批量效率（已实现）

改为 30 s chunk × `batch=2`（`task_service.batch_for_model` /
`chunk_seconds_for_model`）：

| 599 s / 1.7B | 旧（60 s × batch=1） | 新（30 s × batch=2） |
|---|---|---|
| infer | 125.7 s | **39.4 s** |
| RTF | 0.210 | **0.066** |
| ASR 合计 | 135.9 s | 48.7 s |
| 峰值显存 | — | 约 5.3 GB |
| 文本 | — | 几乎一致（4 处数字写法差异） |

310 s / 1.7B 生产实测：ASR 43.8 s、RTF 0.112（旧 0.156、wall 68 s）。

### P2：aux（ForcedAligner）快速模式（已实现）

API `POST /projects/{id}/runs` 新增 `fast` 字段；前端任务栏「快速」勾选框。
`fast=true` 时不加载对齐模型、不收集字符时间，句级时间戳退回按字数插值。
实测 8.4 s 音频 `aux_seconds` 2.14 → 0（常驻后正常路径也已降到 0.09）。

### P2：波动治理（已实现）

- ASR stats 新增 `gpu`：`nvidia-smi` 每 2 s 采样
  `utilization/memory/temperature/sm_clock`，输出均值、峰值、
  `throttled`（只看利用率 ≥30% 的工作时段，避免空闲低频误报）；
- `batching.group_timings` 记录每批耗时 / 音频秒，超过中位数 2× 的批进入
  `batching.slow_groups`，用于区分「模型慢」与「环境抖动」；
- 无 `nvidia-smi`（CPU/MPS）时自动禁用，不影响主流程。

## 5. 优化后收益汇总

| 场景 | 优化前 | 优化后（实测） | 降幅 |
|---|---|---|---|
| 8.4 s / 0.6B（连续任务） | 12–13 s | **0.7–1.9 s** | ~85% |
| 310 s / 0.6B | 51 s | **17.4 s** | -66% |
| 599 s / 0.6B | 47 s | **22.1 s**（含常驻） | -53% |
| 599 s / 0.6B + 交叉验证 | 120 s | 预计 **~60 s** | -50% |
| 310 s / 1.7B | 68 s | **43.8 s** | -36% |
| 599 s / 1.7B | 152 s | 预计 **~60 s** | -60% |
| 60 min / 0.6B | 192 s | 预计 **~135 s**（ITN 70→14 s） | -30% |

> 环境波动仍在（同音频同参数 1.9×），上表为同机连续实测，不代表绝对性能。

## 6. 复现

```bash
cd D:/work/workspace/audio-transcriber

# 历史 run 成本汇总（第 1–3 节数据来源）
python -I backend/scripts/bench_cost_summary.py --limit 60
python -I backend/scripts/bench_cost_summary.py --csv data/perf-summary.csv

# 常驻 worker 收益（连续 N 个任务，对比 load_seconds）
python -I backend/scripts/bench_worker_reuse.py --repeat 3 \
  --model models/Qwen3-ASR-0.6B --audio data/samples/xxx.wav

# ITN 整篇 vs 全句切块并行
python -I backend/scripts/bench_itn_segmented.py --audio <stats.json 的文本文件>
```