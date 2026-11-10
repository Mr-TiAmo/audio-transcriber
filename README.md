# AudioTranscriber 本地音频转文字与校对工具

基于《本地音频转文字与校对工具 技术详细概要设计 V3.0（最终版）》落地的 Windows 本地 Web 应用。

> 设计文档源：`D:\work\workspace\local-asr-portable\docs\本地音频转文字与校对工具_技术详细概要设计_V3.0_最终版.md`

## 技术栈

| 层次 | 选型 |
|---|---|
| 前端 | Vue 3 + TypeScript + Vite + Element Plus + 虚拟列表 |
| 本地 API | Python 3.12 + FastAPI（单进程、单 Worker，监听 127.0.0.1） |
| ASR 主模型 | Qwen3-ASR-0.6B（默认）/ Qwen3-ASR-1.7B（质量优先），Transformers 后端 |
| VAD | Silero VAD（ONNX Runtime CPU）+ merge_short_gaps |
| 时间戳 | VAD 级片段时间戳（已验证基线）；Qwen3-ForcedAligner 待 Spike 验证 |
| 后处理 | wetext ITN（默认开启）+ CEC4-4B 语义纠错（可选，默认关闭）+ verify 校验 |
| 音频处理 | FFmpeg / ffprobe（项目内捆绑静态版，tools/ffmpeg/） |
| 数据 | SQLite（WAL）+ 项目文件目录 + JSON 快照 |
| 进度 | SSE，结构化事件协议 `@@EVENT@@` |

## 目录结构

```text
audio-transcriber/
├── backend/            # FastAPI 后端 + ASR 核心包（复用 local-asr-portable 验证过的逻辑）
│   ├── app/            # api / services / repositories / worker / asr / postprocess
│   └── .venv/          # 主 Python 虚拟环境（torch CUDA + transformers + qwen-asr）
├── frontend/           # Vue 3 工作台
├── scripts/            # install.bat / start.bat / stop.bat / download_models.bat
├── config/             # default.toml
├── tools/ffmpeg/       # 捆绑的 FFmpeg 静态构建
├── models/             # 本地模型权重（Qwen3-ASR-0.6B 等）
├── data/               # projects / database / temp / logs
└── docs/               # 落地文档
```

## 快速开始

```bash
# 1. 安装依赖（首次）
scripts\install.bat

# 2. 下载模型（首次，走 hf-mirror 镜像）
scripts\download_models.bat

# 3. 启动
scripts\start.bat
```

## 开发现状（2026-11-06）

- **阶段 1 后端闭环**：完成。导入 → ASR 子进程 → 后处理 → 片段入库 → 编辑/查找替换/撤销/导出/SSE 全链路，290 单测通过。
- **阶段 2 前端工作台**：完成。项目列表、工作台（播放器联动、虚拟滚动片段列表、防抖自动保存、查找替换、任务进度）已实现；Playwright E2E（`frontend/e2e/workbench.spec.ts`）在 dev（5173）与生产托管（8730）双模式通过。
- **剩余**：CEC4-4B BF16 语义纠错补测（默认关闭的可选项）。

### 转写模型选择（工作台任务栏）

开始 / 重新转写前，可在任务栏下拉框选择模型，默认 `0.6b`：

| 模型键 | 本地目录 | 定位 | 本机实测（RTX 4060 8GB） |
|---|---|---|---|
| `0.6b`（默认） | `models/Qwen3-ASR-0.6B` | 快速，日常转写 | 显存约 2.0GB，8.4s 音频约 15s |
| `1.7b` | `models/Qwen3-ASR-1.7B` | 高质量，难音频 / 终稿重转 | 显存峰值 6.3GB，8.4s 音频约 18-20s（含模型加载） |

- 模型选择只影响**新创建**的任务，同一项目可换模型重转，旧结果保留在各自 run 下。
- 未安装、或下载未完成（目录内仍有 `*.incomplete`）的模型在下拉框置灰；后端创建任务时返回 409，不会跑到一半才失败。
- 1.7B 安装（约 4.4GB）：

```bat
:: 1) 下载（直连；系统代理不可用时会卡住，可加 env -u http_proxy 等）
backend\.venv\Scripts\modelscope.exe download --model Qwen/Qwen3-ASR-1.7B --local_dir models\Qwen3-ASR-1.7B

:: 2) 补齐 tokenizer.json / processor_config.json（ModelScope 快照缺这两个文件）
backend\.venv\Scripts\python.exe backend\scripts\prepare_model_dir.py models\Qwen3-ASR-1.7B
```

### 前端开发与验证

```bash
# 生产模式（日常使用）：start.bat 后浏览器访问 http://127.0.0.1:8730
# 前端改动后重新构建（FastAPI 托管 frontend/dist）
cd frontend && npm run build

# 开发模式：后端 8730 + vite dev 5173（/api 代理到 8730）
cd frontend && npm run dev

# E2E 验证（需后端已启动；默认验 dev，可切换验生产托管）
cd frontend && npx playwright test
cd frontend && E2E_BASE_URL=http://127.0.0.1:8730 npx playwright test
```

## 硬件环境（本机 Spike 基准）

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU（8GB 显存）
- 驱动 CUDA：13.4（torch 走 cu12x wheel）
- Python：3.12（独立 venv）
- 注意：所有 macOS MPS 实测数据（CER/RTF/内存）不能直接作为本机预期，Windows CUDA 性能数据为 Spike 阶段必答项。
