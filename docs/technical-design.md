# Windows 落地技术设计（指针文档）

本项目的完整技术设计以 macOS 调研阶段的最终版为准：

- **源文档**：`D:\work\workspace\local-asr-portable\docs\本地音频转文字与校对工具_技术详细概要设计_V3.0_最终版.md`

本目录记录 Windows 落地过程中的增量决策与 Spike 结果，不复制源文档内容。

## 已确定的 Windows 落地决策（2026-11-06）

| 决策点 | 决定 | 说明 |
|---|---|---|
| Python | 3.12（winget 用户级安装） | 符合 V3.0 基线与 Qwen 官方建议 |
| torch | CUDA wheel（cu128 index） | RTX 4060 8GB，驱动 CUDA 13.4；不从 PyPI 装 CPU 版 |
| FFmpeg 分发 | **项目内捆绑静态版**（`tools/ffmpeg/`） | 解决附录 B 问题 5：目录式便携交付，不依赖系统安装 |
| venv 数量 | 首版 1 个主 venv（`backend/.venv`） | Windows 无 MLX，CEC4-4B 纠错走 CUDA BF16 + Transformers，预计可与主环境合并（Spike 问题 3 验证）；编排层保留跨 venv 能力 |
| 前端 | Vue 3 + TS + Vite + Router + Pinia + Element Plus + @tanstack/vue-virtual | V3.0 §1.2 选型 |
| API 端口 | 127.0.0.1:8730 | 避开常用端口 |

## Windows Spike 待答项（源文档附录 B）——已完成，详见 `docs/spike-report.md`

1. ✅ Qwen3-ASR-0.6B RTF 0.086（单条）/ 0.030（batch 4）；峰值显存 2.02GB；batch 上限 8（推荐 4）
2. ✅ **Qwen3-ForcedAligner 可用，时间戳升级为字符级方案**（3/3 校验合规，双模型 3.57GB）
3. ⏸ CEC4-4B BF16 未测（可选后处理，默认关闭，非阻塞项）
4. ✅ 60s 块长成立（20 分钟 RTF 0.043，0 空转写）
5. ✅ batch 4 加速比 2.87×（> MPS 1.98×）
6. ✅ 单次推理上限 ≈ 300s（与 MPS 一致；600s 走 WDDM 共享显存慢 20 倍，1200s OOM）
7. ✅ 长音频端到端稳定（20 分钟实测；60 分钟外推约 2.6 分钟）

**Spike 退出条件达成**（除可选项 3），可进入阶段 1 后端闭环开发。
