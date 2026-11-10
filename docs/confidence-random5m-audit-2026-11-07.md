# 随机组装 2×5 分钟音频的低置信标亮准确度分析（2026-11-07）

## 1. 数据与合成

从 Chinese-LiPS `processed_test`（3908 条、21 个说话人、16 kHz 单声道）中
**随机打乱后顺序抽取**，每段音频内部跨说话人、跨话题；语句间插入 300 ms 静音。

| 音频 | 语句数 | 说话人数 | 时长 | 参考文本 |
|---|---|---|---|---|
| random1 | 37 | 17 | 303.7 s | `data/bench/random5m/random1.txt` |
| random2 | 34 | 14 | 310.7 s | `data/bench/random5m/random2.txt` |

合成脚本：`backend/scripts/build_random_bench.py`（seed=20261107，可复现）。

## 2. 转写配置

Qwen3-ASR-0.6B + fsmn-vad，cuda，balanced 切块 60 s，批推理；
默认低置信策略 `balanced`（单字 token + 段级门控 + 全局 4% 预算），
`--dump-token-features` 用于离线重放。

## 3. 字符级指标（默认 4% 预算）

| 音频 | 信息字符 | 错误字符 | 标亮 | TP | 覆盖率 | P | R | AUROC | AP |
|---|---|---|---|---|---|---|---|---|---|
| random1 | 1225 | 56 | 49 | 18 | 4.00% | 36.7% | 32.1% | 0.756 | 0.283 |
| random2 | 1300 | 38 | 53 | 16 | 4.08% | 30.2% | 42.1% | 0.812 | 0.178 |
| **合计** | **2525** | **94** | **102** | **34** | **4.04%** | **33.3%** | **36.2%** | **0.776** | **0.203** |

- 基础字符错误率 94/2525 = **3.72%**；随机标亮 4% 的期望精确率约 3.7%，
  当前策略精确率约为随机基线的 **9 倍**（lift ≈ 8.95×）。
- AUROC 与 bench_10min（0.772）一致，说明 entropy 排序能力稳定，不是单一音频过拟合。
- 段级：`review_order` 前 5 覆盖全部 5 个含错段（本数据每段均含错，区分度有限）。

## 4. 覆盖率–精确率扫描（跨两段合并）

| 目标覆盖率 | 实际标亮 | TP | P | R |
|---|---|---|---|---|
| 1% | 27 | 15 | 55.6% | 16.0% |
| 2% | 52 | 23 | 44.2% | 24.5% |
| 4% | 102 | 34 | 33.3% | 36.2% |
| 8% | 116 | 35 | 30.2% | 37.2% |
| 10% | 116 | 35 | 30.2% | 37.2% |

8%/10% 实际只有 116 个候选（占 4.6%）：单字 token + 熵门控 + 分数下限
已耗尽全部候选，继续放宽预算不会增加标亮。若需要更高召回，应启用
`--forced-aligner --low-conf-mode recall`，而不是单纯放大 ratio。

## 5. 分析与结论

1. **精确率随覆盖率单调下降，召回上升后饱和**：1% 档 P 55.6% 适合「快速复核」，
   4% 档是平衡点，8%+ 无增量。
2. **两段音频差异来自错误率**：random1 错误率 4.6%、random2 2.9%；
   固定 4% 预算下，错误越密集，命中率越高（P 36.7% vs 30.2%）。
3. **AUROC 0.78 说明排序信号本身可用**，剩余误差主要来自：
   - 低熵同音/数字错误（约 1/3 错误落在熵分布中低区）；
   - 多字 token 承担的错误无法逐字定位（策略主动放弃这类标亮）；
   - 段级门控会跳过「整段熵都低」的段落。
4. 与 bench_10min 对比：P 33.3%、R 36.2% vs P 32.6%、R 38.7%，
   指标稳定，可作为 V1.1 默认策略的回归基线。

## 6. 复现命令

```bash
cd backend

# 1) 合成两段 5 分钟音频
.venv/Scripts/python.exe scripts/build_random_bench.py \
    --count 2 --minutes 5 --seed 20261107 --out-dir ../data/bench/random5m

# 2) 转写（各一次，含完整信号）
.venv/Scripts/python.exe scripts/qwen3_asr_transcribe.py \
    --model ../models/Qwen3-ASR-0.6B --audio ../data/bench/random5m/random1.wav \
    --output ../data/bench/random5m/run_r1b/raw.txt --device cuda --language Chinese \
    --segmentation vad --vad-model ../models/funasr-fsmn-vad --batch-auto \
    --chunk-strategy balanced --target-chunk 60 --batch-max 8 --dump-token-features

# 3) 对账（支持多文件汇总 + 覆盖率扫描）
.venv/Scripts/python.exe scripts/confidence_audit_prod.py \
    --stats ../data/bench/random5m/run_r1b/raw.stats.json \
            ../data/bench/random5m/run_r2d/raw.stats.json \
    --reference ../data/bench/random5m/random1.txt \
                ../data/bench/random5m/random2.txt
```

产物：`data/bench/random5m/{randomN.wav,randomN.txt,manifest.json}`、
`run_r1b/`、`run_r2d/`、`run_r1b/confidence_audit.json`。

## 7. 本次顺带修复

- `token_char_spans` 逐 token 解码可能比原文长，导致 `keep_char("")` 崩溃
  (`unicodedata.category` 报错)；现已截断越界 span，并让 `keep_char("")` 返回 False。
  新增 `OverlongTokenizer` 单测锁定。