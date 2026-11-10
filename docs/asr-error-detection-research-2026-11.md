# ASR 错误字段识别调研（截至 2026-11）

> 目标：回答「音频转文字后，哪些字/词应当被标记为可疑（低置信）」，并给出适用于本项目
> （Qwen3-ASR-0.6B + 本地部署 + 中文长音频）的落地方案。
>
> 检索时间：2026-11-07。检索渠道：arXiv API（30+ 组查询）、GitHub Search API、论文全文抓取。
> 原始检索结果：`backend/output/research/arxiv_results{,2,3}.json`。

---

## 0. 结论先行

1. **不存在单一信号能可靠识别 ASR 错误。** 2025-2026 年的共识是：输出概率（logprob/entropy）
   只是弱信号，必须与控制覆盖率、解码信息、对齐信息、跨模型/内部表示等结合。
2. **词级置信度做到 P 40%-55% 已属商业 SOTA 水平**（2503.15124 对 9 家商业 ASR + Whisper 的
   实测），我们项目当前字符级 P 33.3% 与这一量级一致，并非实现有 bug，而是信号本身的上限。
3. **高亮并不自动提升纠错效率。** 2503.15124 的 36 人用户研究表明：高亮与否对最终 WER 无显著
   影响（p=0.767）；覆盖率约 4% 的「平衡阈值」最受偏好，覆盖 13%-21% 的高阈值反而干扰用户。
4. **最值得投入的增量信号**（按性价比）：
   - 低成本：token 长度/单字 token 门控（已落地）、全局覆盖率控制、文本侧异常特征、
     ForcedAligner 时间对齐异常；
   - 中成本：beam search 的 rank/score（SR-CEM 路线）、多次采样一致性；
   - 高成本但上限最高：跨模型分歧、decoder hidden-state probing（需标注训练）。
5. **产品上应把「标亮」定位为复核优先级，而不是错误判定。** 覆盖率控制在 4% 左右、句级聚合、
   点击即听，比追求召回更有价值。

---

## 1. 本项目的实测基线（bench_10min，LiPS 参考）

数据：10 分钟合成音频，Qwen3-ASR-0.6B greedy 解码，2220 个归一化字符，83 个真实错误（3.74%）。
标签由 `conservative_norm + align_ops + apply_filler_rules` 与参考文本逐字符对齐得到。

| 指标 | 数值 |
|---|---|
| entropy 单特征 AUROC / AP | 0.662 / 0.152 |
| neg_logprob AUROC / AP | 0.661 / 0.129 |
| neg_margin AUROC / AP | 0.662 / 0.124 |
| 三特征融合 | 无增益 |
| 单字 token 字符错误率 | 10.08%（615 字 / 62 错） |
| 多字 token 字符错误率 | 1.31%（1605 字 / 21 错） |
| 熵 P99+ 分桶错误率 | 39.13% |
| 熵 P95-P99 分桶错误率 | 16.3%-17.8% |
| 熵 P90-P95 分桶错误率 | 6.25% |
| bottom-50% 低熵区错误占比 | 32.5%（同音/近形错，模型很自信） |
| 段内 90%-100% 位置错误率 | 11.4%（chunk 边界效应） |

策略迭代：

| 策略 | 标亮字符 | TP | FP | FN | P | R |
|---|---|---|---|---|---|---|
| 旧：每段 top-10% | 218 | 34 | 184 | 49 | 15.6% | 41.0% |
| 新：单字 token + 熵≥0.2 + 段级门控 | 66 | 22 | 44 | 61 | 33.3% | 26.5% |
| 新策略（服务端真实入库） | 71 | 20 | 51 | 55 | 28.2% | 26.7% |

关键教训：

- 标亮比例（10%）远高于真实错误率（3.74%），随机基线精确率只有 3.7%，理论上限约 37%；
- 同一 token 的多字共享一个熵，按字符排名会把整个词标上，制造大量误报；
- 零错误段被「每段 top-10%」强制标亮，是旧策略精确率低的主因。

---

## 2. 文献证据（按信号族）

### 2.1 输出概率与校准（最基础，也最弱）

- **2503.15124** *Evaluating ASR Confidence Scores for Automated Error Detection in
  User-Assisted Correction Interfaces*（2025-03-19）
  - 9 家商业 ASR + Whisper，394,348 个词对；
  - 最优阈值下词级 **P 0.41-0.55、R 0.36-0.64、F1 0.33-0.55、AUC 0.68-0.87**；
  - 36 人用户研究（德语）：高亮组与对照组最终 WER 0.47 / 0.48 / 0.45（p=0.767），
    **高亮未提升纠错效率**；
  - 平衡阈值覆盖约 **4%**，高阈值覆盖 13%-21%；16 人偏好无高亮，18 人偏好平衡阈值；
  - 论文结论：置信度与错误的相关性不足，高亮常被忽略或造成干扰。
  - 对我们的意义：这组数字与我们实测（AUROC 0.66、P 33%）几乎重合，说明当前实现已接近
    「纯概率信号」的天花板。

- **2502.13446** *Adopting Whisper for Confidence Estimation*（2025-02-19）
  - 微调 Whisper 直接输出词级置信度；Whisper-large 全面超过 CEM 基线，Whisper-tiny 与之相当。
  - 启示：把置信度估计当作独立任务微调，优于复用 ASR 的 softmax。

- **2607.29299** *Leveraging Beam Search Information for Confidence Estimation in E2E ASR*
  - 用 beam search 的 **scores + ranks** 构造 token/word 置信特征，架构无关；
  - token 级 ECE 0.30% / MCE 4.50%，softmax 基线 1.75% / 20.04%；
  - word 级 ECE 0.35% / MCE 8.17%，基线 1.67% / 17.91%。
  - 启示：**解码过程本身的信息比最终 softmax 更有校准价值**，且无需训练。
  - 本项目可行性：`qwen_asr` 的 `generate(**kwargs)` 会把 `num_beams`、`output_scores`
    原样透传给 thinker，技术上可直接开启。

### 2.2 解码搜索 / n-best / lattice

- 经典工程：Kaldi lattice posterior、NeMo `ConfidenceAnnotate`（n-best 一致性）。
- 本项目此前探测：n-best disagreement 特征 AP 0.24（高于 entropy 的 0.15），但成本高、
  在正常音频上噪声大，当时只作为兜底风险标签。结合 SR-CEM 的结论，值得用 **beam rank**
  （而非完整 n-best 文本编辑距离）重新尝试。
- 注意：Qwen3-ASR 是 LLM-based 自回归模型，没有 CTC lattice，Kaldi 路线不直接适用。

### 2.3 对齐与时序

- **whisper-timestamped**（linto-ai/whisper-timestamped，约 2.8k stars，2026-09 仍活跃）
  - DTW + cross-attention 生成词级时间戳，并为每个词/段输出 0-1 confidence；
  - `compute_word_confidence=True` 为默认行为。
  - 启示：词级时间边界稳定性本身可作为置信特征（对齐越「勉强」，越可疑）。
- **Qwen3-ForcedAligner-0.6B**（2601.21337 技术报告配套）
  - 本项目已具备强制对齐能力；可提取每个字的时长、边界、对齐分数；
  - 异常模式：字时长过短/过长、静音段却有字、相邻字边界重叠 → 复核候选。
- **2601.18220** *LLM-ForcedAligner*（2026-01-26）：非自回归 LLM 强制对齐，长音频可用。

### 2.4 内部表示 probing（无参考，上限高）

- **2606.23060** *From Text Metrics to Model Internals: A Study of Whisper ASR Hallucination
  Detection*（2026-06-22）
  - 文本分类器召回高，但无参考场景退化；
  - LLM 检测精度提升有限，不如轻量文本方法；
  - **decoder hidden state probing 在无参考下最强**；text + internal late-fusion 最佳。
- **2604.19565** *Detecting Hallucinations in SpeechLLMs at Inference Time Using Attention Maps*
  （2026-04-21）
  - AUDIORATIO / AUDIOCONSISTENCY / AUDIOENTROPY / TEXTENTROPY + logistic regression；
  - Qwen-2-Audio、Voxtral-3B；PR-AUC 最多提升 0.23；约 100 个 attention head 已足够；
  - 模型相关，需要任务特定训练。
- **2610.25351** *Probing and Mitigating Hallucinations in Speech-augmented Language Models for
  ASR via Small Language Models*（2026-10-18，最新）
  - 用小型 LM 对语音增强 LM 的隐藏状态做 probing，检测并缓解幻觉。
- 对本项目：Qwen3-ASR 的 thinker 是 decoder-only，`output_hidden_states=True` 可取隐状态；
  可用 LiPS 参考自动构造标签，训练轻量 logistic/MLP probe。代价是需要为每个模型版本重训。

### 2.5 跨模型 / 多次采样一致性（无参考、稀疏、可定位）

- **2604.14152** *From Black Box to Glass Box: Cross-Model ASR Disagreement to Prioritize
  Review in Ambient AI Scribe Documentation*（2026-03-02）
  - 8 个 ASR 系统、76,398 个 token；
  - **72.1% 的 token 在 7-8 个模型中近乎一致，2.5% 落在 0-3 个模型一致的高风险带**；
  - 跨模型 ICC 仅 0.131，说明各模型失败模式高度异质；
  - 结论：跨模型分歧是**无参考、稀疏、可定位**的复核优先级信号。
  - 对本项目：可引入第二个本地模型（Paraformer-zh / Whisper-small / Qwen3-ASR-1.7B），
    只在分歧处标亮，覆盖率天然稀疏。
- 多次采样一致性：同一模型 temperature 0.7 采样 2-4 次，按字符编辑距离/投票分歧打分。
  成本线性增加，但对同音错可能有效（模型在两种读法间摇摆）。

### 2.6 幻觉 / 插入删除类错误的专用信号

- Whisper 生态的经典启发式：`no_speech_prob`、`compression_ratio`、`avg_logprob`、
  重复 n-gram 检测。
- **2608.15940** *The Null Token Knows: Reducing Message-Free Hallucination in ASR and NMT*
  （2026-08-16）
- **2608.30776** *Likelihood-Constrained Acoustic Reranking for Training-Free Hallucination
  Mitigation in LLM-Based ASR*（2026-08-31）
- **2606.07473** *Whisper Hallucination Detection and Mitigation via Hidden Representation
  Steering and Sparse AutoEncoders*（2026-06-05）
- **2609.04561** *Reducing Hallucinated Transcripts in Whisper via Hallucination Space Projection*
  （2026-09-03）
- **2609.32560** *How to Reduce Whisper Hallucination*（2026-09-26）
- **2606.23048** *HALAS: A Human-Annotated Dataset of Hallucinations of Modern ASR Systems*
  （2026-06-22）
- 说明：本项目当前错误以替换（S）为主，幻觉类（插入/整段编造）占比低；这些方法可作为
  长音频、静音段、音乐段的专项兜底。

### 2.7 置信度引导的后处理

- **2509.25048** *Confidence-Guided Error Correction for Disordered Speech Recognition*
  （2025-09-29）
  - 把置信度注入 prompt，微调 LLaMA 3.1 做纠错；
  - WER 相对降低 10%（Speech Accessibility）/ 47%（TORGO）。
  - 启示：置信度的更高价值不在「标亮」，而在「驱动 LLM 纠错」；但需要纠错模型与评测。

### 2.8 置信度估计的专门训练目标

- **TruCLES**（Interspeech 2025 tutorial 关联仓库 `madhavlab/2025_trucles_interspeech`）
  - CEM 训练目标 = 真实类概率 + 预测/参考词级相似度；
  - CTC：AUC-ROC 0.8383 / AUPRC 0.9660；RNN-T：AUC-ROC 0.9017。
  - 需要帧级对齐与训练流程，不适合当前「复用现成模型」的约束。

---

## 3. 方法对比矩阵

| 信号族 | 代表工作 | 粒度 | 额外成本 | 是否需训练 | 预期收益 | 本项目适配 |
|---|---|---|---|---|---|---|
| softmax logprob / entropy / margin | 2503.15124 | token/字符 | 无 | 否 | 弱-中（AUROC 0.66） | 已落地 |
| token 长度 / 单字 token | 本项目实测 | 字符 | 无 | 否 | 中（错误率 7.7 倍差） | 已落地 |
| beam score / rank | 2607.29299 (SR-CEM) | token/词 | 4-8x 解码 | 否 | 中-强（校准显著改善） | 高 |
| n-best / lattice 一致性 | Kaldi、NeMo | 词 | 高 | 否 | 中 | 中 |
| 多次采样一致性 | self-consistency | 字符/词 | 2-4x 推理 | 否 | 中 | 高 |
| 强制对齐异常 | whisper-timestamped、Qwen3-ForcedAligner | 字/词 | 低-中 | 否 | 中 | 高 |
| 跨模型分歧 | 2604.14152 | 词/字符 | 1 个额外模型 | 否 | 强（稀疏高精度） | 中-高 |
| hidden-state probing | 2606.23060、2610.25351 | token/段 | 中 | 是（轻量） | 强（无参考最佳） | 中 |
| attention-map 特征 | 2604.19565 | 段/token | 中 | 是 | 中-强 | 低-中 |
| 文本侧异常（重复、标点、数字） | Whisper 启发式 | 段/字符 | 无 | 否 | 弱-中 | 高 |
| 专用 CEM 微调 | 2502.13446、TruCLES | 词 | 训练 | 是 | 强 | 低（无标注管线） |

---

## 4. 对 Qwen3-ASR-0.6B 本地部署的推荐路线

### P0（已完成）

- token entropy + 单字 token 过滤 + 段级门控（见 `docs/confidence-highlight-tuning.md`）。
- 字符级 P 33.3% / R 26.5%，服务端 P 28.2% / R 26.7%。

### P1（1-2 天，无需额外模型）— ✅ 已落地（见 §9）

1. **全局覆盖率控制**
   - 在绝对阈值（熵 ≥ 0.2）之上，增加全片候选上限：按分数取 top **4%**（可配置 2%-8%）；
   - 依据：2503.15124 中平衡阈值覆盖率约 4%，用户偏好最高；
   - 实现：转写结束后对全部候选统一排序截断，而不是每段独立 top-K。
2. **文本侧异常特征**
   - 重复 n-gram（连续 3 字重复 ≥2 次）、异常标点、句末截断、纯数字/英文串；
   - 中英混排、数字、专名可用正则 + 字频先验；
   - 这些特征用于**段级风险**，不直接逐字标亮。
3. **ForcedAligner 时间对齐异常**
   - 字时长 < 40ms 或 > 1.2s、静音段出现文字、相邻边界重叠；
   - 与 entropy 做 OR/加权，成本低，适合捕获「语速突变/吞字」。
4. **段级排序**
   - 段内最大单字熵 + 段尾 10% 位置加成（实测错误率 11.4%）+ 段平均熵；
   - 工作台按风险从高到低排序，提供「下一处可疑」按钮。

### P2（约 1 周，推理成本增加）— ✅ 已落地（beam rank 实测无效，默认关闭）

1. **beam rank 特征（首选）**
   - `num_beams=4`（先 4，再评估 8），保留每步 beam 的 score 与 rank；
   - 特征：最终 token 在 beam 中的 rank、top1-top2 score margin、beam 间一致性；
   - 与 entropy 拼接后用简单逻辑回归/分位数融合，先在 bench_10min 上量 AUROC/AP；
   - 注意 `generate` 在 beam search 下 `scores` 形状与 greedy 不同，需要按步对齐 token。
2. **二次采样一致性（备选）**
   - 同 prompt 采样 2 次（temperature 0.7、top_p 0.9），按字符对齐计算分歧；
   - 只在高分歧且 entropy 中等的字符上标亮；
   - 预期能捞回一部分「低熵同音错」（当前 32.5% 的错误落在 bottom-50% 熵区）。

### P3（2-4 周，需要额外依赖/训练）

1. **跨模型分歧**：引入 Paraformer-zh 或 Whisper-small 作为第二意见，只在分歧处标亮；
   覆盖率天然在 2%-5%，精确率通常显著高于单模型置信度。
2. **hidden-state probing**：用 LiPS 参考自动构造字符级标签，取 thinker 最后几层隐状态，
   训练 logistic regression / 2 层 MLP；按模型版本重训，避免跨模型漂移。
3. **置信度驱动纠错（远期）**：参考 2509.25048，把低置信片段交给 LLM 结合上下文重写，
   但必须保留原文与 diff，且需要独立评测。

### P4（产品闭环）

- 记录用户对每个标亮字符的「确认/修改/忽略」；
- 用这些反馈校准阈值（按项目/说话人/领域分组）；
- 定期用 bench_10min + LiPS 回归，防止调参过拟合单一音频。

---

## 5. 评估协议（避免自欺）

1. **固定测试集**：bench_10min（LiPS 参考）+ 至少 2 段真实录音（不同说话人/领域）。
2. **字符级指标**：P / R / F1、AUROC、AP；同时报告**覆盖率**与**标亮绝对字数**。
3. **覆盖率-精确率曲线**：按 1%/2%/4%/8%/10% 覆盖率分别报告 P/R，避免只挑一个点。
4. **段级指标**：有错段召回率、平均复核字数/分钟。
5. **消融**：每个新特征单独 AUROC 增量 + 融合后 P/R 增量；无增量不上线。
6. **用户验证**：至少 5 人、每人 2 段，比较「有标亮/无标亮」的纠错耗时与最终 WER
   （参考 2503.15124 的结论：不要假设标亮一定有用）。

---

## 6. 产品交互建议

- **标亮 = 复核优先级，不是错误判定**；文案避免「此处有错」，改为「建议核对」。
- 覆盖率默认约 4%，提供「更多/更少」两档（2% / 8%）。
- 字符级虚线 + 句级风险色两级呈现；句级用于导航，字符级用于定位。
- 点击标亮字符即播放对应时间片（ForcedAligner 时间戳），减少来回找音频。
- 修正后即时消除标亮，并记录修正内容作为反馈。
- 对零错误段不标亮；宁可少标，不要整段飘红。

---

## 7. 局限与风险

- 文献以英语为主，中文同音字/近形字、长音频、Qwen3-ASR 的公开研究很少；
- 本项目的字符级指标与论文的词级指标不可直接比较，只能看量级与趋势；
- beam search / 多次采样会增加延迟与显存，0.6B 在 CPU 上可能不可接受，需实测；
- hidden-state probing 与跨模型分歧都依赖具体模型，换模型需重标定；
- 32.5% 的错误落在低熵区（同音/近形），纯概率与熵类信号无法检出，这是硬上限；
- 用户研究样本小（36 人、单一语言），结论需在本项目场景复核。

---

## 8. 参考文献与仓库

**论文（arXiv）**

- 2503.15124 — Evaluating ASR Confidence Scores for Automated Error Detection in User-Assisted
  Correction Interfaces — https://arxiv.org/abs/2503.15124
- 2502.13446 — Adopting Whisper for Confidence Estimation — https://arxiv.org/abs/2502.13446
- 2607.29299 — Leveraging Beam Search Information for Confidence Estimation in E2E ASR —
  https://arxiv.org/abs/2607.29299
- 2604.14152 — From Black Box to Glass Box: Cross-Model ASR Disagreement to Prioritize Review —
  https://arxiv.org/abs/2604.14152
- 2606.23060 — From Text Metrics to Model Internals: A Study of Whisper ASR Hallucination
  Detection — https://arxiv.org/abs/2606.23060
- 2604.19565 — Detecting Hallucinations in SpeechLLMs at Inference Time Using Attention Maps —
  https://arxiv.org/abs/2604.19565
- 2610.25351 — Probing and Mitigating Hallucinations in Speech-augmented Language Models for ASR
  via Small Language Models — https://arxiv.org/abs/2610.25351
- 2509.25048 — Confidence-Guided Error Correction for Disordered Speech Recognition —
  https://arxiv.org/abs/2509.25048
- 2601.21337 — Qwen3-ASR Technical Report — https://arxiv.org/abs/2601.21337
- 2601.18220 — LLM-ForcedAligner — https://arxiv.org/abs/2601.18220
- 2608.15940 — The Null Token Knows — https://arxiv.org/abs/2608.15940
- 2608.30776 — Likelihood-Constrained Acoustic Reranking for Training-Free Hallucination
  Mitigation in LLM-Based ASR — https://arxiv.org/abs/2608.30776
- 2606.07473 — Whisper Hallucination Detection and Mitigation via Hidden Representation Steering
  and Sparse AutoEncoders — https://arxiv.org/abs/2606.07473
- 2609.04561 — Reducing Hallucinated Transcripts in Whisper via Hallucination Space Projection —
  https://arxiv.org/abs/2609.04561
- 2609.32560 — How to Reduce Whisper Hallucination — https://arxiv.org/abs/2609.32560
- 2606.23048 — HALAS: A Human-Annotated Dataset of Hallucinations of Modern ASR Systems —
  https://arxiv.org/abs/2606.23048

**代码 / 工程**

- whisper-timestamped — https://github.com/linto-ai/whisper-timestamped
- TruCLES（Interspeech 2025 tutorial 关联） —
  https://github.com/madhavlab/2025_trucles_interspeech
- Interspeech 2025 confidence estimation tutorial —
  https://github.com/ThishRaj/interspeech_tutorial_2025
- E2E_ASR_Confidence_Estimation —
  https://github.com/KrishnaDN/E2E_ASR_Confidence_Estimation
- BiLatticeRNN-Confidence — https://github.com/alecokas/BiLatticeRNN-Confidence
- Qwen3-ASR — https://github.com/QwenLM/Qwen3-ASR
- NeMo ConfidenceAnnotate —
  https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html
- faster-whisper（词级 probability）— https://github.com/SYSTRAN/faster-whisper

---

## 9. 落地实现与实测（2026-11-07）

P1/P2 已实现并接入生产转写链路，参数与实测如下。

### 9.1 代码位置

| 模块 | 路径 | 职责 |
|---|---|---|
| 生产置信度模块 | `backend/src/local_asr/asr/confidence.py` | token→字符映射、文本/对齐特征、打分、全局预算、段级风险 |
| 转写脚本 | `backend/scripts/qwen3_asr_transcribe.py` | beam/采样/强制对齐特征采集，写 `low_conf_spans`、`risk_flags`、`review_priority`、`review_order` |
| 入库服务 | `backend/src/app/services/segment_service.py` | 去重裁剪后平移 `low_conf_spans`，写入/读出 `review_priority` |
| 离线实验 | `backend/scripts/confidence_p1p2_bench.py` | 逐段重跑 + 特征采集 + AUROC/AP/覆盖率曲线 |
| 生产审计 | `backend/scripts/confidence_audit_prod.py` | 用 LiPS 参考对账实际 stats 的 P/R 与段级排序 |

### 9.2 默认策略（balanced）

1. 候选：只取**单字 token**（多字 token 整词误报，bench 实测单字错误率 10.1% vs 多字 1.3%）；
2. 段级门控：段内单字最高熵 < 0.7 且无 2 个以上对齐异常 → 整段不标；
3. 分数下限 0.1；
4. 全局预算：`ceil(全片信息字符数 × --low-conf-ratio)`，默认 4%（建议范围 2%-8%）；
5. 段级风险：`max_single_entropy + 0.3×段尾max + 0.1×mean + 0.2×文本异常数 + 0.15×对齐异常数`，
   写出 `review_flags`（`low_conf_review` / `text_anomaly` / `alignment_anomaly` / `boundary_uncertain`）
   与 `review_priority`；stats 顶层 `review_order` 给出复核顺序。

### 9.3 可选档位（CLI）

```bash
# 默认：仅 entropy，4% 预算（无额外推理成本）
python scripts/qwen3_asr_transcribe.py ... --low-conf-mode balanced --low-conf-ratio 0.04

# quality：采样一致性参与打分（推理约 ×3），小预算精确率高
... --sample-check 2 --low-conf-mode quality --low-conf-ratio 0.01

# recall：强制对齐时长异常参与打分（需 ForcedAligner），大预算召回高
... --forced-aligner ../models/Qwen3-ForcedAligner-0.6B \
    --low-conf-mode recall --low-conf-ratio 0.10
```

`--beam-size` 保留但不推荐：bench 实测 `beam_rank` AUROC 0.500、`beam_entropy` 0.491，
在本模型上无增量信息，且 beam>1 会让推理成本翻倍以上。

### 9.4 bench_10min 实测（2208 信息字符 / 75 错）

| 策略 | 覆盖率 | 标亮 | TP | P | R | 推理耗时 |
|---|---|---|---|---|---|---|
| balanced 4%（默认） | 4.03% | 89 | 29 | 32.6% | 38.7% | 20.3 s |
| quality 4% + 采样×2 | 4.03% | 89 | 29 | 32.6% | 38.7% | 68.0 s |
| quality 1% + 采样×2 | 1.04% | 23 | 14 | 60.9% | 18.7% | 68.2 s |
| recall 10% + 对齐 | 10.01% | 221 | 56 | 25.3% | 74.7% | 25.2 s |

对比旧 P0 策略（P 15.6% / R 41.0%，约 209 字标亮）：默认档精确率翻倍、标亮量减半、召回基本持平。
段级排序：9 个含错段全部进入 `review_order` 前 9。

### 9.5 已知限制

- 4% 预算下 quality 档与 balanced 完全相同（采样一致性只在小预算时改变排序），
  因此默认不开启，避免 3 倍推理成本；
- 文本侧异常只做段级提示，不逐字标亮；
- 覆盖率与精确率仍受模型本身限制，单字同音错（低熵）难以召回。
  下一步按 §4 P3 引入跨模型分歧或 hidden-state probing。

---

## 10. 一句话总结

**把置信度当「排序信号」而非「判定信号」**：用 token 级 entropy 打底，叠加解码/对齐/跨模型
信息提升排序质量，把标亮覆盖率压到 4% 左右，再用交互与反馈闭环把「复核」做得足够便宜。
单纯提高标亮比例或换一个置信度公式，不会突破 P 40%-55% 的上限。