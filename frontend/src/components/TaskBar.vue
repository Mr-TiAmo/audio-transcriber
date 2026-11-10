<script setup lang="ts">
// 任务视图（§12.1 任务状态切换）：空闲 / 处理中（含后处理子阶段）/ 完成 / 失败 / 已取消
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'

import { api } from '@/api/client'
import { useWorkbenchStore } from '@/stores/workbench'
import type { AsrModel } from '@/types'

const store = useWorkbenchStore()

// 模型下拉：默认 0.6B 快速；1.7B 高质量（显存与耗时更高）
const models = ref<AsrModel[]>([])
const selectedModel = ref('0.6b')
// 快速模式：跳过 ForcedAligner 字符级对齐（省 2–6s），句级时间戳退回按字数插值
const fastMode = ref(false)
// 交叉验证：默认关闭；勾选后第二模型（FunASR，CPU）整片转写，
// 仅核验低置信词组/词语（两模型一致的词取消标亮）。第二模型与主推理
// 并行，短音频额外等待约 30–48s，长音频大部分被主推理覆盖。
const crossCheckMode = ref(false)

async function refreshModels() {
  try {
    models.value = await api.listModels()
  } catch {
    // 列表拉取失败时隐藏下拉，仍用默认模型转写
    models.value = []
  }
}

onMounted(refreshModels)

const statusMeta = computed(() => {
  switch (store.run.status) {
    case 'PROCESSING':
      return { label: '处理中', type: 'primary' as const }
    case 'COMPLETED':
      return { label: '已完成', type: 'success' as const }
    case 'FAILED':
      return { label: '失败', type: 'danger' as const }
    case 'CANCELLED':
      return { label: '已取消', type: 'info' as const }
    default:
      return { label: '空闲', type: 'info' as const }
  }
})

const starting = ref(false)

async function start() {
  if (starting.value) return
  starting.value = true
  try {
    await store.startTranscription(
      selectedModel.value, fastMode.value, crossCheckMode.value)
  } catch (err) {
    ElMessage.error(`启动失败：${(err as Error).message}`)
  } finally {
    starting.value = false
  }
}

async function cancel() {
  try {
    await store.cancelTranscription()
    ElMessage.info('已请求取消，正在终止子进程…')
  } catch (err) {
    ElMessage.error(`取消失败：${(err as Error).message}`)
  }
}

// ---------------------------------------------------------------------------
// 处理中时间提示：已处理（墙钟，从 started_at 起算）+ 预计剩余，
// 以及按时间均匀推进的进度百分比（见 rawPercent / shownPercent）。
//
// 估算基准：
// - ASR 阶段：音频时长 × RTF + 模型加载（经验值，本机实测校准），
//   窗口进度事件到达时用实际耗时反推校准（仅在实际更慢时抬高）；
// - 后处理阶段：音频时长 × 0.03（下限 4s / 上限 3min）。
// 经验系数取本机实测（2026-11-08 RTX 4060 笔记本，见 docs/performance-analysis-2026-11-08.md）：
// 0.6B——8.4s 音频约 13s（加载约 10s + 推理）、310s 音频约 51s（RTF≈0.10）；
// 1.7B 只能 batch=1，310s 实测 RTF 0.156、599s 实测 0.210（wall 152s）。
// 起初取 rtf=0.05 严重低估，会让倒计时提前归零、过早显示「超出预估」。
// 估算偏保守比偏乐观好：校准值只会抬高总时长，不会把估算拉低。
// ---------------------------------------------------------------------------
const MODEL_EST: Record<string, { rtf: number; loadMs: number }> = {
  '0.6b': { rtf: 0.10, loadMs: 10000 },
  '1.7b': { rtf: 0.20, loadMs: 14000 },
}

const nowMs = ref(Date.now())
const phaseStartMs = ref(Date.now())
let ticker: number | null = null

function startTicker() {
  if (ticker != null) return
  nowMs.value = Date.now()
  // 500ms 刷新：进度条按时间平滑推进（1s 时肉眼可见的台阶）
  ticker = window.setInterval(() => { nowMs.value = Date.now() }, 500)
}
function stopTicker() {
  if (ticker != null) {
    window.clearInterval(ticker)
    ticker = null
  }
}

watch(
  () => store.run.status,
  (status) => { if (status === 'PROCESSING') startTicker(); else stopTicker() },
  { immediate: true },
)
// 阶段切换时重置本阶段起点（后处理倒计时依赖它）。
// 倒计时预算：上一阶段已超时（截止时刻已过）才按新阶段预算重新起算；
// 未超时则保留原截止时刻，切换阶段时倒计时不会「回升」。
watch(() => store.stageText, () => {
  phaseStartMs.value = Date.now()
  if (etaDeadlineMs.value !== null && etaDeadlineMs.value <= Date.now()) {
    etaDeadlineMs.value = null
  }
})
onBeforeUnmount(stopTicker)

const startedAtMs = computed(() => {
  const raw = store.run.started_at
  const ts = raw ? Date.parse(raw) : NaN
  // 回退到本阶段起点：started_at 缺失时不至于显示 0 秒
  return Number.isFinite(ts) ? ts : phaseStartMs.value
})
const elapsedMs = computed(() => Math.max(0, nowMs.value - startedAtMs.value))

const modelKey = computed(() => {
  const name = store.run.model_name ?? ''
  if (name.includes('1.7')) return '1.7b'
  if (name.includes('0.6')) return '0.6b'
  return selectedModel.value
})

// 后处理耗时经验值：与音频时长近似线性（bench 10min≈20s，60min≈130s），
// 下限 4s（短音频的 ITN/入库固定开销）、上限 3min
const postEstimateMs = computed(() => {
  const durMs = store.project?.duration_ms ?? 0
  return Math.min(60_000, Math.max(3_000, durMs * 0.002 + 4_000))
})

// 窗口进度到达时校准 ASR 总时长估计（实际比经验值慢时以实际为准）。
// 校准值在下一次事件前保持固定，ETA 才会随时间递减而不是原地漂移。
const asrCalibMs = ref<number | null>(null)
watch(() => store.run.progress_percent, (percent) => {
  if (percent >= 5 && percent < 100) {
    asrCalibMs.value = elapsedMs.value / (percent / 100)
  }
})
// 新任务 / 重新转写：清空上一次运行留下的校准值。
// 阶段起点同时重置：中途刷新/切换项目时 stageText 可能没有变化（watch 不触发），
// 后处理倒计时会错误地从页面加载前的旧起点算起。
watch(() => store.run.id, () => {
  asrCalibMs.value = null
  phaseStartMs.value = Date.now()
  etaDeadlineMs.value = null; asrDoneTs.value = null
})

  // ASR 推理完成后的尾部预算（窗口进度 >= 99% 后，后端仍在做对齐/交叉验证比对，
  // 无额外进度事件，进度条需要按时间持续推进而非停在 85% 不动）。
  const asrDoneTs = ref<number | null>(null)
  const ASR_TAIL_BUDGET_MS = 25_000
  watch(() => store.run.progress_percent, (pct) => {
    if (pct >= 99 && asrDoneTs.value === null) {
      asrDoneTs.value = Date.now()
    }
  })

// ASR 阶段预计时长：音频时长 × RTF + 模型加载（经验值）
// 经验值保证进度条在两次窗口事件之间按时间均匀推进；
// 校准值只在「实际明显更慢」时把总时长抬高，避免 ETA 归零后长时间卡住。
const asrTotalMs = computed(() => {
  const est = MODEL_EST[modelKey.value] ?? MODEL_EST['0.6b']!
  const durMs = store.project?.duration_ms ?? 0
  const base = Math.max(durMs * est.rtf + est.loadMs, 20_000)
  return Math.max(base, asrCalibMs.value ?? 0)
})


// 阶段内的原始剩余时间。ASR 阶段 = ASR 剩余（可归零）+ 尚未开始的后处理：
// 收尾时剩余正好等于后处理预计，切到后处理是连续倒计时，不会先归零再重新起算。
const etaRawMs = computed<number | null>(() => {
  if (store.run.status !== 'PROCESSING') return null
  const post = postEstimateMs.value
  if (store.stageText === '纠错与数字规范化') {
    return Math.max(0, post - (nowMs.value - phaseStartMs.value))
  }
  if (store.stageText === '保存片段') return 0
  return Math.max(0, asrTotalMs.value - elapsedMs.value) + post
})

// ETA 只减不增：窗口进度事件会把校准总时长抬高（实测 23s 时「还需 10 秒」
// 变成 24s 时「还需 15 秒」）。把每次估算折算成「预计完成时刻」并只取最早值，
// 校准变慢时不会把已承诺的完成时间往后推，倒计时仍随时间连续递减。
// 切换阶段 / 新任务时重置（每个阶段一份独立预算）。
const etaDeadlineMs = ref<number | null>(null)
let etaRunId: string | null = null
watch([etaRawMs, () => store.run.id], ([raw]) => {
  if (store.run.id !== etaRunId) {
    etaRunId = store.run.id
    etaDeadlineMs.value = null
  }
  if (raw === null) {
    etaDeadlineMs.value = null
    return
  }
  const candidate = nowMs.value + raw
  etaDeadlineMs.value = etaDeadlineMs.value === null
    ? candidate
    : Math.min(etaDeadlineMs.value, candidate)
}, { immediate: true })

const etaMs = computed<number | null>(() => (
  etaDeadlineMs.value === null
    ? null
    : Math.max(0, etaDeadlineMs.value - nowMs.value)
))

// 各阶段的原始进度：ASR 0–85%，后处理 85–99%，保存 99%，完成 100%。
// 原来直接显示后端窗口进度：10min 音频只有 3 条事件（33/66/100），
// 而且后处理阶段停在 100% 不动，看起来「80% 直接跳 100%」。
// 这里把「真实窗口进度」与「按时间推进的估算进度」取大并在 ASR 阶段压缩到
// 85%，让进度条在两次窗口事件之间也持续、均匀地走。
const rawPercent = computed(() => {
  if (store.run.status === 'COMPLETED') return 100
  if (store.run.status !== 'PROCESSING') return Math.round(store.run.progress_percent)

  const stage = store.stageText
  if (stage === '保存片段') return 99
  if (stage === '纠错与数字规范化') {
    const done = Math.min(
      1, Math.max(0, (nowMs.value - phaseStartMs.value) / postEstimateMs.value))
    return Math.min(99, 85 + 14 * done)
  }
  const total = asrTotalMs.value + postEstimateMs.value
  const ratio = total > 0 ? elapsedMs.value / total : 0
  // 预计时长内线性推进到 75%，超出后指数逼近 85：既保持「一直在动」，
  // 又不会在严重低估（如 1.7B 显存溢出变慢）时虚报 85% 后僵住
  const byTime = ratio <= 1 ? ratio * 75 : 75 + 10 * (1 - Math.exp(-(ratio - 1)))
  const byReal = (() => {
    const base = store.run.progress_percent * 0.85
    if (asrDoneTs.value === null) return base
    // 推理窗口已完成，用尾部预算持续推进（25s 从 85% 升到 95%）：
    // 覆盖交叉验证比对/对齐等无进度事件的尾部，避免进度条停在 85% 不动。
    const tail = Math.min(1, (nowMs.value - asrDoneTs.value) / ASR_TAIL_BUDGET_MS)
    return base + tail * 10
  })()
  return Math.min(95, Math.max(byTime, byReal))
})

// 进度只前进不后退：窗口事件校准（asrCalibMs 变大 → 总时长变大 → 估算比例
// 变小）、阶段切换、刷新页面都可能让 rawPercent 短暂回落。取历史最大值兜底，
// 新任务（run.id 变化）时清零。
const shownPercent = ref(0)
let shownRunId: string | null = null
watch(
  [rawPercent, () => store.run.id, () => store.run.status],
  () => {
    if (store.run.id !== shownRunId) {
      shownRunId = store.run.id
      shownPercent.value = 0
    }
    shownPercent.value = Math.max(shownPercent.value, rawPercent.value)
  },
  { immediate: true },
)

// 已用时间明显超出预计（>15%）且仍未完成：说明估算偏低或任务异常，
// 此时不能再显示「即将完成」，改为如实提示「仍在处理」
const overrun = computed(() => {
  if (store.run.status !== 'PROCESSING') return false
  // 项目未加载完毕（duration_ms 未知）时不触发超限，
  // 避免 asrTotalMs 跌到底线 10s 导致刚启动就误报「超出预估」。
  const durMs = store.project?.duration_ms
  if (durMs == null || durMs === 0) return false
  const stage = store.stageText
  if (stage === '保存片段') return false
  if (stage === '纠错与数字规范化') {
    return nowMs.value - phaseStartMs.value > postEstimateMs.value * 1.15
  }
  // ASR/预检：估算的完成时刻已过，或耗时明显超出 ASR 预算。
  // 前者保证 ETA 归零的同一刻就切换文案，不会停留在「预计还需约 0 秒 / 即将完成」
  return (etaDeadlineMs.value !== null && nowMs.value > etaDeadlineMs.value + 10_000)
    || elapsedMs.value > asrTotalMs.value * 2.0
})

function fmtSpan(ms: number): string {
  const total = Math.max(0, Math.round(ms / 1000))
  if (total < 60) return `${total} 秒`
  const minutes = Math.floor(total / 60)
  const seconds = total % 60
  if (minutes < 60) return seconds ? `${minutes} 分 ${seconds} 秒` : `${minutes} 分`
  const hours = Math.floor(minutes / 60)
  const rest = minutes % 60
  return rest ? `${hours} 小时 ${rest} 分` : `${hours} 小时`
}

const timingText = computed(() => {
  const eta = etaMs.value
  const done = `已处理 ${fmtSpan(elapsedMs.value)}`
  if (eta === null) return done
  if (store.stageText === '保存片段') return `${done} · 保存片段中`
  // 实际跑超预估：诚实提示，不给「即将完成」的假信号
  if (overrun.value) return `${done} · 仍在处理（超出预估）`
  // 2s 内视为收尾：避免「预计还需约 0 秒」的抖动
  return `${done} · ${eta <= 2000 ? '即将完成' : `预计还需约 ${fmtSpan(eta)}`}`
})
</script>

<template>
  <div class="taskbar" v-if="store.run.status !== 'IDLE' || !store.hasTranscript">
    <el-tag :type="statusMeta.type" size="small">{{ statusMeta.label }}</el-tag>

    <!-- 转写模型选择：仅在可发起任务时显示；未安装的模型置灰 -->
    <el-select
      v-if="models.length && store.run.status !== 'PROCESSING'"
      v-model="selectedModel"
      class="model-picker"
      size="small"
      :disabled="starting"
      @visible-change="(open: boolean) => open && refreshModels()"
    >
      <el-option
        v-for="m in models"
        :key="m.key"
        :value="m.key"
        :label="m.available ? m.label : `${m.label}（未安装）`"
        :disabled="!m.available"
      />
    </el-select>
    <el-checkbox
      v-if="models.length && store.run.status !== 'PROCESSING'"
      v-model="fastMode"
      class="fast-mode"
      size="small"
      :disabled="starting"
      title="跳过字级时间戳对齐"
    >快速</el-checkbox>
    <el-checkbox
      v-if="models.length && store.run.status !== 'PROCESSING'"
      v-model="crossCheckMode"
      class="cross-check"
      size="small"
      :disabled="starting"
      title="第二模型核验低置信词组"
    >交叉验证</el-checkbox>

    <template v-if="store.run.status === 'PROCESSING'">
      <el-progress
        class="progress"
        :percentage="Math.round(shownPercent)"
        :stroke-width="8"
      />
      <!-- 阶段文本尚未到达时（刷新后快照未到）不显示，避免与左侧「处理中」标签重复 -->
      <span class="stage" v-if="store.stageText">{{ store.stageText }}</span>
      <span class="timing">{{ timingText }}</span>
      <span class="model timecode" v-if="store.run.model_name">{{ store.run.model_name }}</span>
      <el-button size="small" type="warning" plain @click="cancel">取消任务</el-button>
    </template>

    <template v-else-if="store.run.status === 'FAILED'">
      <span class="error">{{ store.run.error_message ?? '任务失败' }}</span>
      <el-button size="small" type="primary" :loading="starting" @click="start">重新转写</el-button>
    </template>

    <template v-else-if="store.run.status === 'CANCELLED'">
      <el-button size="small" type="primary" :loading="starting" @click="start">重新转写</el-button>
    </template>

    <template v-else-if="store.run.status === 'COMPLETED'">
      <span class="stage">转写完成，可编辑与导出</span>
      <el-button size="small" type="primary" plain :loading="starting" @click="start">
        重新转写
      </el-button>
    </template>

    <template v-else>
      <!-- 空闲且无转写 -->
      <el-button size="small" type="primary" :loading="starting" @click="start">
        开始转写
      </el-button>
    </template>
  </div>
</template>

<style scoped>
.taskbar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  row-gap: 6px;
  gap: 12px;
  padding: 8px 16px;
  border-bottom: 1px solid var(--at-border);
  background: #fafbfc;
}

.progress {
  flex: 1;
  max-width: 420px;
}

/* 进度条宽度过渡：配合 500ms 定时推进，肉眼看不到台阶 */
.progress :deep(.el-progress-bar__inner) {
  transition: width 0.5s linear;
}

.stage {
  color: var(--at-text);
}

/* 已处理 / 预计剩余：弱化显示，避免与阶段文本抢视线 */
.timing {
  color: var(--at-muted);
  font-size: 12px;
  white-space: nowrap;
}

.model {
  color: var(--at-muted);
}

.model-picker {
  width: 170px;
}

.fast-mode,
.cross-check {
  margin-right: -6px;
  font-size: 12px;
}

.error {
  color: var(--el-color-danger);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 60%;
}

/* 平板：进度条占满可用宽度，模型下拉略收窄 */
@media (max-width: 1023px) {
  .taskbar {
    padding: 8px 12px;
    gap: 10px;
  }

  .progress {
    max-width: none;
  }

  .model-picker {
    width: 150px;
  }
}

/* 手机：进度条独占一行（与计时文本分开），隐藏模型名避免挤爆 */
@media (max-width: 640px) {
  .taskbar {
    padding: 8px 10px;
    gap: 8px;
  }

  .progress {
    order: 20;
    flex-basis: 100%;
    max-width: none;
  }

  .timing {
    font-size: 11px;
  }

  .model {
    display: none;
  }

  .model-picker {
    width: 150px;
  }

  .error {
    max-width: 100%;
    white-space: normal;
  }
}
</style>
