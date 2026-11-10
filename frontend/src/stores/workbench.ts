// 工作台 store：当前项目 + 任务生命周期 + 片段数据 + 音频联动（§12.2）
import { defineStore } from 'pinia'
import { computed, ref, shallowRef } from 'vue'

import { api, subscribeEvents } from '@/api/client'
import type { ProjectDetail, RunInfo, Segment } from '@/types'

export type SaveState = 'idle' | 'saving' | 'saved' | 'failed'

export const useWorkbenchStore = defineStore('workbench', () => {
  // ---- 项目与片段 ----
  const project = ref<ProjectDetail | null>(null)
  const segments = ref<Segment[]>([])
  const hasTranscript = ref(false)

  // ---- 任务 ----
  const run = ref<RunInfo>({ id: null, projectId: '', status: 'IDLE', stage: null,
    progress_percent: 0, model_name: null, error_code: null, error_message: null,
    started_at: null })
  const stageText = ref('')        // 人类可读的当前阶段（SSE phase_updated）
  const running = computed(() => run.value.status === 'PROCESSING')

  // ---- 音频状态（联动由 <audio> 事件驱动）----
  const currentTimeMs = ref(0)
  const durationMs = ref(0)
  const playing = ref(false)
  const rate = ref(1)
  // 活动片段：只在变化时更新（§12.2 高亮与滚动不抖动）
  const activeSegmentId = ref<string | null>(null)
  // 自动滚动保护：用户手动滚动后短暂禁用跟随（§12.3）
  const followScroll = ref(true)

  // ---- SSE ----
  let unsubscribe: (() => void) | null = null
  const stageMap: Record<string, string> = {
    preflight: '检查运行环境', transcribing: '语音识别中',
    postprocessing: '纠错与数字规范化', validating: '预检',
    saving: '保存片段', completed: '完成',
  }

  async function load(projectId: string) {
    teardown()
    project.value = await api.getProject(projectId)
    await api.currentTask().then((cur: RunInfo) => {
      if (cur.status === 'PROCESSING') {
        run.value = cur
        // 刷新/重进页面时重建 SSE 订阅，否则进度条与阶段文本冻结，
        // 任务完成也不会自动刷新片段（连接即收 task_snapshot 恢复现场）
        if (cur.id) unsubscribe = subscribeEvents(cur.id, onEvent)
      } else {
        run.value = { ...run.value, status: 'IDLE' }
      }
    }).catch(() => undefined)
    segments.value = []
    hasTranscript.value = false
    await refreshTranscript()
  }

  function teardown() {
    unsubscribe?.()
    unsubscribe = null
    playing.value = false
    currentTimeMs.value = 0
    activeSegmentId.value = null
  }

  async function refreshTranscript() {
    if (!project.value) return
    try {
      const { runId, segments: segs } = await api.getTranscript(project.value.id)
      segments.value = segs
      hasTranscript.value = true
      // 有完成的转写：任务栏显示完成态（含重新转写入口）
      if (run.value.status === 'IDLE' || run.value.status === 'COMPLETED') {
        run.value = { ...run.value, id: runId, status: 'COMPLETED', progress_percent: 100 }
        stageText.value = '完成'
      }
    } catch {
      segments.value = []
      hasTranscript.value = false
    }
  }

  // ------------------------------------------------------------------
  // 任务生命周期
  // ------------------------------------------------------------------
  // model：0.6b（默认，快速）/ 1.7b（高质量），由任务栏下拉框传入；
  // fast：快速模式，跳过字符级对齐（省 2–6s，句级时间戳按字数插值）
  // crossCheck：第二模型交叉验证（默认关闭，勾选后固定增加约 30–48s）
  async function startTranscription(model = '0.6b', fast = false, crossCheck = false) {
    if (!project.value || running.value) return
    const { runId } = await api.startRun(project.value.id, model, fast, crossCheck)
    // started_at 记为本地时刻：任务栏据此显示「已处理多久」（SSE 快照到达后
    // 会以 DB 时间为准，刷新页面也能恢复）
    run.value = { ...run.value, id: runId, status: 'PROCESSING', stage: 'VALIDATING',
      progress_percent: 0, error_code: null, error_message: null,
      started_at: new Date().toISOString() }
    stageText.value = '预检'
    unsubscribe = subscribeEvents(runId, onEvent)
  }

  async function cancelTranscription() {
    if (!project.value || !run.value.id || !running.value) return
    await api.cancelRun(project.value.id, run.value.id)
  }

  function onEvent(event: { kind: string } & Record<string, unknown>) {
    const kind: string = event.kind
    // 连接即达的快照：刷新后恢复进度百分比与阶段文本（stage 为大写枚举）
    if (kind === 'task_snapshot') {
      if (typeof event.progress_percent === 'number') {
        run.value.progress_percent = event.progress_percent
      }
      if (typeof event.stage === 'string' && event.stage) {
        stageText.value = stageMap[event.stage.toLowerCase()] ?? event.stage
      }
      return
    }
    if (kind === 'phase_updated' && typeof event.phase === 'string') {
      stageText.value = stageMap[event.phase] ?? event.phase
    }
    if (kind === 'progress_updated' && typeof event.percent === 'number') {
      run.value.progress_percent = event.percent
    }
    if (kind === 'task_completed') {
      run.value.status = 'COMPLETED'
      run.value.progress_percent = 100
      stageText.value = '完成'
      unsubscribe?.()
      unsubscribe = null
      void refreshTranscript()
    }
    if (kind === 'task_failed') {
      run.value.status = 'FAILED'
      run.value.error_message = String(event.message ?? '任务失败')
      stageText.value = '失败'
      unsubscribe?.()
      unsubscribe = null
    }
    if (kind === 'task_cancelled') {
      run.value.status = 'CANCELLED'
      stageText.value = '已取消'
      unsubscribe?.()
      unsubscribe = null
    }
  }

  // ------------------------------------------------------------------
  // 音频联动（§12.2：二分查找当前片段，只有活动片段变化才更新）
  // ------------------------------------------------------------------
  function onTimeUpdate(ms: number) {
    currentTimeMs.value = ms
    const segs = segments.value
    if (!segs.length) return
    // 二分：最后一个 start_ms <= ms 的片段
    let lo = 0, hi = segs.length - 1, ans = -1
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      if (segs[mid]!.startMs <= ms) { ans = mid; lo = mid + 1 } else { hi = mid - 1 }
    }
    if (ans >= 0 && ms <= segs[ans]!.endMs + 800) {
      // endMs 后 800ms 容差：短暂静音区间保持上一段选中
      const id = segs[ans]!.id
      if (activeSegmentId.value !== id) activeSegmentId.value = id
    } else if (ans >= 0) {
      // 长静音：保持上一段（§12.2 允许不高亮）
    } else {
      activeSegmentId.value = null
    }
  }

  function seekToMs(ms: number) {
    currentTimeMs.value = ms
    onTimeUpdate(ms)
  }

  // ------------------------------------------------------------------
  // 片段编辑（§12.5：防抖自动保存）
  // ------------------------------------------------------------------
  const saveStates = ref<Record<string, SaveState>>({})
  const timers: Record<string, ReturnType<typeof setTimeout>> = {}

  function saveState(id: string): SaveState {
    return saveStates.value[id] ?? 'idle'
  }

  async function saveSegment(segment: Segment, editedText: string) {
    saveStates.value[segment.id] = 'saving'
    try {
      await api.saveSegment(segment.id, { editedText })
      saveStates.value[segment.id] = 'saved'
      segment.editedText = editedText
      segment.finalText = editedText
    } catch {
      saveStates.value[segment.id] = 'failed'
    }
  }

  /** 输入停止约 600ms 后保存（§12.5 可配置范围 500~800ms）。 */
  function scheduleSave(segment: Segment, editedText: string) {
    const id = segment.id
    clearTimeout(timers[id])
    saveStates.value[id] = 'idle'
    timers[id] = setTimeout(() => { void saveSegment(segment, editedText) }, 600)
  }

  /** 切换片段 / Ctrl+S 主动保存（取消挂起的防抖立即写）。 */
  function flushSave(segment: Segment, editedText: string) {
    const id = segment.id
    if (timers[id]) {
      clearTimeout(timers[id])
      delete timers[id]
      void saveSegment(segment, editedText)
    }
  }

  // ------------------------------------------------------------------
  // 查找替换 / 撤销（§12.5）
  // ------------------------------------------------------------------
  async function previewReplace(find: string, replace: string) {
    if (!project.value) throw new Error('未加载项目')
    return api.previewReplace(project.value.id, find, replace)
  }

  async function applyReplace(find: string, replace: string) {
    if (!project.value) throw new Error('未加载项目')
    const result = await api.applyReplace(project.value.id, find, replace)
    if (result.count > 0) await refreshTranscript()
    return result
  }

  async function undoReplace(operationId: string) {
    const result = await api.undoOperation(operationId)
    await refreshTranscript()
    return result
  }

  // 复核过滤 UI 已按用户要求移除：可见列表即全部段
  const visibleSegments = computed(() => segments.value)

  return {
    project, segments, hasTranscript, visibleSegments,
    run, running, stageText,
    currentTimeMs, durationMs, playing, rate, activeSegmentId, followScroll,
    saveStates, saveState, scheduleSave, flushSave, saveSegment,
    load, startTranscription, cancelTranscription, refreshTranscript,
    onTimeUpdate, seekToMs,
    previewReplace, applyReplace, undoReplace,
    teardown,
  }
})
