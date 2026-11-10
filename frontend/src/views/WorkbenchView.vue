<script setup lang="ts">
// 工作台（§12.1 页面区域）：顶部导航 / 播放器 / 操作栏 / 任务视图 / 片段列表
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'

import AudioPlayer from '@/components/AudioPlayer.vue'
import FindReplace from '@/components/FindReplace.vue'
import SegmentList from '@/components/SegmentList.vue'
import TaskBar from '@/components/TaskBar.vue'
import { api } from '@/api/client'
import { useWorkbenchStore } from '@/stores/workbench'

const route = useRoute()
const router = useRouter()
const store = useWorkbenchStore()

const projectId = computed(() => route.params.projectId as string)
const audioRef = ref<HTMLAudioElement | null>(null)
const findReplaceVisible = ref(false)

const fullText = computed(() => store.segments.map((s) => s.finalText).join(''))

// 与项目列表页一致的"X 分 X 秒"格式（59 分钟音频显示 3540s 不可读）
function fmtDuration(ms: number): string {
  const s = Math.round(ms / 1000)
  const m = Math.floor(s / 60)
  return m > 0 ? `${m} 分 ${s % 60} 秒` : `${s} 秒`
}

onMounted(async () => {
  try {
    await store.load(projectId.value)
  } catch (err) {
    // 项目不存在/已删除：给出反馈并回列表，而不是永远停在"加载中…"
    ElMessage.error(`加载项目失败：${(err as Error).message}`)
    await router.replace('/')
  }
})

onBeforeUnmount(() => {
  store.teardown()
})

// audio 元素事件 -> store（§12.2 联动的驱动源）
function onTimeUpdate() {
  const a = audioRef.value
  if (a) store.onTimeUpdate(a.currentTime * 1000)
}

function onDurationChange() {
  const a = audioRef.value
  if (a && Number.isFinite(a.duration)) store.durationMs = a.duration * 1000
}

function onPlayPause(p: boolean) {
  store.playing = p
}

// 点击片段正文：定位但保持暂停（§12.2）
watch(() => store.currentTimeMs, () => { /* currentTimeMs 由 store 驱动，seek 在 SegmentList 调 store */ })

function seekTo(ms: number) {
  const a = audioRef.value
  if (!a) return
  a.currentTime = ms / 1000
  store.seekToMs(ms)
}

async function copyAll() {
  if (!fullText.value) return
  await navigator.clipboard.writeText(fullText.value)
  ElMessage.success(`已复制 ${fullText.value.length} 字`)
}

// 导出是 POST 接口（写方法需会话 Cookie）：window.open 的 GET 会被 SPA
// catch-all 拦截返回 index.html，必须 fetch + blob 触发浏览器下载
async function exportTxt() {
  try {
    const resp = await fetch(api.exportUrl(projectId.value), { method: 'POST' })
    if (!resp.ok) {
      let message = `HTTP ${resp.status}`
      try {
        const data = await resp.json()
        if (data?.detail) message = data.detail
      } catch { /* 非 JSON 错误体 */ }
      throw new Error(message)
    }
    const blob = await resp.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${store.project?.original_name ?? 'transcript'}.txt`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
    ElMessage.success('已导出 TXT')
  } catch (err) {
    ElMessage.error(`导出失败：${(err as Error).message}`)
  }
}

function rateChanged(rate: number) {
  store.rate = rate
}
</script>

<template>
  <div class="workbench">
    <!-- 顶部导航（§12.1） -->
    <header class="topbar">
      <el-button text @click="router.push('/')">←<span class="back-label"> 项目列表</span></el-button>
      <span class="title">{{ store.project?.original_name ?? '加载中…' }}</span>
      <span class="duration timecode" v-if="store.project">
        {{ fmtDuration(store.project.duration_ms) }}
      </span>
    </header>

    <!-- 音频播放器（固定顶部，§12.1） -->
    <AudioPlayer
      :audio-ref="audioRef"
      :current-time-ms="store.currentTimeMs"
      :duration-ms="store.durationMs || store.project?.duration_ms || 0"
      :playing="store.playing"
      :rate="store.rate"
      @seek="seekTo"
      @play-pause="onPlayPause"
      @rate-change="rateChanged"
    />

    <!-- 操作栏（§12.1） -->
    <div class="toolbar">
      <el-button size="small" @click="copyAll" :disabled="!store.hasTranscript">复制全文</el-button>
      <el-button size="small" @click="findReplaceVisible = true" :disabled="!store.hasTranscript">
        查找替换
      </el-button>
      <el-button size="small" @click="exportTxt" :disabled="!store.hasTranscript">导出 TXT</el-button>
    </div>

    <!-- 任务视图（§12.1 任务状态切换） -->
    <TaskBar />

    <!-- 文本列表（§12.3 虚拟滚动） -->
    <div class="list-wrap">
      <audio
        ref="audioRef"
        :src="store.project ? api.audioUrl(store.project.id) : undefined"
        preload="auto"
        @timeupdate="onTimeUpdate"
        @durationchange="onDurationChange"
        @play="onPlayPause(true)"
        @pause="onPlayPause(false)"
        @ended="onPlayPause(false)"
        @ratechange="audioRef && (store.rate = audioRef.playbackRate)"
      />
      <SegmentList :seek="seekTo" />
    </div>

    <FindReplace v-model="findReplaceVisible" />
  </div>
</template>

<style scoped>
.workbench {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: #fff;
}

.topbar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 16px;
  border-bottom: 1px solid var(--at-border);
}

/* 文件名占满剩余宽度并省略，避免小屏把时长挤到第二行 */
.title {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 600;
}

.duration {
  flex-shrink: 0;
}

.toolbar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  padding: 6px 16px;
  border-bottom: 1px solid var(--at-border);
  background: #fafbfc;
}

.list-wrap {
  flex: 1;
  min-height: 0;
  position: relative;
}

/* 平板：收紧页边距 */
@media (max-width: 1023px) {
  .topbar {
    padding: 8px 12px;
  }

  .toolbar {
    padding: 6px 12px;
  }
}

/* 手机：返回按钮只留箭头，标题与时长压成一行 */
@media (max-width: 640px) {
  .topbar {
    gap: 8px;
    padding: 6px 10px;
  }

  .back-label {
    display: none;
  }

  .title {
    font-size: 14px;
  }

  .duration {
    font-size: 11px;
  }

  .toolbar {
    padding: 6px 10px;
    gap: 6px;
  }
}
</style>
