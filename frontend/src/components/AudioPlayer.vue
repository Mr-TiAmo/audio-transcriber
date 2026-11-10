<script setup lang="ts">
// 音频播放器（§12.1）：播放/暂停、时间、seek、音量、倍速
// audio 元素由父组件持有（WorkbenchView），本组件只渲染控件并命令式操作
import { computed, ref, watch } from 'vue'
import { VideoPlay, VideoPause, Mute, Microphone } from '@element-plus/icons-vue'

const props = defineProps<{
  audioRef: HTMLAudioElement | null
  currentTimeMs: number
  durationMs: number
  playing: boolean
  rate: number
}>()

const emit = defineEmits<{
  seek: [ms: number]
  'play-pause': [playing: boolean]
  'rate-change': [rate: number]
}>()

const volume = ref(1)
const muted = ref(false)
const rates = [0.5, 0.75, 1, 1.25, 1.5, 2]

function fmt(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000))
  const m = Math.floor(s / 60)
  return `${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

const position = computed(() => (props.durationMs > 0 ? props.currentTimeMs : 0))

function toggle() {
  const a = props.audioRef
  if (!a) return
  if (a.paused) {
    a.playbackRate = props.rate
    void a.play().then(() => emit('play-pause', true)).catch(() => emit('play-pause', false))
  } else {
    a.pause()
  }
}

function onSeek(ms: number) {
  emit('seek', ms)
}

function onRate(rate: number) {
  if (props.audioRef) props.audioRef.playbackRate = rate
  emit('rate-change', rate)
}

watch(volume, (v) => {
  if (props.audioRef) props.audioRef.volume = v
})

watch(muted, (m) => {
  if (props.audioRef) props.audioRef.muted = m
})
</script>

<template>
  <div class="player">
    <el-button circle :icon="playing ? VideoPause : VideoPlay" @click="toggle" />
    <span class="timecode time">{{ fmt(position) }} / {{ fmt(durationMs) }}</span>
    <el-slider
      class="progress"
      :model-value="position"
      :max="Math.max(durationMs, 1)"
      :step="100"
      :show-tooltip="false"
      @input="(v: number | number[]) => onSeek(Array.isArray(v) ? (v[0] ?? 0) : v)"
    />
    <el-button circle :icon="muted ? Mute : Microphone" text @click="muted = !muted" />
    <el-slider v-model="volume" class="volume" :max="1" :step="0.05" :show-tooltip="false" />
    <el-select
      class="rate"
      :model-value="rate"
      size="small"
      @update:model-value="(v: number) => onRate(v)"
    >
      <el-option v-for="r in rates" :key="r" :value="r" :label="`${r}x`" />
    </el-select>
  </div>
</template>

<style scoped>
.player {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 16px;
  border-bottom: 1px solid var(--at-border);
}

.time {
  min-width: 110px;
  text-align: center;
}

.progress {
  flex: 1;
}

.volume {
  width: 80px;
}

.rate {
  width: 80px;
}

/* 平板：收窄控件，进度条仍与播放键同一行 */
@media (max-width: 1023px) {
  .player {
    padding: 8px 12px;
    gap: 8px;
  }

  .volume {
    width: 60px;
  }
}

/* 手机：进度条独占第二行，否则会被其它控件挤到 0 宽度；
   音量条在触屏上难精确拖动，隐藏（静音按钮保留） */
@media (max-width: 640px) {
  .player {
    flex-wrap: wrap;
    row-gap: 2px;
    padding: 6px 10px;
    gap: 8px;
  }

  .time {
    min-width: 92px;
    font-size: 12px;
  }

  .progress {
    order: 10;
    flex-basis: 100%;
  }

  .volume {
    display: none;
  }

  .rate {
    width: 66px;
    margin-left: auto;
  }
}
</style>
