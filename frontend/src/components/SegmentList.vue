<script setup lang="ts">
// 整句行列表（用户确认的交互）：
// - 显示按句：段拆成整句，一行一句，行首为该句估算的音频起止时间（mm:ss）
// - 直接编辑：句行正文 contenteditable，单击即跳转并可直接改字（无需编辑模式），
//   输入停止 ~600ms 自动保存；写回粒度仍是段（替换段全文中对应句后整段保存，
//   与查找替换/撤销兼容）
// - 点击跳转：单击句行（正文或时间码）→ 播放器跳转到该句起点并保持暂停（§12.2）
// - 复核/编辑按钮 UI 已按用户要求移除（数据层 review_status 字段保留不动）
// - 句级时间优先取后端 ForcedAligner 字符级时间戳；缺失或文本被编辑后
//   回退为按字数比例插值（行首时间码弱化显示，区分"真实"与"估算"）
import { computed, nextTick, ref, watch, type ComponentPublicInstance } from 'vue'
import { useVirtualizer } from '@tanstack/vue-virtual'

import { useWorkbenchStore } from '@/stores/workbench'
import { fmtClock, splitSentences } from '@/utils/sentences'
import type { Segment } from '@/types'

const props = defineProps<{ seek: (ms: number) => void }>()

const store = useWorkbenchStore()
const parentRef = ref<HTMLElement | null>(null)

// ------------------------------------------------------------------
// 扁平行模型：段头行 / 整句行
// ------------------------------------------------------------------
type Row =
  | { type: 'head'; seg: Segment }
  | { type: 'sentence'; seg: Segment; text: string; startMs: number; endMs: number; estimated: boolean; index: number; key: string; lowConfSpans: Array<{ start: number; end: number }> }

const rows = computed<Row[]>(() => {
  const out: Row[] = []
  for (const seg of store.visibleSegments) {
    out.push({ type: 'head', seg })
    // 对齐时间仅在文本未被编辑时有效（长度校验在 splitSentences 内）
    const slices = splitSentences(seg.finalText, seg.startMs, seg.endMs, seg.alignedTokens)
    const perSentenceSpans = sentenceLowConfSpans(seg, slices)
    if (!slices.length) {
      // 空文本段：仍给一个可点击定位/可输入的空句行
      out.push({ type: 'sentence', seg, text: '', startMs: seg.startMs, endMs: seg.endMs,
        estimated: true, index: 0, key: `${seg.id}:0`, lowConfSpans: [] })
      continue
    }
    slices.forEach((s, i) => {
      out.push({ type: 'sentence', seg, text: s.text, startMs: s.startMs, endMs: s.endMs,
        estimated: s.estimated, index: i, key: `${seg.id}:${i}`, lowConfSpans: perSentenceSpans[i] ?? [] })
    })
  }
  return out
})

/** 行 key（虚拟滚动 v-for 用）：句行用稳定 key，段头行用类型+段 id。 */
function rowKey(i: number): string {
  const r = rows.value[i]!
  return r.type === 'sentence' ? r.key : `${r.type}:${r.seg.id}`
}

/** 取句行（模板内避免 TS 断言的类型收窄辅助）。 */
function sentRow(i: number) {
  const r = rows.value[i]
  return r?.type === 'sentence' ? r : null
}

// 当前播放位置对应的行索引（句级高亮 + 跟随滚动）
const activeRowIndex = computed(() => {
  const ms = store.currentTimeMs
  const list = rows.value
  for (let i = 0; i < list.length; i++) {
    const r = list[i]!
    if (r.type === 'sentence' && ms >= r.startMs && ms < r.endMs + 800) return i
  }
  // 未命中句行（静音区/尚未播放）：退回活动段头行
  const id = store.activeSegmentId
  if (id) {
    const h = list.findIndex((r) => r.type === 'head' && r.seg.id === id)
    if (h >= 0) return h
  }
  return -1
})

// 虚拟滚动（§12.3）：行高估算，实际高度动态测量
const virtualizer = useVirtualizer(
  computed(() => ({
    count: rows.value.length,
    getScrollElement: () => parentRef.value,
    estimateSize: () => 34,
    overscan: 10,
  })),
)

// 手动滚动保护（§12.3）：用户滚动后 5s 内播放不抢滚动
let followTimer: ReturnType<typeof setTimeout> | null = null
function onUserScrollIntent() {
  store.followScroll = false
  if (followTimer) clearTimeout(followTimer)
  followTimer = setTimeout(() => { store.followScroll = true }, 5000)
}

// 活动行变化 -> 跟随滚动（句级；只有播放中且未被用户手动滚动打断时）
watch(activeRowIndex, async (idx) => {
  if (idx < 0 || !store.followScroll || !store.playing) return
  await nextTick()
  virtualizer.value.scrollToIndex(idx, { align: 'auto' })
})

// ------------------------------------------------------------------
// 句行交互：单击跳转 + 直接编辑自动保存
// ------------------------------------------------------------------
/** 单击句行（正文或时间码均冒泡到此）：定位到该句起点，保持暂停（用户确认）。 */
function locateSentence(row: Row) {
  if (row.type !== 'sentence') return
  props.seek(row.startMs)
}

/** 点击段头行：定位到段起点。 */
function locateSegment(seg: Segment) {
  props.seek(seg.startMs)
}

/** 句行正文输入：替换所属段全文中对应句 → 整段防抖自动保存。 */
function onSentenceInput(row: Row, ev: Event) {
  if (row.type !== 'sentence') return
  const newText = (ev.target as HTMLElement).textContent ?? ''
  const slices = splitSentences(row.seg.finalText, row.seg.startMs, row.seg.endMs)
  const target = slices[row.index]
  if (!target) return
  slices[row.index] = { ...target, text: newText }
  store.scheduleSave(row.seg, slices.map((s) => s.text).join(''))
}

/** 句行内回车 = 结束本句编辑（不插入换行；保存由 input 防抖负责）。 */
function onSentenceKeydown(ev: KeyboardEvent) {
  if (ev.key === 'Enter') {
    ev.preventDefault()
    ;(ev.target as HTMLElement).blur()
  }
}

// ------------------------------------------------------------------
/**
 * 将段级 lowConfSpans 映射到各句行的局部坐标。
 * 返回每句的 [[start, end], ...]（与 input 的 sentences 同序）。
 */
function sentenceLowConfSpans(
  seg: Segment,
  sentences: Array<{ text: string; startMs: number; endMs: number }>,
): Array<Array<{ start: number; end: number }>> {
  const spans = seg.lowConfSpans
  if (!spans || !spans.length || !sentences.length) return sentences.map(() => [])
  // 后端 low_conf_spans 是 Python 码点索引，前端按码点数组对齐（代理对安全）
  const cps = Array.from(seg.finalText)
  const sentenceBounds: Array<{ start: number; end: number }> = []
  let searchFrom = 0
  for (const s of sentences) {
    const scps = Array.from(s.text)
    const pos = indexOfCps(cps, scps, searchFrom)
    if (pos >= 0) {
      sentenceBounds.push({ start: pos, end: pos + scps.length })
      searchFrom = pos + scps.length
    } else {
      sentenceBounds.push({ start: 0, end: 0 })
    }
  }
  return sentences.map((_, i) => {
    const sb = sentenceBounds[i]
    if (!sb || sb.end <= sb.start) return []
    const local: Array<{ start: number; end: number }> = []
    for (const span of spans) {
      const s = span[0], e = span[1]
      if (e <= sb.start || s >= sb.end) continue
      local.push({
        start: Math.max(0, s - sb.start),
        end: Math.min(sb.end - sb.start, e - sb.start),
      })
    }
    return local
  })
}

/** 在码点数组中按顺序查找子序列；允许跳过原文中的空白（trim/拼接差异）。 */
function indexOfCps(hay: string[], needle: string[], from: number): number {
  if (!needle.length) return -1
  for (let h = from; h + needle.length <= hay.length; h++) {
    if (hay[h] !== needle[0]) continue
    let hi = h, ni = 0
    while (ni < needle.length) {
      if (hi >= hay.length) break
      if (hay[hi] === needle[ni]) { hi++; ni++ }
      else if (/\s/.test(hay[hi]!)) hi++      // 跳过原文多余空白
      else break
    }
    if (ni === needle.length) return h
  }
  return -1
}

/** 将文本中低置信区间用 &lt;mark&gt; 包裹（区间为码点索引） */
function highlightText(text: string, spans: Array<{ start: number; end: number }>): string {
  const cps = Array.from(text)
  if (!cps.length || !spans.length) return escapeHtml(text)
  const sorted = [...spans].sort((a, b) => a.start - b.start)
  const merged: Array<{ start: number; end: number }> = []
  for (const s of sorted) {
    const last = merged[merged.length - 1]
    if (last && s.start <= last.end) {
      last.end = Math.max(last.end, s.end)
    } else {
      merged.push({ ...s })
    }
  }
  const slice = (a: number, b: number) => cps.slice(a, b).join('')
  let out = ''
  let prev = 0
  for (const s of merged) {
    const start = Math.max(0, Math.min(s.start, cps.length))
    const end = Math.max(start, Math.min(s.end, cps.length))
    if (start > prev) out += escapeHtml(slice(prev, start))
    out += '<mark class="lo">' + escapeHtml(slice(start, end)) + '</mark>'
    prev = end
  }
  if (prev < cps.length) out += escapeHtml(slice(prev, cps.length))
  return out
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

// 句行正文 = 非受控编辑区：渲染不写文本节点（保住光标），显式同步外部更新
// （查找替换/重转写/保存回写后 finalText 变化时刷新未聚焦的句行）
// ------------------------------------------------------------------
const textEls = new Map<number, HTMLElement>()

function syncOne(index: number, el: HTMLElement) {
  const r = rows.value[index]
  if (!r || r.type !== 'sentence') return
  if (document.activeElement === el) return  // 编辑中不碰，保光标
  const hasSpans = r.lowConfSpans && r.lowConfSpans.length > 0
  if (hasSpans) {
    const html = highlightText(r.text, r.lowConfSpans)
    if (el.innerHTML !== html) el.innerHTML = html
  } else {
    if (el.textContent !== r.text) el.textContent = r.text
  }
}

// 句行获得焦点（单击即可编辑）时**保留** &lt;mark&gt; 标亮：用户反馈编辑态标亮消失
// 不利于对照修改。contenteditable="plaintext-only" 下 mark 是普通内联元素，
// 光标移动与输入不受影响；若在 mark 内输入，新字符会临时继承标亮，失焦保存后
// rows 更新即按新的 finalText 重建，不影响数据正确性（保存读 textContent）。

/** 句行失焦时：未编辑则重新应用标亮；正在编辑的内容不覆盖，等保存回写后由 rows 同步 */
function onTextBlur(index: number, ev: Event) {
  const el = ev.target
  if (!(el instanceof HTMLElement)) return
  const r = rows.value[index]
  if (!r || r.type !== 'sentence') return
  if ((el.textContent ?? '') !== r.text) return  // 用户刚改过，等防抖保存回写
  syncOne(index, el)
}

/** span ref 回调：登记元素 + 首次挂载/虚拟滚动回收重建时写入文本。 */
function setTextRef(index: number) {
  // 参数类型需兼容 VNodeRef（Element | ComponentPublicInstance | null）
  return (el: Element | ComponentPublicInstance | null) => {
    const htmlEl = el instanceof HTMLElement ? el : null
    if (!htmlEl) {
      textEls.delete(index)
      return
    }
    textEls.set(index, htmlEl)
    syncOne(index, htmlEl)
  }
}

// rows 重算（外部更新）后同步所有未聚焦句行
watch(rows, async () => {
  await nextTick()
  for (const [i, el] of textEls) syncOne(i, el)
}, { flush: 'post' })
</script>

<template>
  <div
    ref="parentRef"
    class="segment-scroll"
    @wheel="onUserScrollIntent"
    @touchstart="onUserScrollIntent"
  >
    <el-empty v-if="!rows.length" :description="store.hasTranscript ? '没有符合条件的片段' : '暂无转写内容'" />
    <div
      v-else
      :style="{ height: `${virtualizer.getTotalSize()}px`, position: 'relative' }"
    >
      <div
        v-for="vRow in virtualizer.getVirtualItems()"
        :key="rowKey(vRow.index)"
        :data-index="vRow.index"
        :ref="(el) => virtualizer.measureElement(el as Element | null)"
        :style="{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          transform: `translateY(${vRow.start}px)`,
        }"
        class="row-anchor"
      >
        <!-- 段头行：段时间范围 + 风险标记 + 保存状态 -->
        <div
          v-if="rows[vRow.index]!.type === 'head'"
          class="segment segment-head"
          :class="{ active: rows[vRow.index]!.seg.id === store.activeSegmentId }"
          @click="locateSegment(rows[vRow.index]!.seg)"
        >
          <span class="timecode">
            {{ fmtClock(rows[vRow.index]!.seg.startMs) }} → {{ fmtClock(rows[vRow.index]!.seg.endMs) }}
          </span>
          <el-tag
            v-if="rows[vRow.index]!.seg.riskFlags.length"
            size="small" type="warning"
          >风险</el-tag>
          <span class="save-state">
            <template v-if="store.saveState(rows[vRow.index]!.seg.id) === 'saving'">保存中…</template>
            <template v-else-if="store.saveState(rows[vRow.index]!.seg.id) === 'saved'">已保存</template>
            <template v-else-if="store.saveState(rows[vRow.index]!.seg.id) === 'failed'">保存失败</template>
          </span>
        </div>

        <!-- 整句行：单击跳转；正文直接可编辑（contenteditable），自动保存 -->
        <div
          v-else
          class="sentence"
          :class="{ current: vRow.index === activeRowIndex }"
          @click="locateSentence(rows[vRow.index]!)"
        >
          <span
            class="timecode sentence-time"
            :class="{ estimated: sentRow(vRow.index)!.estimated }"
            :title="sentRow(vRow.index)!.estimated ? '按字数估算的时间' : '字符级对齐时间'"
          >
            {{ fmtClock(sentRow(vRow.index)!.startMs) }} → {{ fmtClock(sentRow(vRow.index)!.endMs) }}
          </span>
          <!-- 非受控编辑区：文本由 setTextRef/syncOne 显式写入，
               渲染不碰文本节点，编辑时光标与输入内容不被重渲染打断 -->
          <span
            class="text"
            contenteditable="plaintext-only"
            :ref="setTextRef(vRow.index)"
            @input="onSentenceInput(sentRow(vRow.index)!, $event)"
            @keydown="onSentenceKeydown"
            @blur="onTextBlur(vRow.index, $event)"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.segment-scroll {
  height: 100%;
  overflow-y: auto;
  outline: none;
}

.row-anchor {
  padding: 0 16px;
}

/* 段头行 */
.segment-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 12px;
  margin-top: 6px;
  border-radius: 6px 6px 0 0;
  background: #f0f2f5;
  border: 1px solid transparent;
  cursor: pointer;
}

.segment-head.active {
  background: #ecf5ff;
  border-color: var(--at-primary);
}

/* 整句行 */
.sentence {
  display: flex;
  align-items: baseline;
  gap: 12px;
  padding: 3px 12px 3px 20px;
  border-radius: 4px;
  cursor: pointer;
  line-height: 1.6;
}

.sentence:hover {
  background: #f5f7fa;
}

.sentence.current {
  background: #ecf5ff;
}

.sentence-time {
  flex-shrink: 0;
  min-width: 108px;
}

/* 估算时间弱化显示，与真实对齐时间区分 */
.sentence-time.estimated {
  color: var(--at-muted);
  opacity: 0.7;
}

.sentence .text {
  word-break: break-all;
  white-space: pre-wrap;
  cursor: text;
  border-radius: 3px;
}

/* 低置信标亮：柔和橙黄底色 + 下划线波浪 */
.sentence .text :deep(mark.lo) {
  background: rgba(255, 152, 0, 0.18);
  border-bottom: 2px dotted #e65100;
  border-radius: 2px;
  color: inherit;
  padding: 0 1px;
}

/* 直接编辑焦点态：明示"这里可以改字" */
.sentence .text:focus {
  outline: 1px solid var(--at-primary);
  background: #fff;
  padding: 0 4px;
  margin: 0 -4px;
}

.save-state {
  color: var(--at-muted);
  font-size: 12px;
}

/* 大屏：限制正文行宽（1920 屏上一行 180 字难以阅读），段头背景同步对齐 */
@media (min-width: 1440px) {
  .segment-head,
  .sentence {
    max-width: 1280px;
    margin-left: auto;
    margin-right: auto;
  }
}

/* 平板：收窄左右留白，时间码略缩 */
@media (max-width: 1023px) {
  .row-anchor {
    padding: 0 10px;
  }

  .sentence {
    padding-left: 16px;
  }

  .sentence-time {
    min-width: 100px;
  }
}

/* 手机：时间码单独一行放在正文上方——390px 宽下若并排，
   108px 的时间码会占掉近三分之一正文宽度 */
@media (max-width: 640px) {
  .row-anchor {
    padding: 0 4px;
  }

  .segment-head {
    padding: 4px 8px;
  }

  .sentence {
    display: block;
    padding: 6px 8px 8px 10px;
    line-height: 1.7;
  }

  .sentence-time {
    display: block;
    min-width: 0;
    margin-bottom: 2px;
    font-size: 11px;
  }

  .sentence .text {
    font-size: 15px;
  }
}
</style>
