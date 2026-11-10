<script setup lang="ts">
// 全文查找替换（§12.5：先 preview 再确认 apply，批量替换可整组撤销）
import { computed, ref } from 'vue'
import { ElMessage } from 'element-plus'

import { useWorkbenchStore } from '@/stores/workbench'
import type { ReplaceHit } from '@/types'

const visible = defineModel<boolean>({ default: false })
const store = useWorkbenchStore()

const find = ref('')
const replace = ref('')
const hits = ref<ReplaceHit[]>([])
const previewing = ref(false)
const applying = ref(false)
const lastOperationId = ref<string | null>(null)

const hitCount = computed(() => hits.value.length)

async function preview() {
  if (!find.value) {
    ElMessage.warning('请输入查找内容')
    return
  }
  previewing.value = true
  try {
    const result = await store.previewReplace(find.value, replace.value)
    hits.value = result.hits
    lastOperationId.value = null
    if (!result.hits.length) ElMessage.info('没有匹配的片段')
  } catch (err) {
    ElMessage.error((err as Error).message)
  } finally {
    previewing.value = false
  }
}

async function apply() {
  if (!hitCount.value) return
  applying.value = true
  try {
    const result = await store.applyReplace(find.value, replace.value)
    lastOperationId.value = result.operationId
    ElMessage.success(`已替换 ${result.count} 处（可撤销）`)
    hits.value = []
  } catch (err) {
    ElMessage.error((err as Error).message)
  } finally {
    applying.value = false
  }
}

async function undo() {
  if (!lastOperationId.value) return
  try {
    const result = await store.undoReplace(lastOperationId.value)
    if (result.skipped.length) {
      ElMessage.warning(`已恢复 ${result.restored} 处，${result.skipped.length} 处因后续有编辑跳过`)
    } else {
      ElMessage.success(`已撤销，恢复 ${result.restored} 处`)
    }
    lastOperationId.value = null
  } catch (err) {
    ElMessage.error((err as Error).message)
  }
}

function onOpen() {
  hits.value = []
  lastOperationId.value = null
}

function fmt(ms: number): string {
  const s = ms / 1000
  // 与片段列表一致：长音频（≥60s）用 m:ss 格式
  if (s < 60) return `${s.toFixed(1)}s`
  const m = Math.floor(s / 60)
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`
}
</script>

<template>
  <!-- 宽度自适应：桌面 640px，窄屏留出 16px 页边距（手机上 390-32=358） -->
  <el-dialog
    v-model="visible"
    title="查找替换"
    width="min(640px, calc(100vw - 32px))"
    top="8vh"
    @open="onOpen"
  >
    <div class="form">
      <el-input v-model="find" placeholder="查找内容" clearable @keyup.enter="preview" />
      <el-input v-model="replace" placeholder="替换为" clearable />
      <div class="actions">
        <el-button :loading="previewing" @click="preview">预览</el-button>
        <el-button type="primary" :disabled="!hitCount" :loading="applying" @click="apply">
          全部替换（{{ hitCount }} 处）
        </el-button>
        <el-button
          v-if="lastOperationId" type="warning" plain @click="undo"
        >撤销上次替换</el-button>
      </div>
    </div>

    <div class="hits" v-if="hitCount">
      <div class="hit" v-for="h in hits" :key="h.segmentId">
        <span class="timecode">{{ fmt(h.startMs) }}</span>
        <div class="old">{{ h.oldText }}</div>
        <div class="arrow">↓ 替换后</div>
        <div class="new">{{ h.newText }}</div>
      </div>
    </div>
  </el-dialog>
</template>

<style scoped>
.form {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.hits {
  margin-top: 16px;
  max-height: 50vh;
  overflow-y: auto;
  border-top: 1px solid var(--at-border);
  padding-top: 12px;
}

.hit {
  padding: 8px 0;
  border-bottom: 1px dashed var(--at-border);
}

.old {
  color: var(--at-muted);
  text-decoration: line-through;
  margin: 4px 0 0;
}

.arrow {
  font-size: 12px;
  color: var(--at-primary);
  margin: 2px 0;
}

.new {
  font-weight: 600;
}
</style>
