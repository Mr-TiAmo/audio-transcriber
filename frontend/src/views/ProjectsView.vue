<script setup lang="ts">
// 项目列表页（§12.1 顶部导航的历史项目入口）：导入音频 + 进入工作台
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'

import { useProjectsStore } from '@/stores/projects'
import { useWorkbenchStore } from '@/stores/workbench'
import { api } from '@/api/client'
import { useViewport } from '@/composables/useViewport'

const router = useRouter()
const projects = useProjectsStore()
const workbench = useWorkbenchStore()
const busyProjectId = ref<string | null>(null)
// 手机端压缩表格：去掉「导入时间」列（CSS 隐藏会留下列宽空洞），按钮改短文案
const { isMobile } = useViewport()

onMounted(() => {
  void projects.refresh()
})

function fmtDuration(ms: number | null | undefined): string {
  if (!Number.isFinite(ms as number)) return '—'
  const s = Math.round((ms as number) / 1000)
  const m = Math.floor(s / 60)
  return `${m} 分 ${s % 60} 秒`
}

// created_at 是 UTC ISO 串（...+00:00）：直接 slice 会把 UTC 时间当本地时间
// 展示（中国用户差 8 小时），必须经 Date 解析后按本地时区格式化
function fmtTime(iso: string | null | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso.slice(0, 19).replace('T', ' ')
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
    `${p(d.getHours())}:${p(d.getMinutes())}`
}

async function onFileChange(file: File) {
  try {
    const created = await projects.importAudio(file)
    ElMessage.success(`已导入：${created.originalName ?? 'audio'}（${fmtDuration(created.durationMs)}）`)
    await router.push(`/workbench/${created.id}`)
  } catch (err) {
    ElMessage.error(`导入失败：${(err as Error).message}`)
  }
}

async function enterWorkbench(id: string) {
  await router.push(`/workbench/${id}`)
}

async function removeProject(id: string, name: string) {
  try {
    await ElMessageBox.confirm(`删除项目「${name}」及其全部转写数据？`, '删除确认', {
      type: 'warning',
    })
  } catch { /* 取消 */ return }
  busyProjectId.value = id
  try {
    await projects.remove(id)
    ElMessage.success('已删除')
  } catch (err) {
    ElMessage.error(`删除失败：${(err as Error).message}`)
  } finally {
    busyProjectId.value = null
  }
}
</script>

<template>
  <div class="projects-page">
    <header class="page-header">
      <h1>AudioTranscriber 本地转写工作台</h1>
      <p class="hint">音频不出本机 · Qwen3-ASR · 数字规范化</p>
    </header>

    <el-upload
      class="uploader"
      drag
      :auto-upload="false"
      :show-file-list="false"
      accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg"
      :on-change="(f: { raw?: File }) => f.raw && onFileChange(f.raw)"
    >
      <el-icon :size="42"><UploadFilled /></el-icon>
      <div class="el-upload__text">拖入音频文件，或<em>点击选择</em></div>
      <template #tip>
        <div class="el-upload__tip">支持 wav / mp3 / m4a / flac / ogg，时长上限 60 分钟</div>
      </template>
    </el-upload>

    <section class="history">
      <h2>历史项目</h2>
      <el-empty v-if="!projects.loading && !projects.projects.length" description="暂无项目" />
      <el-table v-else :data="projects.projects" v-loading="projects.loading">
        <el-table-column
          prop="original_name" label="名称"
          :min-width="isMobile ? 140 : 240" show-overflow-tooltip
        />
        <el-table-column label="时长" :width="isMobile ? 92 : 120">
          <template #default="{ row }">
            <span class="timecode">{{ fmtDuration(row.duration_ms) }}</span>
          </template>
        </el-table-column>
        <!-- 手机端不为「导入时间」预留列宽，避免整表被挤出横向滚动 -->
        <el-table-column v-if="!isMobile" prop="created_at" label="导入时间" width="180">
          <template #default="{ row }">
            <span class="timecode">{{ fmtTime(row.created_at) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" :width="isMobile ? 132 : 180" align="center">
          <template #default="{ row }">
            <el-button size="small" type="primary" plain @click="enterWorkbench(row.id)">
              {{ isMobile ? '打开' : '打开工作台' }}
            </el-button>
            <el-button
              size="small" type="danger" plain
              :loading="busyProjectId === row.id"
              @click="removeProject(row.id, row.original_name)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </section>
  </div>
</template>

<style scoped>
.projects-page {
  /* 大屏放宽：表格在 1920 屏上不再缩成中间一条 */
  max-width: 1100px;
  margin: 0 auto;
  padding: 32px 24px;
}

.page-header h1 {
  margin: 0 0 4px;
  font-size: 22px;
}

.hint {
  color: var(--at-muted);
  margin: 0 0 24px;
}

.history {
  margin-top: 32px;
}

.history h2 {
  font-size: 16px;
  margin: 0 0 12px;
}

/* 平板：收窄页边距 */
@media (max-width: 1023px) {
  .projects-page {
    padding: 24px 16px;
  }
}

/* 手机：标题缩小、拖拽区降低高度，给历史列表留出首屏空间 */
@media (max-width: 640px) {
  .projects-page {
    padding: 16px 10px;
  }

  .page-header h1 {
    font-size: 18px;
    line-height: 1.4;
  }

  .hint {
    margin-bottom: 16px;
    font-size: 12px;
  }

  .history {
    margin-top: 20px;
  }

  .uploader :deep(.el-upload-dragger) {
    padding: 20px 12px;
  }

  .uploader :deep(.el-upload__text) {
    font-size: 13px;
  }
}
</style>
