// 后端 API 封装（§11 接口清单；开发期经 vite proxy 同源访问）
import type {
  AsrModel, ImportedProject, ProjectDetail, ProjectListItem, ReplaceHit, RunInfo, Segment,
} from '@/types'

const BASE = '/api/v1'

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, headers: {} }
  if (body !== undefined) {
    init.headers = { 'Content-Type': 'application/json' }
    init.body = JSON.stringify(body)
  }
  const resp = await fetch(BASE + path, init)
  if (!resp.ok) {
    let message = `HTTP ${resp.status}`
    try {
      const data = await resp.json()
      if (data?.detail) message = data.detail
    } catch { /* 非 JSON 错误体 */ }
    throw new Error(message)
  }
  const ct = resp.headers.get('content-type') || ''
  if (ct.includes('application/json')) return resp.json() as Promise<T>
  return undefined as T
}

export const api = {
  // 项目
  listProjects: () => request<ProjectListItem[]>('GET', '/projects'),
  getProject: (id: string) => request<ProjectDetail>('GET', `/projects/${id}`),
  createProject: async (file: File) => {
    const form = new FormData()
    form.append('file', file)
    const resp = await fetch(`${BASE}/projects`, { method: 'POST', body: form })
    if (!resp.ok) {
      let message = `HTTP ${resp.status}`
      // 注意：resp.json() 只能读一次，先落变量再取 detail
      try {
        const data = await resp.json()
        if (data?.detail) message = data.detail
      } catch { /* 非 JSON 错误体 */ }
      throw new Error(message)
    }
    return resp.json() as Promise<ImportedProject>
  },
  deleteProject: (id: string) => request<{ deleted: string }>('DELETE', `/projects/${id}`),

  // 任务
  // 模型列表：未安装的模型 available=false，前端下拉置灰
  listModels: () => request<AsrModel[]>('GET', '/models'),
  startRun: (projectId: string, model = '0.6b', fast = false) =>
    request<{ runId: string; status: string }>(
      'POST', `/projects/${projectId}/runs`, { model, fast }),
  cancelRun: (projectId: string, runId: string) =>
    request<{ runId: string; status: string }>('POST', `/projects/${projectId}/runs/${runId}/cancel`),
  currentTask: () => request<RunInfo>('GET', '/tasks/current'),

  // 转写内容
  getTranscript: (projectId: string) =>
    request<{ runId: string; segments: Segment[] }>('GET', `/projects/${projectId}/transcript`),
  saveSegment: (segmentId: string, payload: { editedText?: string; reviewStatus?: string }) =>
    request<{ id: string; saved: boolean; updatedAt: string }>(
      'PATCH', `/segments/${segmentId}`, payload),

  // 查找替换 / 撤销（§12.5）
  previewReplace: (projectId: string, find: string, replace: string) =>
    request<{ mode: string; hits: ReplaceHit[] }>(
      'POST', `/projects/${projectId}/replace`, { mode: 'preview', find, replace }),
  applyReplace: (projectId: string, find: string, replace: string) =>
    request<{ mode: string; operationId: string | null; count: number; segments: ReplaceHit[] }>(
      'POST', `/projects/${projectId}/replace`, { mode: 'apply', find, replace }),
  undoOperation: (operationId: string) =>
    request<{ restored: number; skipped: string[] }>('POST', `/edit-operations/${operationId}/undo`),

  // 资源 URL（音频走 proxy 支持 Range；导出为文件下载）
  audioUrl: (projectId: string) => `${BASE}/projects/${projectId}/audio`,
  exportUrl: (projectId: string) => `${BASE}/projects/${projectId}/export`,
}

/** SSE 订阅任务事件（§11.1）；返回取消订阅函数。 */
export function subscribeEvents(
  runId: string,
  onEvent: (event: { kind: string } & Record<string, unknown>) => void,
): () => void {
  const es = new EventSource(`${BASE}/tasks/${runId}/events`)
  const forward = (ev: MessageEvent) => {
    try { onEvent(JSON.parse(ev.data)) } catch { /* 忽略坏帧 */ }
  }
  // 后端事件名用连字符（asr-completed 等）
  es.onmessage = forward
  const kinds = [
    'task-snapshot', 'task-completed', 'task-failed', 'task-cancelled',
    'asr-started', 'asr-model-loaded', 'asr-completed',
    'progress-updated', 'phase-updated', 'postprocess-completed',
  ]
  for (const k of kinds) es.addEventListener(k, forward as EventListener)
  return () => es.close()
}
