// 与后端 §11 接口对应的类型（字段名与实际 JSON 对齐）

export interface ProjectListItem {
  id: string
  original_name: string
  duration_ms: number
  status: string
  created_at: string
  updated_at: string
}

/** POST /projects 导入响应（§11：后端返回驼峰字段，与列表的蛇形不同） */
export interface ImportedProject {
  id: string
  originalName: string | null
  durationMs: number
  format: string | null
  codec: string | null
  sampleRate: number | null
  channels: number | null
}

/** GET /models：可选 ASR 模型（前端下拉框）。available=false 表示本地未安装。 */
export interface AsrModel {
  key: string
  name: string
  label: string
  available: boolean
}

export interface ProjectDetail {
  id: string
  original_name: string
  audio_path: string
  pcm_path: string
  file_size: number
  duration_ms: number
  format: string | null
  codec: string | null
  sample_rate: number | null
  channels: number | null
  status: string
  created_at: string
  updated_at: string
}

export interface Segment {
  id: string
  runId: string
  sequenceNo: number
  startMs: number
  endMs: number
  originalText: string
  editedText: string | null
  finalText: string
  reviewStatus: 'pending' | 'confirmed'
  riskFlags: string[]
  // 低置信字符区间：后端 JSON 为 [[start, end], ...]（码点索引）
  lowConfSpans?: Array<[number, number]>
  // 字符级对齐时间（毫秒，绝对时间；码点索引与 finalText 对齐）：
  // 由 ForcedAligner 产出，缺失字符为 null。仅原始文本可用，编辑后失效。
  alignedTokens?: Array<[number, number] | null> | null
  // 段级风险分（高=优先复核），由 ASR 脚本按 entropy/文本/对齐异常加权
  reviewPriority?: number | null
  updatedAt: string
}

export interface RunInfo {
  id: string | null
  projectId: string
  status: 'IDLE' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'CANCELLED'
  stage: string | null
  progress_percent: number
  model_name: string | null
  error_code: string | null
  error_message: string | null
  started_at: string | null
}

export interface ReplaceHit {
  segmentId: string
  sequenceNo: number
  startMs: number
  endMs: number
  oldText: string
  newText: string
}
