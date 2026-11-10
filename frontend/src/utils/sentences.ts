// 整句拆分与句级时间定位。
// 优先使用后端 ForcedAligner 产出的字符级时间戳（alignedTokens，毫秒）；
// 没有对齐数据（模型缺失/编辑后失效/旧数据）时回退到「按字数比例插值」，
// 并在 SentenceSlice.estimated 上标明，避免把估算时间当作真实时间展示。

export interface SentenceSlice {
  text: string
  startMs: number
  endMs: number
  /** true = 该句时间是按字数插值的估算值，false = 来自字符级对齐 */
  estimated: boolean
}

// 句末标点（含换行）：一级拆分点
const SENTENCE_END_CHARS = '。！？；!?;\n'
// 停顿标点：超长句的二级拆分点
const CLAUSE_END_CHARS = '，、：,:'

// 超过该字数的一"句"按停顿标点二次拆分（口语转写常见一逗到底）
const MAX_SENTENCE_CHARS = 60
// 短于该字数的碎句不允许独立成行（用户确认 16 字）：与相邻句双向合并，
// 避免"是从1896年至1931年，"这类短行单独占一行
const MIN_SENTENCE_CHARS = 16

/** 码点区间（start/end 均为码点下标，与后端 alignedTokens 口径一致）。 */
interface Range {
  text: string
  start: number
  end: number
}

function splitRanges(
  cps: string[],
  from: number,
  to: number,
  isDelim: (ch: string) => boolean,
): Range[] {
  const out: Range[] = []
  let start = from
  const push = (s: number, e: number) => {
    // 两端空白不进入句子文本，但保留原始码点区间用于查对齐时间
    while (s < e && /\s/.test(cps[s]!)) s++
    while (e > s && /\s/.test(cps[e - 1]!)) e--
    if (e > s) out.push({ text: cps.slice(s, e).join(''), start: s, end: e })
  }
  for (let i = from; i < to; i++) {
    if (isDelim(cps[i]!)) {
      push(start, i + 1)
      start = i + 1
    }
  }
  push(start, to)
  return out
}

/**
 * 取一句内首个/末个有对齐时间的字符，作为句子起止。
 * 对齐器不处理标点，所以这里天然跳过标点；整句都无时间时返回 null。
 */
function alignedSentenceTime(
  charTimes: Array<[number, number] | null> | null | undefined,
  from: number,
  to: number,
): [number, number] | null {
  if (!charTimes || !charTimes.length) return null
  let first: number | null = null
  let last: number | null = null
  const end = Math.min(to, charTimes.length)
  for (let i = from; i < end; i++) {
    const t = charTimes[i]
    if (!t) continue
    if (first === null) first = t[0]
    last = last === null ? t[1] : Math.max(last, t[1])
  }
  if (first === null || last === null) return null
  return [first, last]
}

/**
 * 把段内文本拆成整句列表。
 *
 * @param charTimes 可选：段文本逐字符的对齐时间 `[startMs, endMs]`（码点对齐），
 *   缺失字符为 null。该数组与传入的 `text` 必须等长且按码点计。
 */
export function splitSentences(
  text: string,
  startMs: number,
  endMs: number,
  charTimes?: Array<[number, number] | null> | null,
): SentenceSlice[] {
  const cps = Array.from(text)
  // 字符时间戳必须与文本逐码点对齐；长度不符说明文本已被编辑，
  // 此时对齐数据已失效，整体回退到估算，避免错位到别的句子。
  const times = charTimes && charTimes.length === cps.length ? charTimes : null
  const raw = splitRanges(cps, 0, cps.length, (ch) => SENTENCE_END_CHARS.includes(ch))

  const parts: Range[] = []
  for (const r of raw) {
    if (r.end - r.start > MAX_SENTENCE_CHARS) {
      parts.push(...splitRanges(cps, r.start, r.end, (ch) => CLAUSE_END_CHARS.includes(ch)))
    } else {
      parts.push(r)
    }
  }

  // 碎片双向合并：上一句或当前句任一不足 MIN_SENTENCE_CHARS 就并入上一句
  const merged: Range[] = []
  for (const p of parts) {
    const last = merged[merged.length - 1]
    if (last && (last.end - last.start < MIN_SENTENCE_CHARS || p.end - p.start < MIN_SENTENCE_CHARS)) {
      merged[merged.length - 1] = { text: last.text + p.text, start: last.start, end: p.end }
    } else {
      merged.push(p)
    }
  }
  // 首句是碎片且无上一句可并：向后并入下一句
  if (merged.length > 1 && merged[0]!.end - merged[0]!.start < MIN_SENTENCE_CHARS) {
    const head = merged.shift()!
    merged[0] = { text: head.text + merged[0]!.text, start: head.start, end: merged[0]!.end }
  }
  if (!merged.length) return []

  const duration = Math.max(0, endMs - startMs)
  const total = merged.reduce((n, r) => n + (r.end - r.start), 0) || 1
  let acc = 0
  return merged.map((r) => {
    const estStart = startMs + Math.round((acc / total) * duration)
    acc += r.end - r.start
    const estEnd = startMs + Math.round((acc / total) * duration)

    const hit = alignedSentenceTime(times, r.start, r.end)
    if (hit) {
      // 对齐时间可能落在块的 overlap 区（段边界外），夹到本段范围内
      const st = Math.max(startMs, Math.min(hit[0], endMs - 1))
      const en = Math.max(st + 1, Math.min(hit[1], endMs))
      return { text: r.text, startMs: st, endMs: en, estimated: false }
    }
    return {
      text: r.text,
      startMs: estStart,
      endMs: Math.max(estEnd, estStart + 1),
      estimated: true,
    }
  })
}

/** mm:ss 时间显示（上限 60 分钟音频，无需小时位）。 */
export function fmtClock(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000))
  const m = Math.floor(s / 60)
  return `${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}