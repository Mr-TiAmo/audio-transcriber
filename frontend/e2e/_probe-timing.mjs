// 临时探针：实际跑一次转写，逐秒采样 阶段 / 进度 / 计时文本，
// 用于回归「百分比回退」「ETA 归零后又重新倒计时」两个显示问题。
// 用法：node e2e/_probe-timing.mjs <projectId> [超时秒]
import { chromium } from '@playwright/test'

const BASE = 'http://127.0.0.1:8730'
const pid = process.argv[2]
const limitS = Number(process.argv[3] ?? 150)

const browser = await chromium.launch()
const page = await browser.newPage({ locale: 'zh-CN' })
await page.goto(`${BASE}/workbench/${pid}`)
// 等 store 加载完快照（IDLE 且无转写时任务栏不渲染）
await page.locator('.taskbar').waitFor({ timeout: 30_000 })
// 已有任务在跑时直接接入采样，否则点「重新转写」
const running = await page.locator('.taskbar .timing').count()
if (!running) {
  await page.getByRole('button', { name: /重新转写|开始转写/ }).click()
}
// 等任务真正进入处理中（点击到 POST 返回之间仍显示旧状态）
await page.locator('.taskbar .timing').waitFor({ state: 'attached', timeout: 20_000 })

const samples = []
const t0 = Date.now()
let lastStatus = ''
while ((Date.now() - t0) / 1000 < limitS) {
  const snap = await page.evaluate(() => {
    const q = (sel) => document.querySelector(sel)?.textContent?.trim() ?? ''
    const bar = document.querySelector('.progress .el-progress-bar__inner')
    return {
      status: q('.taskbar .el-tag'),
      stage: q('.taskbar .stage'),
      timing: q('.taskbar .timing'),
      percent: bar?.style.width ?? '',
    }
  })
  const t = ((Date.now() - t0) / 1000).toFixed(1)
  samples.push({ t, ...snap })
  console.log(t, snap.status, snap.stage, snap.percent, snap.timing)
  lastStatus = snap.status
  if (snap.status.includes('已完成') || snap.status.includes('失败')) break
  await page.waitForTimeout(1000)
}
await browser.close()

// 分析：进度回退、ETA 归零后回升
const bad = []
for (let i = 1; i < samples.length; i++) {
  const a = samples[i - 1], b = samples[i]
  // 宽度带 % 单位（CSS 过渡中的中间值），解析成数字比较
  const pa = parseFloat(a.percent), pb = parseFloat(b.percent)
  if (pa && pb && pb < pa) bad.push(`进度回退 ${a.t}s ${pa}% -> ${b.t}s ${pb}%`)
  const ea = a.timing.match(/(\d+(?: 分)? ?\d* ?秒)/g)
  const eb = b.timing.match(/(\d+(?: 分)? ?\d* ?秒)/g)
  if (ea?.length >= 2 && eb?.length >= 2 && a.stage === b.stage) {
    const toS = (s) => {
      const m = s.match(/(?:(\d+) 分 )?(\d+) 秒/)
      return Number(m?.[1] ?? 0) * 60 + Number(m?.[2] ?? 0)
    }
    if (toS(eb[1]) - toS(ea[1]) > 2) bad.push(`ETA 回升 ${a.t}s ${ea[1]} -> ${b.t}s ${eb[1]}`)
  }
}
console.log('\n最终状态:', lastStatus)
console.log('异常:', bad.length ? bad.join('\n') : '无')