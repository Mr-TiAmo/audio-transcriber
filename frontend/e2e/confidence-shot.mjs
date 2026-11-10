// 临时脚本：打开工作台，统计低置信标亮 <mark class="lo"> 并截图。
// 用法：node e2e/confidence-shot.mjs [baseUrl]
import { chromium } from '@playwright/test'

const base = process.argv[2] ?? 'http://127.0.0.1:8730'
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
await page.goto(base, { waitUntil: 'networkidle' })
// 打开 bench_10min 项目（列表首行）
const row = page.locator('tr', { hasText: 'bench_10min.wav' }).first()
await row.getByRole('button', { name: '打开工作台' }).click()
await page.waitForSelector('.sentence .text', { timeout: 30000 })
await page.waitForTimeout(1500)
const marks = page.locator('.sentence .text mark.lo')
const count = await marks.count()
const texts = await marks.allTextContents()
console.log('mark count:', count)
console.log('mark texts:', JSON.stringify(texts))
await page.screenshot({ path: 'e2e/shots/confidence-highlight-new.png', fullPage: false })
await browser.close()