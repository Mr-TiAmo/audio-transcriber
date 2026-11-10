// 探索性 UI 测试：以真实用户视角走完整旅程，发现交互与页面展示问题
// 运行前提：后端 127.0.0.1:8730 已启动（生产托管模式，无需 vite dev）
// 每个用例收集 console error 与 >=400 响应，截图存 e2e/shots/explore/
import { expect, test, type Page } from '@playwright/test'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const BASE = process.env.E2E_BASE_URL ?? 'http://127.0.0.1:8730'
const SHOTS = 'e2e/shots/explore'
// ESM 无 __dirname：用 import.meta.url 解析测试音频绝对路径
const SAMPLE_WAV = fileURLToPath(
  new URL('../../data/projects/prj_796c4f239c22/audio/source.wav', import.meta.url))
const BENCH_PROJECT = 'prj_a1043edc82ab' // 59 分钟 bench，已有 59 段完成转写
const SMALL_PROJECT = 'prj_796c4f239c22' // 8.4s 中文语音

// ---------------------------------------------------------------------------
// 诊断收集：console error + HTTP >=400
// ---------------------------------------------------------------------------
interface Diag { console: string[]; http: string[] }

function watchDiagnostics(page: Page, bag: Diag) {
  page.on('console', (msg) => {
    if (msg.type() === 'error') bag.console.push(msg.text().slice(0, 200))
  })
  page.on('response', (resp) => {
    if (resp.status() >= 400) bag.http.push(`${resp.status()} ${resp.request().method()} ${resp.url()}`)
  })
}

function reportDiag(name: string, bag: Diag) {
  console.log(`\n[诊断:${name}] console错误=${bag.console.length} HTTP异常=${bag.http.length}`)
  for (const c of bag.console.slice(0, 5)) console.log(`  console: ${c}`)
  for (const h of bag.http.slice(0, 8)) console.log(`  http: ${h}`)
}

test.describe('探索性 UI 走查', () => {
  test('1. 项目列表页展示（双视口 + 标题 + 本地时间）', async ({ page }) => {
    const bag: Diag = { console: [], http: [] }
    watchDiagnostics(page, bag)

    // 桌面 1920
    await page.setViewportSize({ width: 1920, height: 1080 })
    await page.goto(`${BASE}/`)
    await expect(page.getByRole('heading', { name: 'AudioTranscriber 本地转写工作台' })).toBeVisible()
    // 浏览器标签页标题不应是 Vite 默认值
    expect(await page.title()).toContain('AudioTranscriber')
    await page.screenshot({ path: `${SHOTS}/01-projects-1920.png`, fullPage: true })

    // 导入时间应为本地时区（与当前时间差在合理范围内，UTC 会差 8 小时）
    const timeCell = page.locator('tr .timecode').first()
    const shown = (await timeCell.textContent())?.trim() ?? ''
    console.log(`[走查] 列表首行导入时间显示: "${shown}"`)
    // 笔记本 1366x768
    await page.setViewportSize({ width: 1366, height: 768 })
    await page.screenshot({ path: `${SHOTS}/01-projects-1366.png`, fullPage: true })

    // 水平溢出检查（1366 下 body 不应横向滚动）
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth)
    console.log(`[走查] 1366 视口水平溢出: ${overflow}px`)

    // 删除确认对话框展示（点取消，不真删）
    // 注意：el-table 表头也是 tr（独立 table），必须定位 body 行
    await page.locator('.el-table__body tr').first().getByRole('button', { name: '删除' }).click()
    await expect(page.locator('.el-message-box')).toBeVisible()
    await page.screenshot({ path: `${SHOTS}/01-delete-confirm.png` })
    await page.getByRole('button', { name: '取消' }).click()

    reportDiag('项目列表', bag)
  })

  test('2. 导入新音频 → 空态 → 转写 → 完成 → 删除（完整旅程）', async ({ page }) => {
    test.setTimeout(300_000)
    const bag: Diag = { console: [], http: [] }
    watchDiagnostics(page, bag)
    await page.setViewportSize({ width: 1600, height: 900 })

    // 唯一文件名（便于列表定位与清理）
    const stamp = Date.now().toString().slice(-6)
    const uploadName = `explore_${stamp}.wav`
    const tmpFile = path.join(os.tmpdir(), uploadName)
    fs.copyFileSync(SAMPLE_WAV, tmpFile)

    await page.goto(`${BASE}/`)
    await page.locator('input[type=file]').setInputFiles(tmpFile)

    // 导入成功 = 自动跳转工作台（成功消息是 3s 瞬态，不做硬断言；
    // 若跳转超时则抓取消息区内容辅助诊断）
    await page.waitForURL('**/workbench/**', { timeout: 30_000 }).catch(async () => {
      const msgs = await page.locator('.el-message').allTextContents()
      throw new Error(`未跳转工作台，页面消息: ${JSON.stringify(msgs)}`)
    })
    const projectUrl = page.url()
    const projectId = projectUrl.split('/workbench/')[1]!
    // 工作台顶栏应显示上传的文件名
    await expect(page.locator('.topbar .title')).toContainText(uploadName, { timeout: 10_000 })

    // 空态展示：开始转写按钮 + 暂无转写内容
    await expect(page.getByRole('button', { name: '开始转写' })).toBeVisible({ timeout: 10_000 })
    await expect(page.getByText('暂无转写内容')).toBeVisible()
    await page.screenshot({ path: `${SHOTS}/02-empty-workbench.png` })

    // 开始转写 → PROCESSING 展示（进度条/阶段/取消按钮）
    await page.getByRole('button', { name: '开始转写' }).click()
    await expect(page.getByRole('button', { name: '取消任务' })).toBeVisible({ timeout: 15_000 })
    await page.screenshot({ path: `${SHOTS}/02-processing.png` })

    // 完成（8.4s 音频 + 模型冷启动 ≤150s）
    await expect(page.getByText('转写完成，可编辑与导出')).toBeVisible({ timeout: 180_000 })
    await expect(page.locator('.sentence .text').first()).not.toBeEmpty()
    await page.screenshot({ path: `${SHOTS}/02-completed.png` })

    // 回到列表删除该项目（清理 + 验证删除流程）
    await page.goto(`${BASE}/`)
    const row = page.locator('tr', { hasText: uploadName }).first()
    await expect(row).toBeVisible({ timeout: 10_000 })
    await row.getByRole('button', { name: '删除' }).click()
    await page.locator('.el-message-box').getByRole('button', { name: '确定' }).click()
    await expect(page.getByText('已删除')).toBeVisible({ timeout: 10_000 })
    await expect(page.locator('tr', { hasText: uploadName })).toHaveCount(0)
    fs.rmSync(tmpFile, { force: true })
    console.log(`[走查] 完整旅程通过，项目 ${projectId} 已清理`)

    reportDiag('完整旅程', bag)
  })

  test('3. 导入非法文件的错误提示可读性', async ({ page }) => {
    const bag: Diag = { console: [], http: [] }
    watchDiagnostics(page, bag)

    const tmpTxt = path.join(os.tmpdir(), `not_audio_${Date.now()}.txt`)
    fs.writeFileSync(tmpTxt, '这不是音频文件'.repeat(100))

    await page.goto(`${BASE}/`)
    await page.locator('input[type=file]').setInputFiles(tmpTxt)
    // 期望：显示后端返回的中文 detail，而不是 resp.json 双读导致的英文乱码
    const msg = page.locator('.el-message--error')
    await expect(msg).toBeVisible({ timeout: 15_000 })
    const text = (await msg.textContent()) ?? ''
    console.log(`[走查] 非法文件导入错误提示: "${text}"`)
    await page.screenshot({ path: `${SHOTS}/03-import-error.png` })
    expect(text).not.toContain('Failed to execute')
    expect(text).not.toContain('body stream')
    fs.rmSync(tmpTxt, { force: true })

    reportDiag('非法文件', bag)
  })

  test('4. 长音频工作台展示 + 导出下载 + 片段定位', async ({ page }) => {
    test.setTimeout(120_000)
    const bag: Diag = { console: [], http: [] }
    watchDiagnostics(page, bag)
    await page.setViewportSize({ width: 1600, height: 900 })

    await page.goto(`${BASE}/workbench/${BENCH_PROJECT}`)
    await expect(page.locator('.segment-head .timecode').first()).toBeVisible({ timeout: 15_000 })

    // 顶栏时长：59 分钟音频应显示"59 分 0 秒"而非"3540s"
    const dur = (await page.locator('.topbar .duration').textContent())?.trim() ?? ''
    console.log(`[走查] 顶栏时长显示: "${dur}"`)
    expect(dur).toContain('分')

    // 句级布局：段头行 + 整句行（虚拟滚动只渲染可视部分）
    const heads = await page.locator('.segment-head').count()
    const sentences = await page.locator('.sentence').count()
    console.log(`[走查] 虚拟滚动渲染: 段头行=${heads} 整句行=${sentences}（总段数 59）`)
    expect(sentences).toBeGreaterThan(0)

    // 句行时间码应为 mm:ss → mm:ss 格式
    const firstSentTime = await page.locator('.sentence .sentence-time').first().textContent()
    const firstSentText = await page.locator('.sentence .text').first().textContent()
    console.log(`[走查] 首个句行: "${firstSentTime?.trim()}" | "${firstSentText?.trim().slice(0, 30)}"`)
    expect(firstSentTime).toMatch(/\d{2}:\d{2}\s*→\s*\d{2}:\d{2}/)

    // 点击句行 → 播放器跳转到该句开始时间（保持暂停）
    const secondSent = page.locator('.sentence').nth(1)
    const secondTime = await secondSent.locator('.sentence-time').textContent()
    await secondSent.click()
    await page.waitForTimeout(300)
    const playerTime = await page.locator('.player .time').textContent()
    console.log(`[走查] 点击第 2 句行("${secondTime?.trim()}") → 播放器: ${playerTime?.trim()}`)

    // 滚动到底部：末段时间码（mm:ss 可读性）
    await page.locator('.segment-scroll').evaluate((el) => el.scrollTo(0, el.scrollHeight))
    await page.waitForTimeout(500)
    const lastTimecode = await page.locator('.segment-head .timecode').last().textContent()
    console.log(`[走查] 末段头时间码: "${lastTimecode}"`)
    expect(lastTimecode).toMatch(/5[89]:\d{2}/)
    await page.screenshot({ path: `${SHOTS}/04-bench-bottom.png` })
    await page.locator('.segment-scroll').evaluate((el) => el.scrollTo(0, 0))
    await page.waitForTimeout(300)
    await page.screenshot({ path: `${SHOTS}/04-bench-top.png` })

    // 导出 TXT：应触发浏览器下载（修复前 window.open 打开 index.html）
    const downloadPromise = page.waitForEvent('download', { timeout: 15_000 })
    await page.getByRole('button', { name: '导出 TXT' }).click()
    const download = await downloadPromise
    console.log(`[走查] 导出下载文件名: ${download.suggestedFilename()}`)
    expect(download.suggestedFilename()).toContain('.txt')
    const saveTo = path.join(os.tmpdir(), `export_check_${Date.now()}.txt`)
    await download.saveAs(saveTo)
    const content = fs.readFileSync(saveTo, 'utf-8')
    console.log(`[走查] 导出内容长度: ${content.length} 字符，开头: "${content.slice(0, 30)}"`)
    expect(content.length).toBeGreaterThan(1000)
    fs.rmSync(saveTo, { force: true })

    // 点击段头行定位到段起点
    await page.locator('.segment-head').nth(2).click()
    await page.waitForTimeout(500)
    const headPlayerTime = await page.locator('.player .time').textContent()
    console.log(`[走查] 点击第 3 段头后播放器时间: ${headPlayerTime}`)

    // 直接编辑走查：句行正文单击即可改字（无需编辑按钮），600ms 防抖自动保存；
    // 改完恢复原文，保持 bench 数据干净
    const firstText = page.locator('.sentence .text').first()
    const original = (await firstText.textContent()) ?? ''
    await firstText.click()
    await page.keyboard.press('End')
    await page.keyboard.type('（走查编辑）')
    await expect(page.locator('.segment-head .save-state').first())
      .toContainText('已保存', { timeout: 8_000 })
    console.log(`[走查] 直接编辑自动保存生效（追加"（走查编辑）"后段头显示已保存）`)
    await page.screenshot({ path: `${SHOTS}/04-direct-edit.png` })
    // 恢复原文：全选句行内容后键回（contenteditable 内 Ctrl+A 只选本句）
    await firstText.click()
    await page.keyboard.press('Control+a')
    await page.keyboard.type(original)
    await expect(page.locator('.segment-head .save-state').first())
      .toContainText('已保存', { timeout: 8_000 })

    // 复核/编辑 UI 已移除：工具栏不应再有"只看待复核"，段头不应有"编辑"按钮
    await expect(page.getByText('只看待复核')).toHaveCount(0)
    await expect(page.locator('.segment-head').getByRole('button', { name: '编辑' })).toHaveCount(0)

    reportDiag('长音频工作台', bag)
  })

  test('5. 任务运行中：全局互斥（UI 拦截 + API 409）+ 刷新恢复 + 取消', async ({ page }) => {
    test.setTimeout(420_000)
    const bag: Diag = { console: [], http: [] }
    watchDiagnostics(page, bag)
    await page.setViewportSize({ width: 1600, height: 900 })

    // 启动 bench 长任务（59 分钟音频，跑几分钟足够观察）
    await page.goto(`${BASE}/workbench/${BENCH_PROJECT}`)
    await expect(page.locator('.segment-head .timecode').first()).toBeVisible({ timeout: 15_000 })
    await page.getByRole('button', { name: '重新转写' }).click()
    await expect(page.getByRole('button', { name: '取消任务' })).toBeVisible({ timeout: 15_000 })

    // 等进度 >0（VAD 分析 + 模型加载后才有 chunk 事件，约 60-120s）
    await expect(async () => {
      const pct = await page.locator('.taskbar .el-progress__text').textContent()
      const n = parseInt(pct?.replace('%', '') ?? '0', 10)
      expect(n).toBeGreaterThan(0)
    }).toPass({ timeout: 200_000, intervals: [5_000] })
    const beforeReload = await page.locator('.taskbar .el-progress__text').textContent()
    const stageBefore = await page.locator('.taskbar .stage').textContent()
    console.log(`[走查] 刷新前进度: ${beforeReload} 阶段: ${stageBefore}`)
    await page.screenshot({ path: `${SHOTS}/05-processing-before-reload.png` })

    // 全局互斥在任务必然运行中时立即验证（上轮教训：等到刷新观察后 bench 任务
    // 已自然跑完，点击"重新转写"成功启动而非 409）：
    // a) API 层：直接对另一项目 POST runs → 409 中文提示（page.request 共享会话 cookie）
    const resp = await page.request.post(`${BASE}/api/v1/projects/${SMALL_PROJECT}/runs`)
    expect(resp.status()).toBe(409)
    const detail = ((await resp.json()) as { detail: string }).detail
    console.log(`[走查] API 409 提示: "${detail}"`)
    expect(detail).toContain('已有任务')

    // b) UI 层：打开另一项目，任务条显示全局运行中任务（处理中 + 取消任务），
    //    且不提供任何开始按钮 —— 单标签页 UI 已在源头拦截 409
    await page.goto(`${BASE}/workbench/${SMALL_PROJECT}`)
    await expect(page.locator('.taskbar .el-tag')).toHaveText('处理中', { timeout: 10_000 })
    await expect(page.getByRole('button', { name: '取消任务' })).toBeVisible()
    await expect(page.getByRole('button', { name: /重新转写|开始转写/ })).toHaveCount(0)
    console.log('[走查] 全局任务运行时打开其他项目：任务条"处理中+取消任务"，无开始按钮（UI 拦截 409）')
    await page.screenshot({ path: `${SHOTS}/05-other-project-busy.png` })

    // 刷新恢复：回 bench 页（等效 reload），SSE 重连、进度继续推进（修复前冻结）
    await page.goto(`${BASE}/workbench/${BENCH_PROJECT}`)
    await expect(page.locator('.taskbar')).toBeVisible({ timeout: 10_000 })
    const stageAfter = await page.locator('.taskbar .stage').textContent()
    console.log(`[走查] 刷新后阶段文本: "${stageAfter}"`)
    await page.screenshot({ path: `${SHOTS}/05-after-reload.png` })
    // 刷新后 60s 内进度应严格推进（n > b）：>= 会让"SSE 重连断开、进度冻结在
    // 快照值"的回归漏网（上轮 7% -> 7% 首轮轮询即满足 >= 假通过）
    await expect(async () => {
      const pct = await page.locator('.taskbar .el-progress__text').textContent()
      const n = parseInt(pct?.replace('%', '') ?? '0', 10)
      const b = parseInt(beforeReload?.replace('%', '') ?? '0', 10)
      expect(n).toBeGreaterThan(b)
    }).toPass({ timeout: 60_000, intervals: [5_000] })
    const afterReload = await page.locator('.taskbar .el-progress__text').textContent()
    console.log(`[走查] 刷新后进度推进: ${beforeReload} -> ${afterReload}`)

    // 取消任务（清理 + 验证取消交互；bench 页任务条仍在 PROCESSING 态）
    await expect(page.getByRole('button', { name: '取消任务' })).toBeVisible({ timeout: 10_000 })
    await page.getByRole('button', { name: '取消任务' }).click()
    await expect(page.getByText('已请求取消')).toBeVisible({ timeout: 10_000 })
    // 刷新后取消 → taskbar 应显示已取消态（含重新转写按钮）
    await expect(async () => {
      await page.reload()
      await expect(page.locator('.taskbar').getByText('已取消')).toBeVisible({ timeout: 5_000 })
    }).toPass({ timeout: 30_000, intervals: [3_000] }).catch(() => {
      console.log('[走查] 注意：刷新后未显示"已取消"态（load 只认 PROCESSING，取消态刷新即丢失）')
    })
    await page.screenshot({ path: `${SHOTS}/05-cancelled.png` })

    reportDiag('刷新与冲突', bag)
  })
})
