// 阶段 2 端到端冒烟（Playwright 无头）：工作台全链路 UI 验证
// 运行前提：后端 127.0.0.1:8730 已启动；dev 模式另需 vite 5173
// E2E_BASE_URL 可切换验证目标：默认 vite dev 5173，设 http://127.0.0.1:8730 验证生产托管
import { expect, test } from '@playwright/test'

const BASE = process.env.E2E_BASE_URL ?? 'http://localhost:5173'

test('工作台：打开项目 -> 片段编辑 -> 查找替换 -> 重新转写', async ({ page }) => {
  // 拦截音频请求（无头环境不播放音频，只验证文本链路）
  await page.route('**/audio', (route) => route.abort())

  // 前置用例可能留下正在终止的任务，先等后端空闲，避免重新转写被 409 拒绝
  await expect.poll(async () => {
    const resp = await page.request.get(`${BASE}/api/v1/tasks/current`)
    return (await resp.json()).status
  }, { timeout: 30_000 }).toBe('IDLE')

  // 1. 项目列表页渲染
  await page.goto(`${BASE}/`)
  await expect(page.getByRole('heading', { name: 'AudioTranscriber 本地转写工作台' })).toBeVisible()
  await expect(page.getByText('历史项目')).toBeVisible()
  await page.screenshot({ path: 'e2e/shots/projects.png' })

  // 2. 打开已有项目的工作台（已有完成的转写；按行文本定位，不依赖列表顺序）
  const row007 = page.locator('tr', { hasText: '007' }).first()
  await row007.getByRole('button', { name: '打开工作台' }).click()
  await expect(page.locator('.title')).toContainText('007')

  // 3. 片段加载：段头行时间码（mm:ss）+ 整句行正文存在（句级布局）
  await expect(page.locator('.segment-head .timecode')).toContainText('00:00', { timeout: 10_000 })
  // 句级布局下首段可能拆出多个句行：定位符加 .first() 避免 strict mode violation
  await expect(page.locator('.sentence .text').first()).toContainText('电影')
  await page.screenshot({ path: 'e2e/shots/workbench-loaded.png' })

  // 4. 直接编辑句行（单击正文即编辑，无需编辑按钮；防抖 600ms -> 已保存）
  const editText = page.locator('.sentence .text').first()
  await editText.click()
  await editText.fill('说起电影，在座的各位一定都不陌生（编辑测试）。')
  await expect(page.locator('.segment-head .save-state').first()).toContainText('已保存', { timeout: 8_000 })
  await expect(page.locator('.sentence .text').first()).toContainText('编辑测试')

  // 5. 查找替换：预览 -> 应用 -> 撤销
  await page.getByRole('button', { name: '查找替换' }).click()
  await page.locator('.el-dialog input').first().fill('编辑测试')
  await page.getByRole('button', { name: '预览' }).click()
  await expect(page.locator('.hit')).toBeVisible({ timeout: 8_000 })
  await page.screenshot({ path: 'e2e/shots/find-preview.png' })
  await page.getByRole('button', { name: /全部替换/ }).click()
  await expect(page.getByText('已替换 1 处（可撤销）')).toBeVisible({ timeout: 8_000 })
  await page.getByRole('button', { name: '撤销上次替换' }).click()
  await expect(page.getByText(/已撤销，恢复 1 处/)).toBeVisible({ timeout: 8_000 })
  await page.keyboard.press('Escape')

  // 6. 重新转写（SSE 进度 -> 完成后刷新片段）
  // 模型常驻后短音频 1–2s 就完成，不再断言「取消任务」按钮（可能一闪而过）；
  // 以创建任务请求 + 最终完成态作为链路成功的判据。
  const rerunPromise = page.waitForRequest(
    (r) => r.method() === 'POST' && r.url().includes('/runs'), { timeout: 10_000 })
  await page.getByRole('button', { name: '重新转写' }).click()
  await rerunPromise
  await expect(page.locator('.taskbar')).toBeVisible({ timeout: 10_000 })
  await page.screenshot({ path: 'e2e/shots/rerun-progress.png' })
  // 完成后：回到完成态（转写全链路最长 ~2 分钟）
  await expect(page.getByText('转写完成，可编辑与导出')).toBeVisible({ timeout: 150_000 })
  await expect(page.locator('.sentence .text').first()).toContainText('说起电影', { timeout: 10_000 })
  await page.screenshot({ path: 'e2e/shots/workbench-final.png' })
})
