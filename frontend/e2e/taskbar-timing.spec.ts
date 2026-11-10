// 任务栏时间提示回归：处理中必须显示「已处理 X」与「预计还需约 Y」
// 运行前提：后端 127.0.0.1:8730 已启动；测试结束取消任务，不等待完整转写
import { expect, test } from '@playwright/test'

const BASE = process.env.E2E_BASE_URL ?? 'http://127.0.0.1:8730'
// 310s 语音：常驻 worker 下任务约 30s，足够观察两秒的计时刷新；
// 8.4s 短音频在模型常驻后 1–2s 就结束，计时元素会在断言前消失。
const LONG_PROJECT = 'prj_d508e9a1c029'

test('处理中显示已处理时长与预计剩余，且每秒刷新', async ({ page }) => {
  // 上一个用例（model-picker）取消后任务可能仍在终止中，先等空闲
  await expect.poll(async () => {
    const resp = await page.request.get(`${BASE}/api/v1/tasks/current`)
    return (await resp.json()).status
  }, { timeout: 30_000 }).toBe('IDLE')

  await page.goto(`${BASE}/workbench/${LONG_PROJECT}`)

  // 默认 0.6B，直接重新转写
  const reqPromise = page.waitForRequest(
    (r) => r.method() === 'POST' && r.url().includes('/runs'), { timeout: 10_000 })
  await page.getByRole('button', { name: /重新转写|开始转写/ }).click()
  await reqPromise

  const timing = page.locator('.timing')
  await expect(timing).toBeVisible({ timeout: 10_000 })
  await expect(timing).toContainText('已处理')
  await expect(timing).toContainText(/预计还需约|即将完成/)

  // 已处理时长每秒递增（读到相同文本可能刚好跨过同一秒，重试一次）
  const first = await timing.textContent()
  await expect(timing).not.toHaveText(first ?? '', { timeout: 3_000 })

  // 清理：取消任务，避免影响后续用例
  const cancelBtn = page.getByRole('button', { name: '取消任务' })
  if (await cancelBtn.isVisible().catch(() => false)) {
    await cancelBtn.click()
  }
})