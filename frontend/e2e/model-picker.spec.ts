// 模型下拉框回归（任务栏）：默认 0.6B，可切换 1.7B，选中值随创建任务请求下发
// 运行前提：后端 127.0.0.1:8730 已启动，且 models/ 下两个模型均已安装
import { expect, test } from '@playwright/test'

const BASE = process.env.E2E_BASE_URL ?? 'http://127.0.0.1:8730'
const SMALL_PROJECT = 'prj_796c4f239c22' // 8.4s 中文语音，已有完成转写

test('任务栏模型下拉：默认 0.6B、切换 1.7B 后请求携带 model', async ({ page }) => {
  // 上一个用例可能留下正在终止的任务，先等空闲再发起新任务
  await expect.poll(async () => {
    const resp = await page.request.get(`${BASE}/api/v1/tasks/current`)
    return (await resp.json()).status
  }, { timeout: 30_000 }).toBe('IDLE')

  await page.goto(`${BASE}/workbench/${SMALL_PROJECT}`)

  const select = page.locator('.model-picker')
  await expect(select).toBeVisible()
  await expect(select).toContainText('0.6B')

  // 展开：两个模型都可选（未安装会置灰）
  await select.click()
  const items = page.locator('.el-select-dropdown__item:visible')
  await expect(items).toHaveCount(2)
  await expect(items.nth(1)).toContainText('1.7B')
  await expect(items.nth(1)).not.toHaveClass(/is-disabled/)
  await items.nth(1).click()
  await expect(select).toContainText('1.7B')

  // 重新转写：请求体应带 {"model":"1.7b"}，快速模式默认关闭
  const reqPromise = page.waitForRequest(
    (r) => r.method() === 'POST' && r.url().includes('/runs'), { timeout: 10_000 })
  await page.getByRole('button', { name: /重新转写|开始转写/ }).click()
  const req = await reqPromise
  expect(JSON.parse(req.postData() ?? '{}')).toMatchObject({ model: '1.7b', fast: false })

  // 立即取消，避免测试等待完整转写
  const cancelBtn = page.getByRole('button', { name: '取消任务' })
  if (await cancelBtn.isVisible().catch(() => false)) {
    await cancelBtn.click()
  }
})