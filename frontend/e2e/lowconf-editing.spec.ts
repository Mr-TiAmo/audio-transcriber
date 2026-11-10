// 低置信标亮在编辑态的可见性回归：单击句行进入编辑后 <mark class="lo"> 不能消失
// （用户反馈：进入修改状态后标识不见，不利于对照修改）
import { expect, test } from '@playwright/test'

const BASE = process.env.E2E_BASE_URL ?? 'http://127.0.0.1:8730'
const BENCH_PROJECT = 'prj_cb744b5145a7' // bench_10min，最新转写含 3 段低置信标亮

test('单击句行进入编辑态后低置信标亮仍可见', async ({ page }) => {
  await page.goto(`${BASE}/workbench/${BENCH_PROJECT}`)

  const row = page.locator('.sentence .text').filter({ has: page.locator('mark.lo') }).first()
  await expect(row).toBeVisible({ timeout: 15_000 })
  const marksBefore = await row.locator('mark.lo').count()
  expect(marksBefore).toBeGreaterThan(0)

  // 单击即进入编辑（contenteditable focus）
  await row.click()
  await expect(row).toBeFocused()
  // 标亮不应被剥离
  expect(await row.locator('mark.lo').count()).toBe(marksBefore)
  await expect(row.locator('mark.lo').first()).toBeVisible()

  // 失焦：文本未改，标亮保持
  await page.locator('.taskbar').click({ position: { x: 5, y: 5 } })
  await expect(row.locator('mark.lo').first()).toBeVisible()
})