// 响应式回归：大屏 PC / 平板 / 手机三种视口下页面不得出现横向溢出，
// 且手机端表格按预期压缩（隐藏导入时间列、按钮短文案）。
// 运行前提：后端 127.0.0.1:8730 已启动并完成前端构建。
import { expect, test } from '@playwright/test'

const BASE = process.env.E2E_BASE_URL ?? 'http://127.0.0.1:8730'
const WORKBENCH_PROJECT = 'prj_d508e9a1c029'

const VIEWPORTS = [
  { name: 'pc', width: 1440, height: 900 },
  { name: 'tablet', width: 834, height: 1112 },
  { name: 'phone', width: 390, height: 844 },
]

async function docWidth(page: import('@playwright/test').Page) {
  return page.evaluate(() => ({
    docW: document.documentElement.scrollWidth,
    winW: window.innerWidth,
  }))
}

for (const vp of VIEWPORTS) {
  test(`${vp.name} ${vp.width}px：列表页与工作台无横向溢出`, async ({ browser }) => {
    const page = await browser.newPage({ viewport: { width: vp.width, height: vp.height } })
    for (const path of ['/', `/workbench/${WORKBENCH_PROJECT}`]) {
      await page.goto(`${BASE}${path}`)
      // 等 store 拉完项目/转写数据（表格行或片段行渲染出来）
      await page.waitForTimeout(1000)
      const { docW, winW } = await docWidth(page)
      expect(docW, `${path} 横向溢出`).toBeLessThanOrEqual(winW + 1)
    }
    await page.close()
  })
}

test('手机端项目表格压缩：隐藏导入时间列、操作按钮短文案', async ({ browser }) => {
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } })
  await page.goto(`${BASE}/`)
  await expect(page.getByRole('heading', { name: 'AudioTranscriber 本地转写工作台' })).toBeVisible()
  await expect(page.getByRole('columnheader', { name: '导入时间' })).toHaveCount(0)
  await expect(page.getByRole('columnheader', { name: '名称' })).toBeVisible()
  if (await page.locator('.el-table__body tr').count()) {
    await expect(page.locator('.el-table__body tr').first().getByRole('button', { name: '打开' })).toBeVisible()
  }
  await page.close()
})

test('桌面端保留导入时间列', async ({ browser }) => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } })
  await page.goto(`${BASE}/`)
  await expect(page.getByRole('columnheader', { name: '导入时间' })).toBeVisible()
  await page.close()
})