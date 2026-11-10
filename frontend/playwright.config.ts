// Playwright 配置：本地无头冒烟（非 CI 门禁，阶段 2 验收辅助）
import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  timeout: 180_000,
  use: {
    headless: true,
    // 与 spec 内 E2E_BASE_URL 保持一致：默认 dev 5173，可切换生产 8730
    baseURL: process.env.E2E_BASE_URL ?? 'http://localhost:5173',
    locale: 'zh-CN',
  },
  outputDir: './e2e/results',
  retries: 0,
  workers: 1,
})
