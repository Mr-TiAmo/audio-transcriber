import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    vueDevTools(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  // 开发期同源代理：前端 5173 -> 本地后端 8730（绕过 CORS，音频 Range 同样经代理）
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8730',
        changeOrigin: true,
      },
    },
  },
})
