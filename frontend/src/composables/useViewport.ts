// 视口断点（与 CSS 媒体查询保持一致）：
// 手机 ≤640px、平板 641–1023px、PC ≥1024px。
// 仅用于无法用 CSS 表达的场景（如 el-table 列的显隐，CSS 隐藏会留下列宽空洞）。
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

export function useViewport() {
  const width = ref(typeof window === 'undefined' ? 1280 : window.innerWidth)
  const isMobile = computed(() => width.value <= 640)
  const isTablet = computed(() => width.value > 640 && width.value <= 1023)

  function onResize() {
    width.value = window.innerWidth
  }

  onMounted(() => window.addEventListener('resize', onResize))
  onBeforeUnmount(() => window.removeEventListener('resize', onResize))

  return { width, isMobile, isTablet }
}