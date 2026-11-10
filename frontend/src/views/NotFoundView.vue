<script setup lang="ts">
// 404 兜底页：未知路径不再白屏（本地应用手输 URL / 旧书签场景）
import { useRouter } from 'vue-router'

const router = useRouter()
</script>

<template>
  <div class="notfound">
    <el-result icon="warning" title="页面不存在" sub-title="请检查地址栏路径是否正确">
      <template #extra>
        <el-button type="primary" @click="router.replace('/')">返回项目列表</el-button>
      </template>
    </el-result>
  </div>
</template>

<style scoped>
.notfound {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
