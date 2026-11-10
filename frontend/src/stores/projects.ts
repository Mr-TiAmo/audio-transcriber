// 项目列表 store（导入 / 历史 / 删除）
import { defineStore } from 'pinia'
import { ref } from 'vue'

import { api } from '@/api/client'
import type { ImportedProject, ProjectListItem } from '@/types'

export const useProjectsStore = defineStore('projects', () => {
  const projects = ref<ProjectListItem[]>([])
  const loading = ref(false)
  const importing = ref(false)

  async function refresh() {
    loading.value = true
    try {
      projects.value = await api.listProjects()
    } finally {
      loading.value = false
    }
  }

  async function importAudio(file: File): Promise<ImportedProject> {
    importing.value = true
    try {
      const created = await api.createProject(file)
      await refresh()
      return created
    } finally {
      importing.value = false
    }
  }

  async function remove(projectId: string) {
    await api.deleteProject(projectId)
    await refresh()
  }

  return { projects, loading, importing, refresh, importAudio, remove }
})
