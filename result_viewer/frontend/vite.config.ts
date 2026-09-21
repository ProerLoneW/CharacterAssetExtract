import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base: './' 使构建产物以相对路径引用静态资源,
// 页面挂在任何平台代理前缀下都能正确加载(文档三 §9.2)。
export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    outDir: '../frontend_dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8930', changeOrigin: false },
      '/media': { target: 'http://127.0.0.1:8930', changeOrigin: false },
    },
  },
})
