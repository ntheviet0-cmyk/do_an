import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: { '/api': process.env.M5_API_PROXY || 'http://127.0.0.1:8000' } },
  preview: { proxy: { '/api': process.env.M5_API_PROXY || 'http://127.0.0.1:8000' } },
  build: { rollupOptions: { output: { manualChunks: { charts: ['echarts'], vendor: ['react', 'react-dom', 'react-router-dom', '@tanstack/react-query'] } } } },
})
