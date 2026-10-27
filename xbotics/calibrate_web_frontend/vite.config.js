import { defineConfig } from 'vite';

// 构建产物直接进 Python 包，随 lerobot 一起装；SO101 的 URDF / STL 由后端从包里的 model/ 目录出，不复制进构建产物。
export default defineConfig({
  base: './',
  publicDir: false,
  build: {
    outDir: '../../src/lerobot/xbotics/calibrate_web/static',
    emptyOutDir: true,
    target: 'es2020',
  },
  server: {
    host: '127.0.0.1',
    proxy: {
      '/api': 'http://127.0.0.1:8006',
      '/model': 'http://127.0.0.1:8006',
    },
  },
});
