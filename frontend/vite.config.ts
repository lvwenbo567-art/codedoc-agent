import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backendTarget =
  process.env.VITE_BACKEND_PROXY_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/langgraph": {
        target: backendTarget,
        changeOrigin: true,
      },
      "/agent-quality": {
        target: backendTarget,
        changeOrigin: true,
      },
      "/version": {
        target: backendTarget,
        changeOrigin: true,
      },
      "/health": {
        target: backendTarget,
        changeOrigin: true,
      },
      "/scan": {
        target: backendTarget,
        changeOrigin: true,
      },
      "/index": {
        target: backendTarget,
        changeOrigin: true,
      },
      "/projects": {
        target: backendTarget,
        changeOrigin: true,
      },
      "/files": {
        target: backendTarget,
        changeOrigin: true,
      },
      "/chunks": {
        target: backendTarget,
        changeOrigin: true,
      },
      "/project-upload": {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
});
