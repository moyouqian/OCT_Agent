# OCT Agent Backend

OCT 应变估计的 LangGraph + FastAPI 后端。

完整的安装、配置、架构与开发说明见仓库根目录的 [`../README.md`](../README.md)。

## 快速启动

```powershell
cd backend
uv sync
uv run langgraph dev          # localhost:2024
```

LangGraph server 读取 `langgraph.json`，暴露 `agent` 图，并挂载 `src/agent/app.py` 的 FastAPI。


