# OCT Agent

基于 LangGraph 的多智能体科研助手，面向 **OCT（光学相干断层成像）应变估计**。
它把科学计算、联网文献调研、本地知识库检索整合进同一个对话式界面：用户用自然语言提问、上传 `.mat` 附件，结果以聊天消息 + 右侧交互式热力图面板的形式呈现。

> 主体验是「自然语言对话 + 附件 + artifact 结果面板」，能力通过 LangGraph 子图横向扩展，而非单一的应变计算工具。

---

## 功能概览

| 子图 | 作用 | 触发方式 |
|---|---|---|
| **Chat** | 通用问答、OCT 概念解释、长期记忆管理 | 记忆命令（`记住` / `忘记` / `查看记忆` / `清除摘要`），或检索闸门判定为闲聊时兜底 |
| **Strain Estimation** | OCT 应变计算，LLM 自主决定调用哪种方法 | 出现「计算 / 估计 / 运行…」等动作意图词，或已上传 `.mat` 文件 |
| **Deep Research** | 7 步深度研究流水线，输出带来源的 Markdown 综述 | 前端 `Deep Research` 按钮，或「文献 / 综述 / 调研 / 最新…」等意图词 |
| **Self-RAG** | 本地论文 / 笔记知识库的混合检索问答 | 「知识库 / 本地检索 / 论文库…」等关键词，或作为路由兜底 |

### 应变计算方法

LLM 根据用户选择的开关，自主决定调用以下一种或多种工具（可并行）：

- `vector_method_g` —— 基于 SciPy `convolve2d` 的矢量法（参数 `Nx` / `Nz` / `g`）
- `cnn_method` —— PyTorch UNet 推理（`assets/cnn/Unet.py` + `assets/cnn/model.pth`）
- `bnn_method` —— 贝叶斯 UNet++ + MC dropout，额外输出认知不确定性（`assets/bnn/bunetPP.py`，参数 `MC_test`）

### Deep Research 流水线

`澄清 → 研究简报 → 计划 → 生成查询 → 多源检索 → 压缩 → 报告`。
信息源默认 Tavily（缺 `TAVILY_API_KEY` 时降级并返回清晰中文提示，不会让图崩溃），并行检索本地知识库后合并去重；多个子问题在 `RESEARCH_BATCH_TIMEOUT_SECONDS`（默认 120s）总超时内并发完成，超时的子问题会在报告中标注缺失。范围不清时先返回澄清问题（`research_pending=True`），用户补充后续接。

### Self-RAG 知识库引擎

`backend/src/self_rag_engine/` 是一个相对完整的 RAG 子系统：
- 文档解析：Docling / GROBID / pdfplumber / python-docx（支持 `.pdf` `.docx` `.md` `.txt`）
- 中文友好的清洗、分块（parent-child）、乱码修复
- 混合检索：ChromaDB 向量 + BM25（jieba 分词）+ 重排序
- 引用对齐与来源标注

---

## 技术栈

- **后端**：Python ≥ 3.11、LangGraph、LangChain、FastAPI、PyTorch、SciPy、ChromaDB、由 [`uv`](https://docs.astral.sh/uv/) 管理依赖
- **前端**：React 19 + TypeScript + Vite，`@langchain/langgraph-sdk` 流式对话，Plotly 渲染热力图
- **LLM Provider**：按整组优先级选择 —— SiliconFlow（硅基流动，主路径）> Groq > OpenAI 兼容

---

## 快速开始

### 1. 配置环境变量

在项目根目录创建 `.env`：

```bash
# LLM（优先级：SiliconFlow > Groq > OpenAI；只需配置其中一组）
SILICONFLOW_API_KEY=
SILICONFLOW_API_BASE=https://api.siliconflow.cn/v1
SILICONFLOW_API_MODEL=Qwen/Qwen2.5-72B-Instruct

# 联网检索（deep_research 需要，缺失时优雅降级）
TAVILY_API_KEY=

# GPU 推理
INFERENCE_DEVICE=auto          # auto | cpu | cuda:0
MIN_FREE_GPU_MEMORY_GB=2

# 安全：是否允许直接传入本地文件路径
ALLOW_LOCAL_FILE_PATHS=false
```

### 2. 启动后端

```powershell
cd backend
uv sync                    # 安装依赖
uv run langgraph dev       # 在 localhost:2024 启动
```

LangGraph server 读取 `backend/langgraph.json`，暴露 `agent` 图，并挂载 `src/agent/app.py` 中的 FastAPI（文件上传 / 结果下载 / 知识库接口）。

> **Windows 端口占用**：若上次 dev server 退出后 `2024` 仍被保留，换端口启动 `uv run langgraph dev --port 2025`，前端对应设置 `$env:VITE_API_URL="http://localhost:2025"`。

### 3. 启动前端

```powershell
cd frontend
npm install
npm run dev                # 在 localhost:5173 启动
```

### 一键启动（两端同时）

```powershell
make dev
```

---

## 测试与构建

```powershell
cd backend && uv run pytest                                   # 全部后端测试
uv run pytest tests/test_services.py::test_vector_method_shape # 单个测试
cd frontend && npm run build                                  # 前端生产构建
```

---

## 项目结构

```
OCT_Agent/
├── assets/                      # 模型权重与网络定义
│   ├── cnn/Unet.py + model.pth
│   └── bnn/bunetPP.py + model.pth
├── backend/
│   ├── langgraph.json           # 图入口 + FastAPI 挂载 + .env 路径
│   ├── src/agent/
│   │   ├── graph.py             # Supervisor + 4 个子图编排
│   │   ├── prompts.py           # 所有 system prompt（集中管理，禁止内联）
│   │   ├── tools.py             # @tool 应变计算函数
│   │   ├── schemas.py           # OctGraphState 及全部 TypedDict
│   │   ├── config.py            # LLM provider 选择
│   │   ├── app.py               # FastAPI：上传 / 结果 / 知识库接口
│   │   ├── self_rag.py          # 内置知识库适配层
│   │   ├── research/            # deep research 子图
│   │   └── services/            # mat_io / memory / models / paths / storage
│   ├── src/self_rag_engine/     # 独立的混合检索 RAG 引擎
│   └── tests/
├── frontend/
│   └── src/
│       ├── App.tsx              # 主组件
│       ├── components/          # 文件上传 / 方法面板 / 结果画廊 / 热力图等
│       ├── lib/api.ts           # 后端 REST 调用封装
│       └── types/api.ts         # 前后端共享类型
└── Makefile
```

---

## 开发约定

- **新增应变方法**：在 `tools.py` 用 `@tool` 实现并加入 `TOOLS`，更新 `prompts.py` 的 `build_strain_prompt()`，补充测试。
- **修改提示词**：只改 `prompts.py` 的 builder 函数，**不要**在图节点里内联大段 system prompt。
- **前端 API 变更**：先改 `frontend/src/types/api.ts`，再改 `frontend/src/lib/api.ts`。
- **结果可视化**：前端通过 `GET /api/results/{result_id}/array?name=strain` 取数组并用 Plotly 渲染。

更多面向 Claude Code 的工作指引见 [`CLAUDE.md`](./CLAUDE.md)，设计决策记录见 [`AGENT.md`](./AGENT.md)。
