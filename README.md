# OCT Agent

A LangGraph-based multi-agent research assistant for **OCT (Optical Coherence Tomography)**.
It unifies scientific computation, web-based literature research, and a local knowledge base behind a single conversational interface: users ask in natural language, attach `.mat` files, and receive results as chat messages plus an interactive heatmap panel on the right.

> The core experience is "natural-language chat + attachments + an artifact result panel". Capabilities scale horizontally via LangGraph subgraphs, not as a single monolithic strain-computation tool.

---

## Features

| Subgraph | Purpose | Trigger |
|---|---|---|
| **Strain Estimation** | OCT strain computation; the LLM decides which method(s) to invoke | Action-intent keywords like "compute / estimate / run…", or when a `.mat` file has been uploaded |
| **Deep Research** | 7-step deep research pipeline that outputs a cited Markdown report | The `Deep Research` button in the frontend, or intent keywords like "literature / review / survey / latest…" |
| **Self-RAG** | Hybrid retrieval Q&A over a local paper / notes knowledge base | Keywords like "knowledge base / local search / paper library…", or as the default routing fallback |

### Strain Computation Methods

Based on the toggles selected by the user, the LLM autonomously decides which of the following tools to invoke (in parallel when appropriate):

- `vector_method_g` — vector method based on SciPy `convolve2d` (parameters `Nx` / `Nz` / `g`)
- `cnn_method` — PyTorch UNet inference (`assets/cnn/Unet.py` + `assets/cnn/model.pth`)
- `bnn_method` — Bayesian UNet++ with MC dropout, additionally outputting epistemic uncertainty (`assets/bnn/bunetPP.py`, parameter `MC_test`)

### Deep Research Pipeline

`Clarify → Research brief → Plan → Generate queries → Multi-source retrieval → Compress → Report`.
The default information source is Tavily (when `TAVILY_API_KEY` is missing, it gracefully degrades and returns a clear message instead of crashing the graph). The local knowledge base is searched in parallel and results are merged and deduplicated. Multiple sub-questions run concurrently within a `RESEARCH_BATCH_TIMEOUT_SECONDS` (default 120s) total timeout; sub-questions that time out are flagged as missing in the report. When the scope is unclear, the pipeline first returns a clarification question (`research_pending=True`) and resumes after the user replies.

### Self-RAG Knowledge Base Engine

`backend/src/self_rag_engine/` is a relatively complete RAG subsystem:
- Document parsing: Docling / GROBID / pdfplumber / python-docx (supports `.pdf` `.docx` `.md` `.txt`)
- Chinese-friendly cleaning, parent-child chunking, and mojibake repair
- Hybrid retrieval: ChromaDB vectors + BM25 (jieba tokenization) + reranking
- Citation alignment and source attribution

#### Retrieval Evaluation (2026-05-24)

Test set of 20 questions with 96 manually annotated relevant sections (33 core gold + 63 supporting); retrieval config: dense + BM25 + RRF, cloud rerank (Qwen3-Reranker-4B), top_k=30.

|  | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| **gold_recall** | 0.380 | 0.796 | 0.861 | 1.000 |
| **sup_recall** | 0.062 | 0.330 | 0.643 | 1.000 |
| **section_recall** | 0.175 | 0.478 | 0.713 | 1.000 |
| **any_hit** | 0.833 | 1.000 | 1.000 | 1.000 |
| **NDCG** (g=2, s=1) | 0.722 | 0.763 | 0.789 | 0.880 |
| **Precision** | 0.833 | 0.778 | 0.722 | — |

**MRR@10 = 0.917** ｜ **MAP = 0.827** ｜ End-to-end retrieval latency **3.6 s/query**

> recall@10 = 1.0 is a constructive result (all gold items come from top-10 annotations of the current build). Actual ranking quality should be judged by recall@1/3/5, MRR, and NDCG.

---

## Tech Stack

- **Backend**: Python ≥ 3.11, LangGraph, LangChain, FastAPI, PyTorch, SciPy, ChromaDB; dependencies managed by [`uv`](https://docs.astral.sh/uv/)
- **Frontend**: React 19 + TypeScript + Vite, `@langchain/langgraph-sdk` for streaming chat, Plotly for heatmap rendering
- **LLM Providers**: selected by group priority — SiliconFlow (primary) > Groq > OpenAI-compatible

---

## Quick Start

### 1. Configure Environment Variables

Create a `.env` in the project root:

```bash
# LLM (priority: SiliconFlow > Groq > OpenAI; only one group needs to be configured)
SILICONFLOW_API_KEY=
SILICONFLOW_API_BASE=https://api.siliconflow.cn/v1
SILICONFLOW_API_MODEL=Qwen/Qwen2.5-72B-Instruct

# Web search (required for deep_research; gracefully degraded if missing)
TAVILY_API_KEY=

# GPU inference
INFERENCE_DEVICE=auto          # auto | cpu | cuda:0
MIN_FREE_GPU_MEMORY_GB=2

# Security: whether to allow passing local file paths directly
ALLOW_LOCAL_FILE_PATHS=false
```

### 2. Start the Backend

```powershell
cd backend
uv sync                    # Install dependencies
uv run langgraph dev       # Start  the Backend
```

The LangGraph server reads `backend/langgraph.json`, exposes the `agent` graph, and mounts the FastAPI app from `src/agent/app.py` (file upload / result download / knowledge base endpoints).

> **Port conflict on Windows**: if `2024` remains held after the previous dev server exited, switch ports with `uv run langgraph dev --port 2025` and configure the frontend with `$env:VITE_API_URL="http://localhost:2025"`.

### 3. Start the Frontend

```powershell
cd frontend
npm install
npm run dev                # Start the Frontend
```

### One-Shot Start (Both Sides)

```powershell
make dev
```

---

## Testing & Build

```powershell
cd backend && uv run pytest                                   # All backend tests
uv run pytest tests/test_services.py::test_vector_method_shape # Single test
cd frontend && npm run build                                  # Frontend production build
```

---

## Project Structure

```
OCT_Agent/
├── assets/                      # Model weights and network definitions
│   ├── cnn/Unet.py + model.pth
│   └── bnn/bunetPP.py + model.pth
├── backend/
│   ├── langgraph.json           # Graph entry + FastAPI mount + .env path
│   ├── src/agent/
│   │   ├── graph.py             # Supervisor + 3 subgraph orchestration
│   │   ├── prompts.py           # All system prompts (centralized; no inlining)
│   │   ├── tools.py             # @tool strain computation functions
│   │   ├── schemas.py           # OctGraphState and all TypedDicts
│   │   ├── config.py            # LLM provider selection
│   │   ├── app.py               # FastAPI: upload / results / knowledge base endpoints
│   │   ├── self_rag.py          # Built-in knowledge base adapter
│   │   ├── research/            # Deep research subgraph
│   │   └── services/            # mat_io / memory / models / paths / storage
│   ├── src/self_rag_engine/     # Standalone hybrid-retrieval RAG engine
│   └── tests/
├── frontend/
│   └── src/
│       ├── App.tsx              # Main component
│       ├── components/          # File upload / method panel / result gallery / heatmap, etc.
│       ├── lib/api.ts           # Backend REST call wrappers
│       └── types/api.ts         # Shared frontend/backend types
└── Makefile
```
