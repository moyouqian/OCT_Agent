# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OCT Agent is a LangGraph-based multi-agent research assistant for OCT (Optical Coherence Tomography) strain estimation. It combines scientific computation (strain analysis of .mat files), web research, and a local knowledge base in a single chat-driven interface.

## Commands

### Backend
```powershell
cd backend
uv sync                    # Install dependencies
uv run langgraph dev       # Start server on localhost:2024
uv run pytest              # Run all tests
uv run pytest tests/test_services.py::test_vector_method_shape  # Single test
```

### Frontend
```powershell
cd frontend
npm install
npm run dev                # Start Vite on localhost:5173
npm run build              # Production build to dist/
```

### Combined
```powershell
make dev                   # Start both backend and frontend
```

## Architecture

### Supervisor + 4 Subgraph Pattern

The core is a LangGraph supervisor in `backend/src/agent/graph.py` that routes each user message to one of four subgraphs:

1. **Chat** — General conversation + long-term memory (SQLite). Commands: `记住`, `忘记`, `查看记忆`.
2. **Strain Estimation** — Multi-step LLM-driven tool calling. The LLM decides which methods to invoke:
   - `vector_method_g`: SciPy-based convolution
   - `cnn_method`: PyTorch UNet (`assets/cnn/Unet.py`)
   - `bnn_method`: Bayesian UNet++ with MC dropout (`assets/bnn/bunetPP.py`)
3. **Deep Research** — 7-step pipeline: clarify → brief → plan → query → search (Tavily) → compress → report
4. **Self-RAG** — ChromaDB + BM25 (jieba for Chinese) hybrid retrieval from local docs

Routing uses `requested_sub_agent` (explicit, e.g., from a UI button) or keyword matching from user text (both Chinese and English keywords).

### State Flow

`OctGraphState` (defined in `backend/src/agent/schemas.py`) carries all state across nodes:
- `messages`: append-only chat history
- `file_ids`: uploaded .mat file UUIDs
- `result_refs`: accumulated computation results (append-only via `operator.add`)
- `run_dir`: timestamped directory for current computation run
- `research_*`: fields used by the deep research pipeline
- `sub_agent` / `requested_sub_agent`: routing control

### File Storage

- `.mat` uploads → `backend/data/uploads/{file_id}/`
- Computation results → `backend/data/runs/{timestamp}_{run_name}/`
- Index files: `backend/data/uploads.json`, `backend/data/results.json`
- Knowledge base: `backend/data/self_rag/` (ChromaDB + SQLite)
- Long-term memory: SQLite in-process (via `backend/src/agent/services/memory.py`)

### LangGraph Server Config

`backend/langgraph.json` registers:
- Graph entry point: `./src/agent/graph.py:graph`
- FastAPI app (file upload/download endpoints): `./src/agent/app.py:app`
- Env file: `../.env`

## Key Files

| File | Purpose |
|---|---|
| `backend/src/agent/graph.py` | Supervisor + all 4 subgraphs wired together |
| `backend/src/agent/prompts.py` | All system prompts (centralized, use builder functions) |
| `backend/src/agent/tools.py` | `@tool`-decorated strain computation functions |
| `backend/src/agent/schemas.py` | `OctGraphState` and all TypedDict definitions |
| `backend/src/agent/config.py` | LLM provider selection (SiliconFlow → Groq → OpenAI fallback) |
| `backend/src/agent/app.py` | FastAPI endpoints for upload, results, knowledge base |
| `backend/src/self_rag_engine/graph.py` | RAG retrieval pipeline |
| `backend/src/self_rag_engine/ingestion.py` | Document parsing & ChromaDB/BM25 indexing |
| `frontend/src/App.tsx` | Main React component |
| `frontend/src/lib/api.ts` | Fetch helpers for backend REST API |

## Environment Variables (.env)

```
# LLM (priority: SiliconFlow > Groq > OpenAI)
SILICONFLOW_API_KEY=
SILICONFLOW_API_BASE=https://api.siliconflow.cn/v1
SILICONFLOW_API_MODEL=Qwen/Qwen2.5-72B-Instruct

# Web search (required for deep_research; gracefully degraded if missing)
TAVILY_API_KEY=

# GPU inference
INFERENCE_DEVICE=auto        # auto | cpu | cuda:0
MIN_FREE_GPU_MEMORY_GB=2

# Security
ALLOW_LOCAL_FILE_PATHS=false
```

## Development Patterns

- **Adding a strain method**: Implement in `tools.py` with `@tool`, add to the `TOOLS` list, update `build_strain_prompt()` in `prompts.py`, add a test.
- **Modifying prompts**: Edit builder functions in `prompts.py` only — never inline prompts in graph nodes.
- **Frontend API changes**: Update `frontend/src/types/api.ts` first, then `frontend/src/lib/api.ts`.
- **Result visualization**: Frontend fetches array data via `GET /api/results/{result_id}/array?name=strain` and renders with Plotly.
