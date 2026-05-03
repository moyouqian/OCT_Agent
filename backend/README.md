# OCT Agent Backend

LangGraph + FastAPI backend for OCT strain estimation.

## Development

```powershell
cd backend
uv sync
uv run langgraph dev
```

The LangGraph server reads `langgraph.json`, exposes the `agent` graph, and mounts the FastAPI app from `src/agent/app.py`.

If Windows keeps `2024` reserved after a previous dev server exits, run the backend on another port:

```powershell
uv run langgraph dev --port 2025
```

Then start the frontend with the matching API URL:

```powershell
cd ../frontend
$env:VITE_API_URL="http://localhost:2025"
npm.cmd run dev
```

## Environment

- `GROQ_API_KEY`, `GROQ_API_BASE`, `GROQ_API_MODEL` for the notebook-compatible chat model.
- Or `OPENAI_API_KEY`, `OPENAI_API_BASE`, `MODEL`.
- `ALLOW_LOCAL_FILE_PATHS=false` by default. Set to `true` only for trusted local development.
