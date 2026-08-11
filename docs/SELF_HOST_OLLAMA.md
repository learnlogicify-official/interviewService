# Self-host / bring-your-own LLM for Nex AI Interview
#
# `interview-service` talks OpenAI Chat Completions. Any compatible server works:
# Ollama, vLLM, LocalAI, llama.cpp server, LiteLLM, OpenRouter, Groq, etc.

## Quick map

| Goal | What to do |
|------|------------|
| Fastest path | OpenAI `gpt-4o-mini` (paid API) |
| Self-host on Railway (CPU demo) | Deploy `deploy/ollama` + small model `llama3.2:3b` |
| Self-host for real quality | GPU box (RunPod / Vast / local) running Ollama/vLLM; Railway only runs interview-service |

Voice interviews need ~1–2s replies. CPU Ollama on Railway is OK for demos, not ideal for production.

---

## A) Wire interview-service (always)

On the **interview-service** Railway service, set:

### OpenAI (default)

```text
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

### Self-hosted Ollama (OpenAI-compatible)

```text
OPENAI_API_KEY=ollama
OPENAI_MODEL=llama3.2:3b
OPENAI_BASE_URL=https://YOUR-OLLAMA.up.railway.app/v1
```

If both services are in the same Railway project and private networking is on:

```text
OPENAI_BASE_URL=http://ollama.railway.internal:${PORT}/v1
```

Use the private hostname Railway shows for the Ollama service. Private ports are often `11434` if you fixed Ollama there; if Ollama binds `$PORT`, use that port.

Check:

```bash
curl -s https://YOUR-INTERVIEW.up.railway.app/v1/health
# {"ok":true,"llm_configured":true,"llm_model":"llama3.2:3b",...}
```

---

## B) Deploy Ollama on Railway (CPU demo)

1. Create a **new Railway service** in the same project.
2. Connect this repo (or copy `deploy/ollama`).
3. Set **Root Directory** to `deploy/ollama` (Dockerfile deploy).
4. Variables:

| Variable | Example | Notes |
|----------|---------|--------|
| `OLLAMA_MODEL` | `llama3.2:3b` | Keep ≤3B on small Railway RAM |
| `PORT` | (Railway sets) | start.sh binds Ollama to `$PORT` |

5. Give the service **≥8 GB RAM** if possible (3B model). 32B+ will not fit.
6. First deploy pulls the model — expect several minutes; watch logs for `Ollama ready`.
7. Public URL health: `https://YOUR-OLLAMA.up.railway.app/api/tags`

Then point interview-service `OPENAI_BASE_URL` at `https://YOUR-OLLAMA.up.railway.app/v1`.

### Suggested CPU models

| Model | RAM ballpark | Use |
|-------|----------------|-----|
| `llama3.2:1b` | ~2–4 GB | Smoke test only |
| `llama3.2:3b` | ~4–8 GB | Demo interviewer |
| `qwen2.5:3b` | ~4–8 GB | Decent small alt |
| `llama3.1:8b` | ~10–16 GB | Only if Railway plan allows |

---

## C) Better self-host (GPU) — recommended if not using OpenAI

1. Rent a GPU (RunPod / Vast / Lambda) or use a local machine with NVIDIA.
2. Install [Ollama](https://ollama.com) and run:

```bash
ollama pull llama3.1:8b
ollama serve   # or system service
```

Expose HTTPS (Caddy/nginx + TLS) or use a tunnel.

3. On Railway **interview-service** only:

```text
OPENAI_API_KEY=ollama
OPENAI_MODEL=llama3.1:8b
OPENAI_BASE_URL=https://your-gpu-host/v1
```

Keep Postgres + FastAPI on Railway; keep the heavy model off Railway CPU.

---

## D) Local Docker smoke test

From repo root `interview-service/`:

```bash
docker build -t nex-interview-ollama -f deploy/ollama/Dockerfile deploy/ollama
docker run --rm -p 11434:11434 -e PORT=11434 -e OLLAMA_MODEL=llama3.2:3b nex-interview-ollama
```

In another terminal:

```bash
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1
export OPENAI_MODEL=llama3.2:3b
export SHARED_SECRET=dev-change-me-interview-secret
uvicorn app.main:app --reload --port 8091
curl -s http://127.0.0.1:8091/v1/health
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `llm_configured: false` | Set non-empty `OPENAI_API_KEY` (use `ollama` for local) |
| 404 on `/v1/chat/completions` | Base URL must end with `/v1`, not bare host |
| OOM / killed on Railway | Smaller `OLLAMA_MODEL` or more RAM |
| Slow voice replies | Use GPU host or hosted API (`gpt-4o-mini` / Groq) |
| JSON parse errors from tiny models | Prefer ≥3B instruct models; `gpt-4o-mini` is more reliable for JSON turns |
