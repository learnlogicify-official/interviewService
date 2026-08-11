# Ollama sidecar for Nex AI Interview

Deploy this folder as its **own Railway service** (Dockerfile).

1. New service → Dockerfile  
2. Root directory: `deploy/ollama`  
3. Env: `OLLAMA_MODEL=llama3.2:3b` (≥8GB RAM recommended)  
4. Point **interview-service** at it:

```text
OPENAI_API_KEY=ollama
OPENAI_MODEL=llama3.2:3b
OPENAI_BASE_URL=https://<this-service>.up.railway.app/v1
```

Full guide: [../../docs/SELF_HOST_OLLAMA.md](../../docs/SELF_HOST_OLLAMA.md)
