# Nex AI Interview Service
#
# Local run:
#   cd interview-service
#   python -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   cp .env.example .env   # edit secrets / LLM settings
#   uvicorn app.main:app --reload --port 8091
#
# Deploy (Railway / Railpack):
#   Start command is set in `railpack.json` / `Procfile`:
#     uvicorn app.main:app --host 0.0.0.0 --port $PORT
#   Set env vars on the host:
#     SHARED_SECRET=...          (must match Moodle plugin setting)
#     DATABASE_URL=...           (Railway Postgres is fine; sqlite used if unset)
#     OPENAI_API_KEY=...         (REQUIRED for dynamic AI — any non-empty token)
#     OPENAI_MODEL=gpt-4o-mini   (or llama3.2:3b for Ollama)
#     OPENAI_BASE_URL=https://api.openai.com/v1
#     GLADIA_API_KEY=...         (recommended — realtime STT via app.gladia.io)
#     STT_PROVIDER=auto          (auto|gladia|openai)
#
# Self-host LLM (Ollama on Railway or GPU):
#   See docs/SELF_HOST_OLLAMA.md and deploy/ollama/
#
# Without OPENAI_API_KEY the service falls back to a small static question bank
# (questions will repeat). With the key, each turn is generated from the live transcript.
#
# Moodle `mod_aiinterview` talks to this API with HMAC signatures.
#
# Check: GET /v1/health → {"ok":true,"llm_configured":true,"llm_model":"..."}
