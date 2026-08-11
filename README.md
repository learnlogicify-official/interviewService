# Nex AI Interview Service
#
# Local run:
#   cd interview-service
#   python -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   export SHARED_SECRET='same-as-moodle-plugin-setting'
#   uvicorn app.main:app --reload --port 8091
#
# Deploy (Railway / Railpack):
#   Start command is set in `railpack.json` / `Procfile`:
#     uvicorn app.main:app --host 0.0.0.0 --port $PORT
#   Set env vars on the host:
#     SHARED_SECRET=...   (must match Moodle plugin setting)
#     DATABASE_URL=...    (optional; defaults to sqlite ./interview.db)
#     OPENAI_API_KEY=...  (optional)
#
# Moodle `mod_aiinterview` talks to this API with HMAC signatures.
#
# Optional: OPENAI_API_KEY for polished phrasing (MVP works offline with heuristics).
