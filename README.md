# Nex AI Interview Service
#
# Local run:
#   cd interview-service
#   python -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   export SHARED_SECRET='same-as-moodle-plugin-setting'
#   uvicorn app.main:app --reload --port 8091
#
# Moodle `mod_aiinterview` talks to this API with HMAC signatures.
#
# Optional: OPENAI_API_KEY for polished phrasing (MVP works offline with heuristics).
