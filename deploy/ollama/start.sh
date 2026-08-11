#!/bin/sh
# Start Ollama bound to Railway's $PORT, then pull the interview model.
set -eu

MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
PORT_VALUE="${PORT:-11434}"

export OLLAMA_HOST="0.0.0.0:${PORT_VALUE}"

echo "Starting Ollama on ${OLLAMA_HOST}"
ollama serve &
pid=$!

# Give the daemon time to bind (Railway cold start / first boot).
sleep 8

echo "Pulling model: ${MODEL} (first boot can take several minutes)"
# Retry pull a few times while the daemon finishes starting.
n=0
while [ "$n" -lt 10 ]; do
  if ollama pull "${MODEL}"; then
    echo "Ollama ready. Model=${MODEL}"
    wait "$pid"
    exit 0
  fi
  n=$((n + 1))
  echo "Pull attempt ${n} failed — retrying in 5s..."
  sleep 5
done

echo "ERROR: could not pull ${MODEL}. Check RAM/disk logs."
kill "$pid" 2>/dev/null || true
exit 1
