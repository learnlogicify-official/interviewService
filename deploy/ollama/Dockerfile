# Self-hosted Ollama for Nex AI Interview (Railway / Docker).
# Prefer a small model on CPU (llama3.2:3b). For quality, run this on a GPU host
# and point interview-service OPENAI_BASE_URL at it.

FROM ollama/ollama:latest

ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_MODEL=llama3.2:3b

COPY start.sh /usr/local/bin/interview-ollama-start.sh
RUN chmod +x /usr/local/bin/interview-ollama-start.sh

EXPOSE 11434

ENTRYPOINT ["/usr/local/bin/interview-ollama-start.sh"]
