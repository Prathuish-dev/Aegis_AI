# ============================================================
# Aegis AI — FastAPI Backend Dockerfile (single-stage)
# Uses lean requirements.api.txt — no torch/sentence-transformers
# Exposes port 8000
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# Install build tools (needed for some packages like grpcio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.api.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.api.txt

# Copy application source
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

# Create runtime data directories
RUN mkdir -p data/logs data/chroma_db data/knowledge_base

# Environment defaults (override via docker-compose .env)
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FIX_MODE=suggest \
    LOG_DB_PATH=data/logs/events.db \
    CHROMA_PERSIST_DIR=data/chroma_db \
    PORT=8000

EXPOSE 8000

# Health-check: poll /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
