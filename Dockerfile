# ============================================================
# Aegis AI — FastAPI Backend Dockerfile
# Multi-stage build: python:3.11-slim base
# Exposes port 8000
# ============================================================

# ---- Stage 1: Build dependencies ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt && \
    pip install --prefix=/install --no-cache-dir uvicorn[standard] fastapi

# ---- Stage 2: Runtime image ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

# Create data directories that the app writes to
RUN mkdir -p data/logs data/knowledge_base

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
