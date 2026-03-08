# ─── Build stage ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Runtime stage ────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ app/
COPY scripts/ scripts/
COPY Data/ Data/

# Expose the API port
EXPOSE 8000

# Run setup pipeline then start the server.
# The setup pipeline is idempotent — it will skip work if artefacts exist.
CMD ["sh", "-c", "python -m scripts.setup_pipeline && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
