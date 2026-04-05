# ── Customer Data Cleaning — OpenEnv ──────────────────────────────────────────
# Hugging Face Spaces compatible Dockerfile (runs as non-root user 1000)
# Build: docker build -t dataclean-env .
# Run:   docker run -p 7860:7860 dataclean-env

FROM python:3.11-slim

# HF Spaces requires non-root
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY --chown=appuser:appuser . .

USER appuser

# Hugging Face Spaces uses port 7860
EXPOSE 7860

ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import requests; r=requests.get('http://localhost:7860/health',timeout=5); r.raise_for_status()"

CMD ["python", "app.py"]
