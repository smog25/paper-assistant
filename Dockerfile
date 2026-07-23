# Backend image (FastAPI + OCR toolchain + SPECTER runtime).
# NOTE: the SPECTER model itself is NOT baked in — it downloads on first
# start into HF_HOME (persisted via the ./data bind mount in compose).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/data/hf-cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first: the default PyPI resolve on linux pulls the CUDA
# build plus nvidia-* wheels (~6 GB). Version must match requirements.lock.txt.
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY backend.py alembic.ini ./
COPY alembic/ alembic/
COPY app/ app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=4)"

# Tables come from Alembic only (no create_all anywhere in the app).
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
