# ----------- STAGE 1: Builder -----------
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ----------- STAGE 2: Runtime -----------
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app.py .

# Expose FastAPI port
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
