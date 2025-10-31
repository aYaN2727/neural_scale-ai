FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# Install system dependencies needed by scientific/python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
       git \
       libgomp1 \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests first for better layer caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8080

# Start the FastAPI app with uvicorn on port 8080
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]


