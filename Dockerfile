# Dockerfile
# =============================================================================
# RespiraCare-ICU — Production Container
#
# HuggingFace Spaces runs Docker containers on port 7860 by default.
# This container starts the FastAPI server and exposes it on that port.
#
# Build:  docker build -t respiraCare-icu .
# Run:    docker run -p 7860:7860 respiraCare-icu
# Test:   curl http://localhost:7860/ping
# =============================================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# gcc is needed for some Python packages that compile C extensions
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker layer caching means this layer
# is only rebuilt when requirements.txt changes, not on every code change
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Create a non-root user for security
# HuggingFace Spaces requires non-root execution
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port 7860 — required by HuggingFace Spaces
EXPOSE 7860

# Health check — HuggingFace uses this to verify the container started
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/ping || exit 1

# Start the FastAPI server
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "30"]