# syntax=docker/dockerfile:1
#
# Container image for the EO Explorer Streamlit app.
# Build:  docker build -t eo-explorer-app .
# Run:    docker run -p 8501:8501 eo-explorer-app
#
FROM python:3.12-slim

# System libraries needed by rasterio / GDAL / leafmap raster handling.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
        libgl1 \
        git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install the reused flagship index package (eo-monitor) from git.
# In CI/deploy this lets indices work without a sibling checkout.
RUN pip install "eo-monitor @ git+https://github.com/JosephMbuh/eo-monitor.git" \
    || echo "WARNING: eo-monitor install failed; index computation will be disabled at runtime."

# Copy the application code.
COPY pyproject.toml README.md ./
COPY app ./app
COPY .streamlit ./.streamlit

# Install the app itself (so `from app import ...` resolves cleanly).
RUN pip install --no-deps -e .

EXPOSE 8501

# Simple healthcheck against Streamlit's built-in endpoint.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
