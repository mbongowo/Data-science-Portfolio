# syntax=docker/dockerfile:1
#
# Container image for the EO Explorer Streamlit app.
# Build:  docker build -t eo-explorer-app .
# Run:    docker run -p 8501:8501 eo-explorer-app
#
FROM python:3.12-slim

# System libraries needed by rasterio / GDAL raster handling, plus git so pip
# can install eo-monitor from this repo.
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

# Install Python dependencies first for better layer caching. The deploy
# manifest (app/requirements.txt) lists the runtime deps and installs eo-monitor
# from this repo, so no separate install step is needed.
COPY app/requirements.txt ./app/requirements.txt
RUN pip install --upgrade pip && pip install -r app/requirements.txt

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
