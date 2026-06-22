# Reproducible image for log-anomaly.
# The pure-numpy core (templating, features, PCA / z-score detectors, metrics)
# runs here; full Spark parsing at scale runs separately via pixi/conda-forge,
# which ships the JVM pyspark needs.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project and install it without re-resolving deps.
COPY . /app
RUN pip install --no-deps -e .

# Default: run the test suite (the pure-numpy known-answer tests).
CMD ["pytest", "-q"]
