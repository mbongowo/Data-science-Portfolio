# Reproducible image for clickstream-pipeline.
# A plain python:3.12-slim base is enough: this image runs the pure-Python core
# and its test suite. The full streaming stack (Kafka + Spark, plus a JVM) is
# heavy and is meant to run via docker-compose, not from this single image.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project and install it without re-resolving deps.
COPY . /app
RUN pip install --no-deps -e .

# Default: run the test suite (the pure-Python known-answer tests).
CMD ["pytest", "-q"]
