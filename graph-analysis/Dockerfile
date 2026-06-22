# Reproducible image for graph-analysis.
# The pure-numpy graph core and its known-answer tests need only the slim
# Python deps. GraphFrames at scale runs on a Spark cluster separately; this
# image is for the core and the test suite, not for distributed jobs.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project and install it without re-resolving deps.
COPY . /app
RUN pip install --no-deps -e .

# Default: run the test suite (the pure-numpy known-answer graph tests).
CMD ["pytest", "-q"]
