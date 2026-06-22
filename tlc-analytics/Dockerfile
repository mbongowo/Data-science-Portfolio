# Reproducible image for tlc-analytics.
# Slim Python base: this image exercises the pure-pandas reference mart and the
# known-answer test suite. Spark and warehouse runs happen outside this image
# (a Spark cluster / managed warehouse), so pyspark and duckdb are not baked in.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir numpy pandas pyyaml pyarrow pytest ruff mypy

# Copy the project and install it without pulling the engine extras.
COPY . /app
RUN pip install --no-deps -e .

# Default: run the test suite (the pure-pandas known-answer mart tests).
CMD ["pytest", "-q"]
