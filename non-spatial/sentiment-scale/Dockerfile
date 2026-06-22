# Reproducible image for sentiment-scale.
# python:3.12-slim is enough for the pure-numpy/pandas core and the test suite.
# Spark NLP at scale (the pyspark batch + model-inference path) runs separately
# on a cluster; it is intentionally not part of this image.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project and install it without re-resolving the dependency tree.
COPY . /app
RUN pip install --no-deps -e .

# Default: run the test suite (the pure-numpy/pandas known-answer tests).
CMD ["pytest", "-q"]
