# Reproducible image for crypto-backtest.
# The tested core (resampling, indicators, engine, performance) is pure
# numpy/pandas, so a slim Python base is enough to run the suite.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the project and install it without re-resolving dependencies.
COPY . /app
RUN pip install --no-deps -e .

# Large multi-symbol tick resampling runs via Polars / Spark separately
# (see src/backtest/scale_pipeline.py); it is not needed to run the tests.

# Default: run the known-answer test suite.
CMD ["pytest", "-q"]
