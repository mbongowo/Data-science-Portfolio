# Reproducible image for dbt-modern-stack.
# Slim Python base; the data-quality core needs only numpy/pandas/pyyaml.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project and install it without re-resolving deps.
COPY . /app
RUN pip install --no-deps -e .

# Default: run the pure-pandas data-quality known-answer test suite.
# Note: `dbt build` and the orchestrator (Airflow/Dagster) run OUTSIDE this
# image — they need a warehouse connection and a scheduler, not a CI container.
CMD ["pytest", "-q"]
