# Reproducible image for als-recommender.
# The pure-numpy core (ALS, ranking metrics, popularity baseline, split) and its
# known-answer tests need only numpy + pandas + pyyaml + pytest, so the default
# image stays small and does not pull in a JVM.
#
# Spark MLlib ALS training (recsys.spark_als) runs separately: it needs a JVM
# and PySpark, which are best provided by the pixi/conda-forge environment or a
# dedicated Spark image. It is intentionally not part of this test image.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir numpy pandas pyyaml pytest ruff mypy

# Copy the project and install it without re-resolving dependencies.
COPY . /app
RUN pip install --no-deps -e .

# Default: run the pure-numpy known-answer test suite.
CMD ["pytest", "-q"]
