# Containerized FastAPI inference service for the rain-day model.
#
# This image is for YOU to build and run locally (or push to a registry); it is
# NOT built or run in CI. The trained model is expected at /app/outputs/model.pkl
# inside the container (mount it, bake it in, or point MODEL_PATH elsewhere).
#
#   # 1. train a model so outputs/model.pkl exists
#   mlpipe train --config config/config.yaml
#   # 2. build and run the service
#   docker build -t rainday-inference .
#   docker run --rm -p 8000:8000 -v "$(pwd)/outputs:/app/outputs" rainday-inference
#   # 3. probe it
#   curl localhost:8000/health

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/outputs/model.pkl

WORKDIR /app

# Only the dependencies the service itself needs (not the full heavy group).
RUN pip install --no-cache-dir \
    "numpy>=1.24" "pandas>=2.0" "fastapi>=0.110" "uvicorn>=0.27" "pydantic>=2.0"

COPY src/ ./src/
ENV PYTHONPATH=/app/src

# The service reads MODEL_PATH at startup; mount or copy outputs/model.pkl in.
EXPOSE 8000
CMD ["uvicorn", "mlpipe.serve:app", "--host", "0.0.0.0", "--port", "8000"]
