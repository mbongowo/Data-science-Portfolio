# Reproducible environment via pixi (conda-forge geospatial stack).
FROM ghcr.io/prefix-dev/pixi:latest

WORKDIR /app

# Resolve the environment first (better layer caching).
COPY pixi.toml pyproject.toml README.md ./
COPY src ./src
RUN pixi install --locked || pixi install

# Project sources & config.
COPY config ./config
COPY data ./data
COPY tests ./tests

# Default: run the pipeline against the bundled config.
ENTRYPOINT ["pixi", "run"]
CMD ["run"]
