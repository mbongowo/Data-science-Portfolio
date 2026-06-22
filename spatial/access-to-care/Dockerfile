# Reproducible environment for the access-to-care pipeline.
# Uses the pixi base image so the conda-forge geospatial stack (GDAL/GEOS/PROJ)
# resolves cleanly. `pixi install` reads pixi.toml (and the generated pixi.lock).
FROM ghcr.io/prefix-dev/pixi:latest

WORKDIR /app

# Install the environment first for better layer caching.
COPY pixi.toml pyproject.toml ./
COPY pixi.lock* ./
RUN pixi install --locked || pixi install

# Copy the project source.
COPY . .

# Default: run the test suite. Override the command to run the pipeline, e.g.
#   docker run --rm -it <image> pixi run download
#   docker run --rm -it <image> pixi run access data/raw/cameroon-latest.osm.pbf data/raw/healthsites_cameroon.geojson
CMD ["pixi", "run", "test"]
